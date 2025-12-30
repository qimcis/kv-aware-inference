# building a kv cache playground from scratch: a deep dive into llm inference systems

hey everyone. i recently wrapped up a side project i've been poking at for a while: a kv-aware inference playground. basically, i wanted to strip away all the complexity of a real massive transformer and just focus on the systems stuff—specifically, how the kv cache works, how we store it on a gpu, and how different scheduling policies wreck (or save) your latency.

if you're into mlsys, you know that inference is basically a memory bandwidth problem disguised as a math problem. i built two main binaries for this: `kv_aware` (a c++/cuda synthetic engine) and `sim_main` (a token-time simulator). here's what i learned building it, with a deep dive into the arithmetic that makes or breaks your inference throughput.

## the kv cache: why we need it and what it costs

first, a quick refresher. when an llm generates tokens, it doesn't re-compute the attention for every previous token in the history every single step. that would be insanely slow. instead, we store the key (k) and value (v) vectors for previous tokens in a cache.

here's the math that made everything click for me. in a transformer, attention at layer `l` for generating token `t` needs to look at all previous tokens `0...t-1`. without a kv cache, you'd compute:

```
for each new token t:
  for each layer l:
    for each previous token i in [0, t-1]:
      compute key[i] = W_k @ x[i]
      compute value[i] = W_v @ x[i]
    compute attention using all keys and values
```

the problem? you're recomputing `W_k @ x[i]` and `W_v @ x[i]` for *every* previous token *every* time you generate a new token. for a 1000-token prompt, the 1001st token would recompute 1000 key-value pairs. the 1002nd would compute 1001 pairs. this scales as `O(n²)` in compute.

the kv cache solves this by storing those intermediate results:

```
# prefill phase (process entire prompt once)
for each token i in prompt:
  for each layer l:
    k[l][i] = W_k @ x[i]
    v[l][i] = W_v @ x[i]
    store k[l][i] and v[l][i] in cache

# decode phase (generate one token at a time)
for each new token t:
  for each layer l:
    k[l][t] = W_k @ x[t]  # only compute for NEW token
    v[l][t] = W_v @ x[t]
    store k[l][t] and v[l][t] in cache
    compute attention using cached k[l][0:t] and v[l][0:t]
```

now we're `O(n)` per token instead of `O(n²)`. but there's a catch: memory.

### the memory arithmetic: where your vram goes

let's work through a concrete example. say you're running llama-7b (32 layers, 32 attention heads, 4096 hidden dimension, head dimension of 128). you want to generate a response to a 2048-token prompt.

first, let's calculate the kv cache size for a single sequence at a single layer. as detailed in [kipply's excellent breakdown of transformer inference arithmetic](https://kipp.ly/transformer-inference-arithmetic/), each token needs to store:
- one key vector: `hidden_dim` floats
- one value vector: `hidden_dim` floats

so per token, per layer: `2 * hidden_dim * sizeof(float)` bytes.

for llama-7b with fp16 (2 bytes per element):
```
per token per layer = 2 * 4096 * 2 bytes = 16,384 bytes = 16 KB
```

across all 32 layers:
```
per token all layers = 16 KB * 32 = 512 KB per token
```

for a 2048-token context:
```
total kv cache = 512 KB * 2048 = 1,048,576 KB ≈ 1 GB
```

**one sequence with a 2048-token context uses 1 gb of vram just for the kv cache.** this is separate from the model weights (which are ~13 gb for llama-7b in fp16) and the activations needed during forward passes.

now imagine you're serving a batch of 32 concurrent users, each with 2048-token contexts. that's 32 gb *just for kv cache*. suddenly your 80gb a100 doesn't seem so spacious.

this math is what drove me to implement the blocking scheme in `src/kv_cache.cpp`. you can't just allocate a giant contiguous buffer per sequence because:
1. sequences have wildly different lengths
2. you don't know in advance how long they'll be
3. memory fragmentation becomes a nightmare

### blocked storage: the pagedattention insight

instead of allocating a massive chunk per sequence, i went with fixed-size blocks. this is inspired by vllm's pagedattention ([kwon et al., 2023](https://arxiv.org/abs/2309.06180)).

in my implementation, each block stores `block_size` tokens worth of kv data. the actual memory calculation is in `KVCache::bytes_per_block`:

```cpp
size_t bytes_per_block = block_size * hidden_dim * 2 * sizeof(float);
```

for `block_size = 16` and `hidden_dim = 4096`:
```
bytes_per_block = 16 * 4096 * 2 * 4 = 524,288 bytes = 512 KB
```

notice this is *per layer*. so a single block across all 32 layers of llama-7b would actually be `512 KB * 32 = 16 MB`.

why blocks? because now you can:
- allocate memory in fixed chunks (easier memory management)
- share blocks between sequences with common prefixes
- evict individual blocks without disrupting entire sequences
- track granular usage statistics per block

in `allocate_block()`, when i run out of free blocks, i need to evict something. this is where policy meets arithmetic:

```cpp
BlockID KVCache::allocate_block(SequenceID seq_id) {
    BlockID block_id;
    if (!free_blocks_.empty()) {
        block_id = free_blocks_.back();
        free_blocks_.pop_back();
        log_event("ALLOC", block_id, seq_id);
    } else {
        // eviction logic based on policy
        block_id = evict_victim();
        log_event("EVICT", block_id, seq_id);
    }
    // reset scores for reuse
    block_frequency_[block_id] = 0.0f;
    block_value_[block_id] = 0.0f;
    return block_id;
}
```

## eviction policies: the forgotten art of cache management

once you've burned through your `max_blocks` budget, you're playing a zero-sum game. every new block needs to kick out an old one. the policy you choose fundamentally changes your system's behavior.

### lru: the classic approach

least recently used is straightforward. i maintain `lru_list_`, a deque tracking block access order. every time `touch_block()` is called (when a block is read during attention), i move it to the front:

```cpp
void KVCache::touch_block(BlockID block_id) {
    if (policy_ == EvictionPolicy::LRU) {
        // remove from current position
        lru_list_.erase(
            std::remove(lru_list_.begin(), lru_list_.end(), block_id),
            lru_list_.end()
        );
        // move to front (most recent)
        lru_list_.push_front(block_id);
    }
    // update decay-based scores...
}
```

when evicting, i grab the back of the deque (least recently used):

```cpp
BlockID victim = lru_list_.back();
lru_list_.pop_back();
```

lru works great when you have repeated queries with the same prefix. imagine a chatbot where 80% of users start with "summarize this document: ". those shared prefix blocks stay hot in the cache.

but lru fails spectacularly with sequential access patterns. if you're processing a long document that doesn't fit in cache, every new block evicts an old one, and you get 0% hit rate. this is called "cache pollution" or "scanning."

### sliding window: the pragmatist's choice

sliding window is brutal but effective. instead of trying to be clever about what to keep, you just enforce a hard rule: only keep the last `window_size` tokens per sequence.

in `store_token()`, after placing the new token, i check if the sequence has exceeded its window:

```cpp
void KVCache::store_token(SequenceID seq_id, const float* key, const float* value) {
    // ... place token in block ...
    
    if (window_size_ > 0) {
        auto& seq_tokens = seq_tokens_[seq_id];
        while (seq_tokens.size() > window_size_) {
            TokenID old_token = seq_tokens.front();
            seq_tokens.pop_front();
            // evict the block containing old_token
            // (simplified; actual code handles block ref counting)
        }
    }
}
```

this gives you predictable memory usage: `sequences * window_size * bytes_per_token`. for serving systems, this predictability is gold. you can calculate exactly how many concurrent sequences you can handle.

the cost? you lose long-range attention. models like llama with 4096-token context windows get crippled if you set `window_size = 1024`. the model literally can't "see" tokens beyond that horizon.

but for applications where recent context matters most (chat, code completion), sliding window is incredibly practical. it's what i used in my default config because it prevented memory explosions during long trace runs.

### lfu with exponential decay: keeping the important stuff

lru and sliding window are recency-based. but what if some blocks are more "important" than others? this is where frequency and value heuristics come in.

i implemented a decay-based lfu where each block tracks a score that increases on access but decays over time:

```cpp
void KVCache::touch_block(BlockID block_id) {
    // ... lru list updates ...
    
    if (policy_ == EvictionPolicy::LFU || policy_ == EvictionPolicy::COST) {
        block_frequency_[block_id] = block_frequency_[block_id] * frequency_decay_ + 1.0f;
        block_value_[block_id] = block_value_[block_id] * value_decay_ + 1.0f;
    }
}
```

the decay formula `score = score * decay + 1` is interesting. with `decay = 0.9`:
- first access: `score = 0 * 0.9 + 1 = 1.0`
- second access: `score = 1.0 * 0.9 + 1 = 1.9`
- third access: `score = 1.9 * 0.9 + 1 = 2.71`
- if not accessed for 10 steps: `score = 2.71 * 0.9^10 ≈ 0.94`

this creates a "half-life" effect. frequently accessed blocks build up high scores, but stale blocks gradually decay. when evicting, i pick the block with the lowest score.

the `COST` policy is similar but uses a different decay rate for the value heuristic, letting you tune "how much history matters" vs "how much recent frequency matters."

in practice, these performed better than pure lru for workloads with zipfian access patterns (common in real systems where a few popular prefixes dominate traffic).

## moving data: the cuda implementation

okay, now let's talk about how bits actually move from cpu to gpu. this is where `src/cuda_kv.cu` and `src/cuda_kv.h` come in.

### the layout: flat buffers with block addressing

on the gpu, i allocate two massive flat buffers:

```cpp
struct DeviceKVLayout {
    float* d_keys;    // [max_blocks * block_size * hidden_dim]
    float* d_values;  // [max_blocks * block_size * hidden_dim]
    float* d_staging; // [batch_size * 2 * hidden_dim]
    cudaStream_t stream;
};
```

when a token at position `slot` in block `block_id` needs to be stored, the address is:

```cpp
size_t key_offset = (block_id * block_size + slot) * hidden_dim;
size_t value_offset = key_offset; // same for values
```

this is critical to understand. unlike a naive "one big array per sequence" approach, we're randomly scattering data into blocks. this means non-contiguous memory access, which can hurt cache locality on the gpu.

### the three-stage pipeline

moving data efficiently requires a pipeline:

**stage 1: host staging**

in `KVCache::store_token()`, i pack the key and value into a host-side staging buffer:

```cpp
void KVCache::store_token(SequenceID seq_id, const float* key, const float* value) {
    // ... find block and slot ...
    
    // pack into staging buffer
    std::memcpy(staging_buffer_.data(), key, hidden_dim_ * sizeof(float));
    std::memcpy(staging_buffer_.data() + hidden_dim_, value, hidden_dim_ * sizeof(float));
    
    // ...
}
```

this gives us a contiguous chunk: `[key[0]...key[hidden_dim-1], value[0]...value[hidden_dim-1]]`.

**stage 2: async h2d transfer**

then i launch an async copy to the gpu staging buffer:

```cpp
void DeviceKVLayout::stage_block(const float* h_kv_data, size_t token_count) {
    size_t bytes = token_count * 2 * hidden_dim * sizeof(float);
    cudaMemcpyAsync(d_staging, h_kv_data, bytes, 
                    cudaMemcpyHostToDevice, stream);
}
```

the `Async` is key. this doesn't block the cpu thread. we can queue up more work while the transfer happens.

**stage 3: scatter kernel**

finally, a custom kernel moves data from the staging buffer to the final location:

```cpp
__global__ void move_block_kernel(
    float* d_keys, float* d_values,
    const float* d_staging,
    BlockID block_id, size_t slot,
    size_t block_size, size_t hidden_dim
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = 2 * hidden_dim; // key + value
    
    if (idx < total) {
        size_t base_offset = (block_id * block_size + slot) * hidden_dim;
        
        if (idx < hidden_dim) {
            // copy key
            d_keys[base_offset + idx] = d_staging[idx];
        } else {
            // copy value
            size_t value_idx = idx - hidden_dim;
            d_values[base_offset + value_idx] = d_staging[idx];
        }
    }
}
```

this kernel is launched with a 1d grid over `2 * hidden_dim` elements. each thread copies one float from staging to its final home.

why not just use `cudaMemcpy` directly to the final location? because we want to:
1. overlap transfers (staging buffer acts as a double-buffer)
2. batch multiple tokens into one kernel launch
3. decouple transfer from placement (easier to add prefetching later)

### bandwidth math: why this matters

let's do some bandwidth calculations. an a100 has ~1,935 gb/s of hbm bandwidth. for llama-7b with `hidden_dim = 4096`:

per token transfer:
```
2 * 4096 * 4 bytes = 32,768 bytes = 32 KB
```

at peak bandwidth:
```
1,935 GB/s / 32 KB = 60,468,750 tokens/second ≈ 60M tokens/s
```

in practice, you get maybe 30-50% of peak due to:
- non-contiguous access (the scatter kills locality)
- pcie bottleneck for cpu→gpu transfers
- kernel launch overhead
- cache misses

for prefill (processing many tokens in parallel), this isn't too bad. but for decode (one token at a time), the overhead becomes significant. this is why techniques like continuous batching and speculation are so important—they amortize the fixed costs across multiple tokens.

in my instrumentation (`src/instrumentation.h`), i track transfer bytes:

```cpp
void log_transfer(size_t bytes) {
    total_transfer_bytes_ += bytes;
    events_.push_back({
        .type = EventType::TRANSFER,
        .timestamp = now_micros(),
        .bytes = bytes
    });
}
```

watching `total_transfer_bytes_` climb in the logs really drives home how much of inference is just "moving floats around."

## the synthetic transformer: deterministic key-value generation

i didn't want to deal with loading real model weights or implementing actual attention math, so i built a fake transformer that produces deterministic k/v vectors.

in `src/transformer.cpp`:

```cpp
void TransformerBlock::forward(TokenID token_id, size_t step,
                                float* key_out, float* value_out) {
    float base = (token_id % vocab_size_) / static_cast<float>(vocab_size_);
    
    for (size_t i = 0; i < hidden_dim_; ++i) {
        key_out[i] = std::sin(base + 0.01f * step + 0.001f * i);
        value_out[i] = std::cos(base + 0.01f * 1.3f * step + 0.001f * 1.3f * i);
    }
}
```

this is just `sin` and `cos` with some offsets. the token id shifts the base frequency, the step adds a temporal component (like positional encoding), and the dimension index adds variation across the vector.

why does this work? because i don't actually care about the *values* of the keys and values. i care about:
- having realistic-sized vectors (4096 floats)
- producing different k/v for different tokens (the `token_id % vocab_size` ensures this)
- being deterministic (same token + step = same k/v, helpful for debugging)

in `src/main.cpp`, i hash human-readable prompts into token ids:

```cpp
std::hash<std::string> hasher;
for (const auto& word : prompt_words) {
    TokenID token_id = static_cast<TokenID>(hasher(word) % vocab_size);
    token_ids.push_back(token_id);
}
```

so if i pass `--prompt "hello world"`, it becomes two deterministic token ids. this let me test the cache with "real-ish" sequences without needing a tokenizer.

## the token-time simulator: queueing theory for llm inference

the `sim_main` binary (`src/sim_main.cpp`) is where i explored the macro-level dynamics. instead of running cuda kernels, i model inference as a queueing system.

### the two-server model

llm inference has two distinct phases with radically different characteristics:

**prefill (prompt processing):**
- processes all prompt tokens in parallel
- high throughput (10,000+ tokens/sec on modern gpus)
- short duration even for long prompts (2048 tokens at 10k tok/s = 204 ms)
- bottleneck: memory bandwidth and matmul compute

**decode (generation):**
- processes one token at a time per sequence
- low throughput (50-100 tokens/sec per sequence)
- duration = `num_output_tokens / tokens_per_sec`
- bottleneck: memory bandwidth (loading kv cache) and sequential dependency

i modeled these as two separate "servers" with different service rates:

```cpp
struct ServerParams {
    float tokens_per_ms_prefill;  // e.g., 10.0 (10k tok/s)
    float tokens_per_ms_decode;   // e.g., 0.1 (100 tok/s)
    size_t max_batch;             // how many requests can process concurrently
};
```

the duration of a phase is:

```cpp
float duration_ms = num_tokens / tokens_per_ms;
```

this is a massive simplification (real systems have variable batch sizes, kernel launch overhead, etc.), but it captures the essential trade-off: prefill is fast and parallel, decode is slow and serial.

### the event loop

the simulator maintains an event priority queue:

```cpp
struct Event {
    float time_ms;
    EventType type;  // ARRIVAL, PREFILL_DONE, DECODE_CHUNK_DONE, etc.
    RequestID request_id;
};

std::priority_queue<Event> event_queue;
```

the main loop processes events chronologically:

```cpp
while (!event_queue.empty()) {
    Event evt = event_queue.top();
    event_queue.pop();
    current_time = evt.time_ms;
    
    switch (evt.type) {
        case ARRIVAL:
            handle_arrival(evt.request_id);
            break;
        case PREFILL_DONE:
            handle_prefill_done(evt.request_id);
            break;
        case DECODE_CHUNK_DONE:
            handle_decode_chunk(evt.request_id);
            break;
    }
}
```

when a request arrives, it enters the prefill queue. when prefill completes, it moves to the decode queue. when decode finishes all tokens, the request is complete.

### scheduling: where policy meets performance

the most interesting part is how we choose which request to process next when the server becomes idle.

**fcfs (first-come, first-serve):**

```cpp
RequestID schedule_fcfs(const std::deque<RequestID>& queue) {
    return queue.front();  // trivial: just pick the oldest
}
```

fcfs is fair but terrible for latency. imagine three requests arrive:
1. request a: 2000-token prefill, 500-token decode
2. request b: 10-token prefill, 10-token decode
3. request c: 50-token prefill, 50-token decode

with fcfs, request b waits behind request a's massive prefill, even though b could finish in milliseconds. this is head-of-line blocking.

**srpt (shortest remaining processing time):**

```cpp
RequestID schedule_srpt(const std::vector<Request>& requests) {
    return *std::min_element(requests.begin(), requests.end(),
        [](const Request& a, const Request& b) {
            size_t remaining_a = a.num_output_tokens - a.generated_tokens;
            size_t remaining_b = b.num_output_tokens - b.generated_tokens;
            return remaining_a < remaining_b;
        });
}
```

srpt is provably optimal for minimizing average completion time in non-preemptive queues. it drastically reduces tail latency because short requests zip through.

but it's "unfair": long requests can get starved if short requests keep arriving. in my grid sweeps (`tools/run_grid.py`), srpt consistently showed 2-3x better p99 latency than fcfs, but at high load (λ > 0.9), some long requests would get delayed indefinitely.

**slo-aware scheduling:**

```cpp
RequestID schedule_slo(const std::vector<Request>& requests, float current_time) {
    return *std::min_element(requests.begin(), requests.end(),
        [current_time](const Request& a, const Request& b) {
            float slack_a = a.slo_ms - (current_time - a.arrival_time);
            float slack_b = b.slo_ms - (current_time - b.arrival_time);
            return slack_a < slack_b;  // prioritize smallest slack
        });
}
```

if each request has a deadline (slo), we prioritize the one with the least "slack" (time remaining until deadline). this is deadline-based scheduling.

in practice, this needs good estimates of how long a request will take. if your estimate is wrong, you'll miss deadlines. but when it works, it's great for mixed workloads where some users pay for guaranteed latency.

### prefix cache modeling

the simulator also includes a simple prefix cache model (`src/simulator.cpp`, `PrefixCacheModel`):

```cpp
struct PrefixCacheModel {
    std::unordered_map<PrefixID, CachedPrefix> cache_;
    size_t capacity_blocks_;
    EvictionPolicy policy_;
    
    size_t check_hit(PrefixID prefix_id, size_t prompt_tokens);
    void update(PrefixID prefix_id, size_t prompt_tokens);
};
```

when a request arrives, i hash its prefix (first `prefix_length` tokens) and check if it's cached:

```cpp
size_t hit_tokens = prefix_cache.check_hit(request.prefix_id, request.prompt_tokens);
size_t effective_prompt_tokens = request.prompt_tokens - hit_tokens;
```

the "effective" prefill work is reduced by the hit tokens. this captures the speedup from reusing cached k/v.

i round prefixes to block boundaries to match the block-based cache:

```cpp
size_t prefix_blocks = (prompt_tokens + block_size - 1) / block_size;
```

so a 50-token prefix with `block_size = 16` uses `ceil(50/16) = 4` blocks = 64 tokens of capacity.

tracking `prefill_hit_rate` across runs showed the impact of cache capacity. with `capacity_blocks = 0` (no cache), hit rate was obviously 0%. with `capacity_blocks = 1000` and zipfian traffic (α = 1.5, very skewed), hit rate reached 40-50%, because the popular prefixes fit in cache.

this is why systems like vllm push so hard on prefix sharing. if 80% of requests share the first 500 tokens, caching those saves 80% * 500 tokens of prefill work on every request.

## trace generation: simulating realistic workloads

real llm traffic is messy. some users send one-word questions ("hi"), others send 4000-token documents. some prefixes appear constantly ("summarize:"), others are unique. load comes in bursts.

`tools/tracegen.py` lets me create synthetic traces that mimic this:

```python
def generate_trace(
    duration_s=60,
    arrival_rate=10.0,  # λ: requests per second
    prompt_mean=512,
    prompt_std=256,
    output_mean=128,
    output_std=64,
    prefix_zipf_alpha=1.5,
    num_unique_prefixes=100,
    prefix_length=50
):
```

**poisson arrivals:**

```python
arrival_times = []
t = 0
while t < duration_s:
    t += np.random.exponential(1.0 / arrival_rate)
    arrival_times.append(t)
```

poisson processes model "random" arrivals well. the inter-arrival time is exponentially distributed with rate λ.

**lognormal lengths:**

```python
prompt_tokens = int(np.random.lognormal(
    mean=np.log(prompt_mean),
    sigma=np.log(1 + prompt_std / prompt_mean)
))
```

lognormal distributions are heavy-tailed: most values are small, but there's a long tail of huge values. this matches real user behavior where most prompts are short but some are 10x longer.

**zipfian prefixes:**

```python
def sample_zipf(n, alpha):
    # rank i has probability ∝ 1 / i^alpha
    weights = np.array([1.0 / (i ** alpha) for i in range(1, n + 1)])
    weights /= weights.sum()
    return np.random.choice(n, p=weights)

prefix_id = sample_zipf(num_unique_prefixes, prefix_zipf_alpha)
```

zipf distributions model "popularity" (like word frequency or web page views). with α = 1.5:
- prefix 1 has weight `1 / 1^1.5 = 1.0`
- prefix 2 has weight `1 / 2^1.5 ≈ 0.35`
- prefix 10 has weight `1 / 10^1.5 ≈ 0.032`

so the top few prefixes appear constantly, while the long tail is rare. this is realistic: in production, common system prompts and popular queries dominate traffic.

the generated trace is a jsonl file:

```json
{"arrival_time": 0.123, "prompt_tokens": 512, "output_tokens": 128, "prefix_id": 3}
{"arrival_time": 0.456, "prompt_tokens": 1024, "output_tokens": 64, "prefix_id": 1}
...
```

## grid studies: sweeping parameter space

once i had the simulator and trace generator, i could explore the design space systematically with `tools/run_grid.py`:

```python
for load_multiplier in [0.5, 0.7, 0.9, 1.0]:
    for scheduler in ['fcfs', 'srpt', 'slo']:
        for cache_policy in ['lru', 'lfu', 'sliding']:
            for cache_blocks in [0, 500, 1000, 2000]:
                # generate trace with arrival_rate * load_multiplier
                # run sim_main with this config
                # collect metrics
```

this generates a grid of results: `runs/grid.csv`.

example metrics:

```csv
load,scheduler,cache_policy,cache_blocks,p50_latency,p99_latency,throughput,hit_rate
0.5,fcfs,none,0,234.5,456.2,8.3,0.0
0.5,srpt,none,0,123.4,289.1,9.1,0.0
0.5,srpt,lru,1000,98.2,201.3,10.5,0.42
0.9,fcfs,none,0,1234.5,8901.2,5.2,0.0
0.9,srpt,lru,1000,456.7,1203.4,8.1,0.38
```

`tools/plot_grid.py` turns this into charts:

- **latency vs load:** as load increases, latency explodes (queueing delay). srpt stays lower than fcfs until very high load.
- **hit rate vs cache capacity:** hit rate saturates once the cache is big enough to hold the working set. for zipfian α = 1.5, this happens around 1000 blocks.
- **throughput vs scheduler:** srpt achieves higher throughput at high load because it avoids wasting time on long requests that would block others.

these plots (`runs/plots/latency_vs_load.png`, etc.) were invaluable for understanding trade-offs. for instance, i learned that adding a prefix cache improves throughput more at high load (0.9) than low load (0.5), because at high load, the queue is long enough that cache hits meaningfully reduce wait time.

## visualization: watching the cache in action

numbers are great, but i wanted to *see* what was happening. `tools/visualize.py` replays the json event log and draws the cache state at each event.

the script builds a grid where rows are blocks and columns are slots:

```python
cache_grid = [[None] * block_size for _ in range(max_blocks)]
```

as it processes events:

```python
if event['type'] == 'token_placed':
    block_id = event['block_id']
    slot = event['slot']
    token_text = event.get('token_text', f"tok_{event['token_id']}")
    cache_grid[block_id][slot] = {
        'token': token_text,
        'seq_id': event['seq_id'],
        'phase': event['phase']  # 'prefill' or 'decode'
    }
```

then it renders the grid to a png:

```python
fig, ax = plt.subplots(figsize=(block_size * 0.5, max_blocks * 0.3))
for block_id in range(max_blocks):
    for slot in range(block_size):
        cell = cache_grid[block_id][slot]
        if cell:
            color = 'lightblue' if cell['phase'] == 'prefill' else 'light
          coral'
                      ax.add_patch(Rectangle((slot, block_id), 1, 1, 
                                              facecolor=color, edgecolor='black'))
                      ax.text(slot + 0.5, block_id + 0.5, cell['token'], 
                              ha='center', va='center', fontsize=6)
          ```
          
          running this on a trace with sliding window policy shows:
          1. blocks filling up from left to right during prefill
          2. decode tokens appending to the same blocks
          3. when the window is exceeded, old blocks get wiped (visually empty)
          4. new sequences reuse the freed blocks (different colors)
          
          i also experimented with rendering cache "heatmaps" where cell color represents access frequency. hot blocks (high lfu score) were red, cold blocks (unused) were blue. watching lru vs lfu side-by-side made the policy differences visceral.
          
          ## instrumentation: the unsung hero
          
          none of this would've been possible without good instrumentation. `src/instrumentation.h` provides:
          
          ```cpp
          class Instrumentation {
              void log_event(const std::string& type, BlockID block_id, SequenceID seq_id);
              void log_token_placement(TokenID token_id, BlockID block_id, size_t slot, 
                                       const std::string& phase);
              void log_transfer(size_t bytes);
              void write_json(const std::string& filepath);
              Summary get_summary();
          };
          ```
          
          every key operation (`allocate_block`, `evict_victim`, `store_token`, `cudaMemcpyAsync`) logs an event with a microsecond-precision timestamp:
          
          ```cpp
          uint64_t now_micros() {
              using namespace std::chrono;
              return duration_cast<microseconds>(
                  steady_clock::now().time_since_epoch()
              ).count();
          }
          ```
          
          this let me answer questions like:
          - how many evictions per second during a burst?
          - what's the distribution of time between token placements?
          - which sequences cause the most evictions?
          
          the json output is structured for post-processing:
          
          ```json
          {
            "events": [
              {"timestamp": 1000, "type": "ALLOC", "block_id": 0, "seq_id": 1},
              {"timestamp": 1050, "type": "TRANSFER", "bytes": 32768},
              {"timestamp": 1100, "type": "token_placed", "token_id": 42, "block_id": 0, "slot": 0, "phase": "prefill"}
            ],
            "summary": {
              "total_allocations": 1024,
              "total_evictions": 128,
              "total_transfer_bytes": 1048576
            }
          }
          ```
          
          i can load this into pandas and do arbitrary analysis:
          
          ```python
          import pandas as pd
          import json
          
          with open('runs/single.json') as f:
              data = json.load(f)
          
          df = pd.DataFrame(data['events'])
          df['timestamp'] = df['timestamp'] / 1e6  # convert to seconds
          
          # plot evictions over time
          evictions = df[df['type'] == 'EVICT'].groupby('timestamp').size()
          evictions.plot(title='evictions per second')
          ```
          
          this kind of observability is critical for systems work. you can't optimize what you can't measure.
          
          ## takeaways: what i actually learned
          
          building this was eye-opening. here's what stuck with me:
          
          ### 1. memory arithmetic dominates everything
          
          the math of `block_size * hidden_dim * 2 * sizeof(float) * num_layers` is your entire world. once you internalize that a single 2048-token sequence in llama-7b costs 1 gb, you understand why:
          - batching is hard (memory grows linearly with batch size)
          - long contexts are expensive (memory grows linearly with sequence length)
          - prefix caching is essential (reusing shared prefixes saves massive amounts of memory)
          
          as kipply points out in their [transformer inference arithmetic](https://kipp.ly/transformer-inference-arithmetic/) writeup, the kv cache often consumes more memory than the model weights during inference. for a 70b model with long contexts, the cache can exceed 100 gb. this is why techniques like quantization (storing kv in int8 instead of fp16) and grouped-query attention (sharing keys across heads to halve kv cache size) are so impactful.
          
          ### 2. there is no perfect eviction policy
          
          lru is simple and works for repeated queries. sliding window is predictable and bounds memory. lfu/cost heuristics are clever but add complexity. each one makes trade-offs:
          - lru: great hit rate for repeated access, terrible for scans
          - sliding window: predictable memory, but loses long-range context
          - lfu: adapts to access patterns, but needs tuning (decay rate matters a lot)
          
          in real systems, you probably want a hybrid: sliding window as a hard cap, with lru/lfu inside that window.
          
          ### 3. prefill and decode are different animals
          
          modeling them as separate servers was illuminating. prefill is embarrassingly parallel and bandwidth-bound. decode is serial and latency-critical.
          
          this explains why:
          - continuous batching works (interleave prefill and decode for different requests)
          - speculative decoding helps (run a small model to "draft" tokens, verify with the big model)
          - chunked prefill is popular (split a long prefill into chunks to avoid blocking decode)
          
          the two-server model is a simplification, but it captures the fundamental tension: do you prioritize throughput (fill the gpu with prefill) or latency (keep decode responsive)?
          
          ### 4. heavy-tailed workloads break naive schedulers
          
          with uniform request sizes, fcfs is fine. but with lognormal prompt lengths (most small, some huge), fcfs causes massive head-of-line blocking.
          
          srpt mitigates this dramatically. in my traces, p99 latency was 3-4x better with srpt than fcfs at high load. the cost is "unfairness"—long requests wait longer. whether that's acceptable depends on your sla.
          
          ### 5. instrumentation is half the work
          
          i spent as much time on `instrumentation.h` and `visualize.py` as on the cache logic itself. but it was worth it. being able to:
          - replay event logs
          - visualize cache occupancy
          - measure hit rates and eviction churn
          - plot latency distributions
          
          ...made debugging and experimentation orders of magnitude faster. without good observability, i would've been flying blind.
          
          ### 6. synthetic workloads > real models for exploration
          
          not using real model weights was liberating. the deterministic transformer let me:
          - iterate quickly (no waiting for model downloads)
          - debug reliably (same inputs = same outputs)
          - focus on the systems problem (memory management, not model accuracy)
          
          once i understood the cache dynamics, porting to a real model (e.g., loading llama weights and using actual attention) would be straightforward. the cache logic doesn't care if the k/v vectors are from `sin/cos` or a real mlp.
          
          ## wrapping up
          
          this project scratched an itch i've had since reading about vllm and pagedattention. i wanted to *feel* the memory pressure, see the evictions happen, and understand why certain policies work better than others.
          
          the code is all there if you want to play with it. some ideas for extensions:
          - implement flash-attention and measure actual kernel times
          - add support for grouped-query attention (fewer k/v vectors)
          - try eviction policies based on attention scores (evict tokens with low attention weight)
          - model multi-tenancy (different users with different slas)
          - add speculation (draft tokens with a small model, verify with a big one)
          
          if you're into mlsys, i highly recommend building something like this. reading papers is great, but writing the code—seeing the bytes add up, watching the cache fill, tuning the decay parameters—gives you an intuition that's hard to get any other way.
          
          and if you do build something, let me know. i'd love to see what you find.
          
          ---
          
          *references:*
          - [transformer inference arithmetic by kipply](https://kipp.ly/transformer-inference-arithmetic/) - excellent breakdown of memory costs and bandwidth limits in llm inference
          - kwon et al., "efficient memory management for large language model serving with pagedattention," 2023 ([arxiv](https://arxiv.org/abs/2309.06180)) - the vllm paper that inspired the block-based cache design

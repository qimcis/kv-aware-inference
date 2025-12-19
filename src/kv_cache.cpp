#include "kv_cache.h"

// block allocation and policy-specific eviction
#include <algorithm>
#include <limits>
#include <stdexcept>

KVCache::KVCache(const KVConfig& cfg, Instrumentation& instr) : cfg_(cfg), instr_(instr) {
    host_keys_.resize(cfg_.max_blocks * cfg_.block_size * cfg_.hidden);    // host K staging
    host_values_.resize(cfg_.max_blocks * cfg_.block_size * cfg_.hidden);  // host V staging
    blocks_.resize(cfg_.max_blocks);                                       // metadata per block
    lru_iters_.assign(cfg_.max_blocks, lru_list_.end());                   // initialize LRU slots
    freq_.assign(cfg_.max_blocks, 0.0);
    value_.assign(cfg_.max_blocks, 0.0);
    device_ = allocate_device_kv(cfg_.max_blocks, cfg_.block_size, cfg_.hidden); // allocate device buffers
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create stream"); // async stream
    owns_stream_ = true;
}

KVCache::~KVCache() {
    if (owns_stream_) {
        cudaStreamDestroy(stream_); // destroy stream if we created it.
    }
    free_device_kv(device_);        // release device side allocations
}

std::size_t KVCache::bytes_per_block() const {
    return cfg_.block_size * cfg_.hidden * sizeof(float) * 2; // K and V, float32
}

void KVCache::log_evictions(int block_id) {
    auto& meta = blocks_[block_id];
    for (const auto& tok : meta.tokens) {
        // emit per-token eviction so the viewer can clear the slot
        instr_.log_token_event("evict", meta.seq_id, static_cast<std::size_t>(block_id),
                               tok.block_offset, tok.token_index, tok.token_id, tok.token_text,
                               tok.decode);
    }
    meta.tokens.clear();
    freq_[block_id] = 0.0;
    value_[block_id] = 0.0;
}

int KVCache::select_victim() {
    if (cfg_.policy == CachePolicy::kLRU || cfg_.policy == CachePolicy::kSlidingWindow) {
        if (lru_list_.empty()) {
            throw std::runtime_error("no blocks available for eviction");
        }
        int id = lru_list_.front();       // oldest by recency
        lru_list_.pop_front();            // remove from head
        lru_iters_[id] = lru_list_.end(); // mark as detached
        return id;
    }

    double best_score = std::numeric_limits<double>::infinity();
    int best_id = -1;
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        if (!blocks_[i].in_use) {
            continue;
        }
        double score = 0.0;
        if (cfg_.policy == CachePolicy::kLFU) {
            score = freq_[i];
        } else if (cfg_.policy == CachePolicy::kCost) {
            score = value_[i];
        }
        if (score < best_score || (score == best_score && static_cast<int>(i) < best_id)) {
            best_score = score;
            best_id = static_cast<int>(i);
        }
    }
    if (best_id < 0) {
        throw std::runtime_error("no blocks available for eviction");
    }
    auto it = lru_iters_[best_id];
    if (it != lru_list_.end()) {
        lru_list_.erase(it);
    }
    lru_iters_[best_id] = lru_list_.end();
    return best_id;
}

void KVCache::touch_block(int block_id) {
    // update decayed frequency/value for LFU/cost
    if (cfg_.policy == CachePolicy::kLFU || cfg_.policy == CachePolicy::kCost) {
        freq_[block_id] = freq_[block_id] * cfg_.decay + 1.0;
        if (cfg_.policy == CachePolicy::kCost) {
            value_[block_id] = value_[block_id] * cfg_.decay + 1.0;
        }
    }
    auto it = lru_iters_[block_id];
    if (it != lru_list_.end()) {
        lru_list_.erase(it); // remove existing position
    }
    lru_list_.push_back(block_id); // move to MRU position
    lru_iters_[block_id] = std::prev(lru_list_.end());
}

int KVCache::allocate_block(std::size_t seq_id) {
    // first look for unused blocks
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        if (!blocks_[i].in_use) {
            // grab a free block and mark it in LRU
            blocks_[i].in_use = true;
            blocks_[i].seq_id = seq_id;
            blocks_[i].version++;
            blocks_[i].tokens.clear();
            lru_list_.push_back(static_cast<int>(i));
            lru_iters_[i] = std::prev(lru_list_.end());
            freq_[i] = 0.0;
            value_[i] = 0.0;
            instr_.log(EventType::kAllocate, "fresh", seq_id, i, 0, bytes_per_block());
            return static_cast<int>(i);
        }
    }

    int evict_id = select_victim(); // pick lru block
    log_evictions(evict_id);
    const char* reason = "evict";
    switch (cfg_.policy) {
    case CachePolicy::kLRU:
    case CachePolicy::kSlidingWindow:
        reason = "lru";
        break;
    case CachePolicy::kLFU:
        reason = "lfu";
        break;
    case CachePolicy::kCost:
        reason = "cost";
        break;
    }
    instr_.log(EventType::kEvict, reason, blocks_[evict_id].seq_id, evict_id, 0, bytes_per_block());
    blocks_[evict_id].seq_id = seq_id;
    blocks_[evict_id].version++;
    blocks_[evict_id].tokens.clear();
    instr_.log(EventType::kReuse, "reassign", seq_id, evict_id, 0, bytes_per_block());
    lru_list_.push_back(evict_id);             // put reused block at mru
    lru_iters_[evict_id] = std::prev(lru_list_.end());
    freq_[evict_id] = 0.0;
    value_[evict_id] = 0.0;
    return evict_id;
}

void KVCache::record_transfer(std::size_t seq_id, std::size_t block_id, std::size_t token_index,
                              std::size_t tokens, bool decode_step) {
    std::size_t bytes = tokens * cfg_.hidden * sizeof(float) * 2; // K+V bytes
    instr_.log(EventType::kTransfer, decode_step ? "decode" : "prefill", seq_id, block_id,
               token_index, bytes);
}

void KVCache::store_token(std::size_t seq_id, std::size_t token_index,
                          int token_id, const std::string& token_text,
                          const std::vector<float>& key, const std::vector<float>& value,
                          bool decode_step) {
    if (key.size() != cfg_.hidden || value.size() != cfg_.hidden) {
        throw std::runtime_error("key/value size mismatch with hidden size");
    }
    auto& seq = sequences_[seq_id];
    if (seq.tokens_in_block == 0) {
        seq.current_block = allocate_block(seq_id); // grab a block when current is exhausted
    }

    std::size_t block_base =
        static_cast<std::size_t>(seq.current_block) * cfg_.block_size * cfg_.hidden;
    std::size_t offset = block_base + seq.tokens_in_block * cfg_.hidden;
    std::copy(key.begin(), key.end(), host_keys_.begin() + offset);   // stage K into host buffer
    std::copy(value.begin(), value.end(), host_values_.begin() + offset); // stage V likewise

    seq.tokens_in_block++;
    seq.total_tokens++;
    std::size_t block_offset = seq.tokens_in_block - 1;

    EventType step_type = decode_step ? EventType::kDecodeStep : EventType::kPrefillStep;
    instr_.log(step_type, decode_step ? "decode_step" : "prefill_step", seq_id,
               static_cast<std::size_t>(seq.current_block), token_index, 0, token_id, token_text,
               decode_step);
    instr_.log_token_event("place", seq_id, static_cast<std::size_t>(seq.current_block), block_offset,
                           token_index, token_id, token_text, decode_step);
    TokenLabel label{token_index, token_id, token_text, decode_step, block_offset, seq.current_block};
    blocks_[seq.current_block].tokens.push_back(label);
    if (cfg_.policy == CachePolicy::kSlidingWindow && cfg_.window_size > 0) {
        auto& window = seq.window_tokens;
        window.push_back(label); // track newest token in window queue.
        while (window.size() > cfg_.window_size) {
            TokenLabel drop = window.front();
            window.pop_front();
            if (drop.block_id >= 0 && static_cast<std::size_t>(drop.block_id) < blocks_.size() &&
                blocks_[drop.block_id].seq_id == seq_id) {
                auto& toks = blocks_[drop.block_id].tokens;
                auto it = std::remove_if(toks.begin(), toks.end(), [&](const TokenLabel& t) {
                    return t.token_index == drop.token_index;
                });
                if (it != toks.end()) {
                    toks.erase(it, toks.end());
                    instr_.log_token_event("window_evict", seq_id, static_cast<std::size_t>(drop.block_id),
                                           drop.block_offset, drop.token_index, drop.token_id,
                                           drop.token_text, drop.decode);
                }
            }
        }
    }

    stage_block(host_keys_.data() + block_base, host_values_.data() + block_base,
                seq.tokens_in_block, device_, stream_); // upload staged K/V to device staging
    move_block_to_cache(device_, static_cast<std::size_t>(seq.current_block), seq.tokens_in_block,
                        stream_); // copy staging into device cache slot
    record_transfer(seq_id, static_cast<std::size_t>(seq.current_block), token_index,
                    seq.tokens_in_block, decode_step);

    touch_block(seq.current_block);
    if (seq.tokens_in_block == cfg_.block_size) {
        seq.tokens_in_block = 0; // next token will allocate or reuse a block
    }
}

void KVCache::synchronize() {
    check_cuda(cudaStreamSynchronize(stream_), "stream sync");
}

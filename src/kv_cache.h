#pragma once

#include "cuda_kv.h"
#include "instrumentation.h"

#include <cstddef>
#include <cuda_runtime.h>
#include <deque>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

// evicition policy for the cache
enum class CachePolicy {
    kLRU,
    kSlidingWindow
};

struct KVConfig {
    std::size_t block_size = 16;
    std::size_t max_blocks = 16;
    std::size_t hidden = 64;
    CachePolicy policy = CachePolicy::kLRU;
    std::size_t window_size = 0; // only used by sliding-window (tokens to keep)
};

class KVCache {
  public:
    KVCache(const KVConfig& cfg, Instrumentation& instr);
    ~KVCache();

    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;

    // store one token's k/v along with its metadata into the cache
    void store_token(std::size_t seq_id, std::size_t token_index,
                     int token_id, const std::string& token_text, const std::vector<float>& key,
                     const std::vector<float>& value, bool decode_step);

    // syncronize CUDA stream to ensure transfers are complete
    void synchronize();
    // bytes consumed by a single block (k+v)
    std::size_t bytes_per_block() const;

  private:
    // record of where a token lives inside a block
    struct TokenLabel {
    std::size_t token_index = 0;
    int token_id = -1;
    std::string token_text;
    bool decode = false;
    std::size_t block_offset = 0;
    int block_id = -1;
};

    struct SequenceState {
        std::size_t total_tokens = 0;
        std::size_t tokens_in_block = 0;
        int current_block = -1;
        std::deque<TokenLabel> window_tokens;
    };

    struct BlockMeta {
        bool in_use = false;
        std::size_t seq_id = 0;
        std::size_t version = 0;
        std::vector<TokenLabel> tokens;
    };

    int allocate_block(std::size_t seq_id);
    void record_transfer(std::size_t seq_id, std::size_t block_id, std::size_t token_index,
                         std::size_t tokens, bool decode_step);
    void log_evictions(int block_id);
    int select_victim();
    void touch_block(int block_id);

    KVConfig cfg_;                          // static cache settings
    Instrumentation& instr_;                // event sink shared with the app
    DeviceKVLayout device_;                 // device buffers for k/v
    cudaStream_t stream_{};                 // stream for async copies
    bool owns_stream_ = false;              // whether we created the stream

    std::vector<float> host_keys_;          // host staging for keys
    std::vector<float> host_values_;        // host staging for values
    std::vector<BlockMeta> blocks_;         // per-block metadata
    std::unordered_map<std::size_t, SequenceState> sequences_; // Per-sequence state
    std::list<int> lru_list_;               // blocks ordered by recency
    std::vector<std::list<int>::iterator> lru_iters_; // iterators for LRU updates
};

// straight-line kv cache
class NaiveCache {
  public:
    NaiveCache(std::size_t hidden, Instrumentation& instr) : hidden_(hidden), instr_(instr) {}

    // append k/v for a token into a contiguous buffer; expands as needed
    void store(std::size_t seq_id, std::size_t token_index, const std::vector<float>& key,
               const std::vector<float>& value) {
        auto& buf = data_[seq_id];
        std::size_t stride = hidden_ * 2;
        if (buf.size() < (token_index + 1) * stride) {
            buf.resize((token_index + 1) * stride);
        }
        float* key_ptr = buf.data() + token_index * stride;
        float* val_ptr = key_ptr + hidden_;
        std::copy(key.begin(), key.end(), key_ptr);
        std::copy(value.begin(), value.end(), val_ptr);
        instr_.log(EventType::kNaiveWrite, "linear", seq_id, 0, token_index,
                   2 * hidden_ * sizeof(float));
    }

  private:
    std::unordered_map<std::size_t, std::vector<float>> data_; // per-sequence linear storage
    std::size_t hidden_;                                      // hidden width for stride math
    Instrumentation& instr_;                                  // event logger for writes
};

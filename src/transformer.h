// small transformer block that emits deterministic k/v vectors
#pragma once

#include <cstddef>
#include <utility>
#include <vector>

struct TransformerConfig {
    std::size_t hidden = 64;
    std::size_t vocab_size = 32000;
};

class TransformerBlock {
    public:
        explicit TransformerBlock(const TransformerConfig& cfg) : cfg_(cfg) {}

    std::pair<std::vector<float>, std::vector<float>> encode_token(int token_id, std::size_t step) const;

    private:
    TransformerConfig cfg_;
};

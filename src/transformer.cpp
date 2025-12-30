#include "transformer.h"

// very simple sinusoidal token encoder for synthetic key-value generation
#include <cmath>

std::pair<std::vector<float>, std::vector<float>> TransformerBlock::encode_token(int token_id, std::size_t step) const {
    std::size_t hidden_stride = cfg_.hidden * cfg_.layers;
    std::vector<float> key(hidden_stride);   // output key vector
    std::vector<float> value(hidden_stride); // output value vector
    float base =
        static_cast<float>(token_id % cfg_.vocab_size) / static_cast<float>(cfg_.vocab_size); // token dependent phase
    for (std::size_t layer = 0; layer < cfg_.layers; ++layer) {
        float layer_shift = static_cast<float>(layer) * 0.1f; // make layers distinct
        for (std::size_t i = 0; i < cfg_.hidden; ++i) {
            std::size_t idx = layer * cfg_.hidden + i;
            float pos = static_cast<float>(step) * 0.01f + static_cast<float>(i) * 0.001f + layer_shift;
            key[idx] = std::sin(base + pos);              // deterministic key component
            value[idx] = std::cos(base + pos * 1.3f);     // deterministic value component
        }
    }
    return {std::move(key), std::move(value)};
};

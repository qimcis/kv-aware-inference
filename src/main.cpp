#include "instrumentation.h"
#include "kv_cache.h"
#include "transformer.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// helper to make synthetic token labels readable
std::string format_token_string(int token_id) {
    return "tok_" + std::to_string(token_id);
}

// convert cache policy enum to a printable string for logs
std::string policy_name(CachePolicy p) {
    switch (p) {
    case CachePolicy::kLRU:
        return "lru";
    case CachePolicy::kSlidingWindow:
        return "window";
    case CachePolicy::kLFU:
        return "lfu";
    case CachePolicy::kCost:
        return "cost";
    }
    return "unknown";
}

// aggregate all run time options from cli flags
struct RunConfig {
    std::size_t batch = 2;
    std::size_t prefill_tokens = 64;
    std::size_t decode_tokens = 32;
    std::size_t block_size = 16;
    std::size_t max_blocks = 32;
    std::size_t hidden = 64;
    std::string json_path;
    std::vector<std::string> prompt_tokens;
    CachePolicy policy = CachePolicy::kLRU;
    std::size_t window_size = 0;
};

// split a raw prompt string into whitespace delimited tokens
std::vector<std::string> split_prompt(const std::string& prompt) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : prompt) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        out.push_back(cur);
    }
    return out;
}

// parse the policy flag into the corresponding enum value
CachePolicy parse_policy(const std::string& s) {
    if (s == "lru") {
        return CachePolicy::kLRU;
    }
    if (s == "window") {
        return CachePolicy::kSlidingWindow;
    }
    if (s == "lfu") {
        return CachePolicy::kLFU;
    }
    if (s == "cost") {
        return CachePolicy::kCost;
    }
    throw std::runtime_error("unknown policy: " + s);
}

// cli parser that populates runconfig and validates required values
RunConfig parse_args(int argc, char** argv) {
    RunConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + arg);
            }
            return argv[++i];
        };
        if (arg == "--batch") {
            cfg.batch = std::stoul(next());
        } else if (arg == "--prefill") {
            cfg.prefill_tokens = std::stoul(next());
        } else if (arg == "--decode") {
            cfg.decode_tokens = std::stoul(next());
        } else if (arg == "--block-size") {
            cfg.block_size = std::stoul(next());
        } else if (arg == "--max-blocks") {
            cfg.max_blocks = std::stoul(next());
        } else if (arg == "--hidden") {
            cfg.hidden = std::stoul(next());
        } else if (arg == "--log-json") {
            cfg.json_path = next();
        } else if (arg == "--prompt") {
            cfg.prompt_tokens = split_prompt(next());
        } else if (arg == "--policy") {
            cfg.policy = parse_policy(next());
        } else if (arg == "--window") {
            cfg.window_size = std::stoul(next());
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (!cfg.prompt_tokens.empty()) {
        cfg.prefill_tokens = cfg.prompt_tokens.size();
    }
    return cfg;
}

// execute a full run using the synthetic kv path
void run_simulation(const RunConfig& run) {
    Instrumentation instr; // collects detailed cache events
    instr.set_run_meta({run.block_size, run.max_blocks, run.hidden, 0, run.batch, false});
    KVCache kv_cache({run.block_size, run.max_blocks, run.hidden, run.policy, run.window_size}, instr); // block cache
    NaiveCache naive(run.hidden, instr); // straight line baseline

    TransformerBlock block({run.hidden, 32000}); // synthetic token emitter
    auto simulate = [&](bool decode_phase, std::size_t step, std::size_t token_index) {
        for (std::size_t b = 0; b < run.batch; ++b) {
            int token_id;
            std::string token_text;
            if (!run.prompt_tokens.empty() && token_index < run.prompt_tokens.size()) {
                token_text = run.prompt_tokens[token_index];
                token_id = static_cast<int>(std::hash<std::string>{}(token_text) % 1000000);
            } else {
                token_id = static_cast<int>((b * 31 + token_index) % 9973);
                token_text = format_token_string(token_id);
            }
            auto kv = block.encode_token(token_id, step); // produce deterministic k v
            kv_cache.store_token(b, token_index, token_id, token_text, kv.first, kv.second, decode_phase); // block cache write
            naive.store(b, token_index, kv.first, kv.second); // baseline append
        }
    };

    for (std::size_t t = 0; t < run.prefill_tokens; ++t) {
        simulate(false, t, t); // prefill uses the same step token index
    }
    for (std::size_t t = 0; t < run.decode_tokens; ++t) {
        simulate(true, run.prefill_tokens + t, run.prefill_tokens + t); // decode steps advance the index
    }

    kv_cache.synchronize(); // ensure transfers complete before reporting

    std::cout << "KV-aware inference demo (synthetic K/V)\n";
    std::cout << "batch=" << run.batch << " prefill=" << run.prefill_tokens
              << " decode=" << run.decode_tokens << " block_size=" << run.block_size
              << " max_blocks=" << run.max_blocks << " hidden=" << run.hidden
              << " policy=" << policy_name(run.policy);
    if (run.policy == CachePolicy::kSlidingWindow && run.window_size > 0) {
        std::cout << " window=" << run.window_size;
    }
    std::cout << "\n";
    std::cout << "Block bytes (K+V): " << kv_cache.bytes_per_block() << "\n";
    std::cout << "Events summary: " << instr.summary() << "\n";
    if (!run.json_path.empty()) {
        std::filesystem::path path(run.json_path);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        instr.write_json(run.json_path);
        std::cout << "Wrote JSON log: " << run.json_path << "\n";
    }
    std::cout << "Detailed timeline:\n";
    instr.dump(std::cout);
}

int main(int argc, char** argv) {
    try {
        RunConfig cfg = parse_args(argc, argv);
        run_simulation(cfg);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

#include "simulator.h"
#include "trace.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

DecodePolicy parse_decode_policy(const std::string& s) {
    if (s == "fcfs") {
        return DecodePolicy::kFCFS;
    }
    if (s == "slo") {
        return DecodePolicy::kSLO;
    }
    if (s == "srpt") {
        return DecodePolicy::kSRPT;
    }
    throw std::runtime_error("unknown decode policy: " + s);
}

CacheSimPolicy parse_cache_policy(const std::string& s) {
    if (s == "lru") {
        return CacheSimPolicy::kLRU;
    }
    if (s == "window") {
        return CacheSimPolicy::kSlidingWindow;
    }
    if (s == "lfu") {
        return CacheSimPolicy::kLFU;
    }
    if (s == "cost") {
        return CacheSimPolicy::kCost;
    }
    throw std::runtime_error("unknown cache policy: " + s);
}

void write_json(const SimResult& res, const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open output path: " + path);
    }
    out << "{";
    out << "\"makespan_ms\":" << res.makespan_ms << ",";
    out << "\"latency_ms\":{\"p50\":" << res.latency_p50_ms << ",\"p90\":" << res.latency_p90_ms
        << ",\"p95\":" << res.latency_p95_ms << ",\"p99\":" << res.latency_p99_ms << "},";
    out << "\"ttft_ms\":{\"p50\":" << res.ttft_p50_ms << "},";
    out << "\"tpot_ms\":{\"p50\":" << res.tpot_p50_ms << "},";
    out << "\"throughput\":{\"rps\":" << res.throughput_rps
        << ",\"output_tokens_per_s\":" << res.throughput_tokens_per_s << "},";
    out << "\"prefill_hit_rate\":" << res.prefill_hit_rate << ",";
    out << "\"requests\":[";
    for (std::size_t i = 0; i < res.requests.size(); ++i) {
        const auto& r = res.requests[i];
        out << "{";
        out << "\"arrival_ms\":" << r.arrival_ms << ",";
        out << "\"prefill_start_ms\":" << r.prefill_start_ms << ",";
        out << "\"prefill_end_ms\":" << r.prefill_end_ms << ",";
        out << "\"decode_start_ms\":" << r.decode_start_ms << ",";
        out << "\"first_token_ms\":" << r.first_token_ms << ",";
        out << "\"completion_ms\":" << r.completion_ms << ",";
        out << "\"prompt_tokens\":" << r.prompt_tokens << ",";
        out << "\"gen_tokens\":" << r.gen_tokens << ",";
        out << "\"cached_prefix_tokens\":" << r.cached_prefix_tokens;
        out << "}";
        if (i + 1 < res.requests.size()) {
            out << ",";
        }
    }
    out << "]";
    out << "}\n";
}

void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " --trace path.jsonl [options]\n"
              << "Options:\n"
              << "  --prefill-rate TOKENS_PER_S   (default 5000)\n"
              << "  --decode-rate TOKENS_PER_S    (default 8000)\n"
              << "  --max-batch N                 (default 4)\n"
              << "  --decode-chunk N              (default 16)\n"
              << "  --prefill-priority FLOAT      (0..1, default 0.5)\n"
              << "  --decode-policy fcfs|slo|srpt (default fcfs)\n"
              << "  --cache-policy lru|window|lfu|cost (default lru)\n"
              << "  --cache-block N               (tokens per block, default 16)\n"
              << "  --cache-capacity N            (tokens total, default 0=unbounded)\n"
              << "  --cache-decay FLOAT           (default 0.9)\n"
              << "  --out path.json               (optional JSON output)\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }
    std::string trace_path;
    std::string out_path;
    SimConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + flag);
            }
            return argv[++i];
        };
        if (arg == "--trace") {
            trace_path = next(arg);
        } else if (arg == "--prefill-rate") {
            cfg.prefill_tokens_per_s = std::stod(next(arg));
        } else if (arg == "--decode-rate") {
            cfg.decode_tokens_per_s = std::stod(next(arg));
        } else if (arg == "--max-batch") {
            cfg.max_batch = static_cast<std::size_t>(std::stoul(next(arg)));
        } else if (arg == "--decode-chunk") {
            cfg.decode_chunk_tokens = static_cast<std::size_t>(std::stoul(next(arg)));
        } else if (arg == "--prefill-priority") {
            cfg.prefill_priority = std::stod(next(arg));
        } else if (arg == "--decode-policy") {
            cfg.decode_policy = parse_decode_policy(next(arg));
        } else if (arg == "--cache-policy") {
            cfg.cache_policy = parse_cache_policy(next(arg));
        } else if (arg == "--cache-block") {
            cfg.cache_block_tokens = static_cast<std::size_t>(std::stoul(next(arg)));
        } else if (arg == "--cache-capacity") {
            cfg.cache_capacity_tokens = static_cast<std::size_t>(std::stoul(next(arg)));
        } else if (arg == "--cache-decay") {
            cfg.cache_decay = std::stod(next(arg));
        } else if (arg == "--out") {
            out_path = next(arg);
        } else {
            usage(argv[0]);
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (trace_path.empty()) {
        usage(argv[0]);
        throw std::runtime_error("missing --trace");
    }

    TraceLoadStats stats;
    auto trace = load_trace_jsonl(trace_path, &stats);
    if (trace.empty()) {
        std::cerr << "Trace is empty or failed to parse\n";
        return 1;
    }
    std::cerr << "Loaded " << stats.parsed << "/" << stats.total_lines << " requests from "
              << trace_path << " (skipped " << stats.skipped << ")\n";

    SimResult res = run_token_time_sim(trace, cfg);
    std::cout << "Makespan: " << res.makespan_ms << " ms\n";
    std::cout << "Latency p50/p90/p95/p99: " << res.latency_p50_ms << " / " << res.latency_p90_ms
              << " / " << res.latency_p95_ms << " / " << res.latency_p99_ms << " ms\n";
    std::cout << "TTFT p50: " << res.ttft_p50_ms << " ms  TPOT p50: " << res.tpot_p50_ms << " ms\n";
    std::cout << "Throughput: " << res.throughput_rps << " req/s, "
              << res.throughput_tokens_per_s << " tokens/s\n";
    std::cout << "Prefill hit rate: " << res.prefill_hit_rate << "\n";
    if (!out_path.empty()) {
        write_json(res, out_path);
        std::cout << "Wrote JSON: " << out_path << "\n";
    }
    return 0;
}

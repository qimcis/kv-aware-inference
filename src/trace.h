#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

// request representation parsed from a JSONL trace.
struct TraceRequest {
    std::string request_id;
    double arrival_ms = 0.0;
    std::size_t prompt_tokens = 0;
    std::size_t gen_tokens = 0;
    std::optional<std::string> prefix_id;
    std::optional<std::size_t> prefix_tokens;
    std::optional<std::string> tenant_id;
    std::optional<std::string> model_id;
    std::optional<double> slo_ms;
};

struct TraceLoadStats {
    std::size_t total_lines = 0;
    std::size_t parsed = 0;
    std::size_t skipped = 0;
};

// load a JSONL trace file containing flat objects with the fields above.
// this is a parser that assumes the generator uses simple string/number values
std::vector<TraceRequest> load_trace_jsonl(const std::string& path, TraceLoadStats* stats = nullptr);

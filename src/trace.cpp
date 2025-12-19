#include "trace.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <type_traits>

namespace {

std::string trim(const std::string& s) {
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    auto begin = std::find_if_not(s.begin(), s.end(), is_space);
    auto end = std::find_if_not(s.rbegin(), s.rend(), is_space).base();
    if (begin >= end) {
        return std::string();
    }
    return std::string(begin, end);
}

// small JSON extractor for flat key/value pairs where values are numbers or unescaped strings
std::unordered_map<std::string, std::string> parse_flat_json(const std::string& line) {
    std::unordered_map<std::string, std::string> out;
    static const std::regex pair_re(R"xx("([^"]+)"\s*:\s*("([^"\\]*)"|[-0-9.eE+]+|true|false|null))xx");
    for (auto it = std::sregex_iterator(line.begin(), line.end(), pair_re);
         it != std::sregex_iterator(); ++it) {
        const auto& m = *it;
        std::string key = m[1].str();
        std::string value;
        if (m[3].matched) {
            value = m[3].str(); // inner string contents
        } else {
            value = m[2].str(); // raw number/boolean/null
        }
        out[key] = value;
    }
    return out;
}

template <typename T>
bool parse_number(const std::unordered_map<std::string, std::string>& kv, const std::string& key,
                  T& out) {
    auto it = kv.find(key);
    if (it == kv.end()) {
        return false;
    }
    try {
        if constexpr (std::is_same<T, double>::value) {
            out = std::stod(it->second);
        } else {
            out = static_cast<T>(std::stoull(it->second));
        }
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

} // namespace

std::vector<TraceRequest> load_trace_jsonl(const std::string& path, TraceLoadStats* stats) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open trace: " + path);
    }

    std::vector<TraceRequest> out;
    std::string line;
    TraceLoadStats local_stats;

    while (std::getline(in, line)) {
        local_stats.total_lines++;
        line = trim(line);
        if (line.empty()) {
            local_stats.skipped++;
            continue;
        }

        auto kv = parse_flat_json(line);
        TraceRequest req;
        if (kv.count("request_id") == 0 || kv.count("t_arrival_ms") == 0 ||
            kv.count("prompt_tokens") == 0 || kv.count("gen_tokens") == 0) {
            local_stats.skipped++;
            continue;
        }

        req.request_id = kv["request_id"];
        if (!parse_number(kv, "t_arrival_ms", req.arrival_ms)) {
            local_stats.skipped++;
            continue;
        }
        if (!parse_number(kv, "prompt_tokens", req.prompt_tokens)) {
            local_stats.skipped++;
            continue;
        }
        if (!parse_number(kv, "gen_tokens", req.gen_tokens)) {
            local_stats.skipped++;
            continue;
        }

        auto pit = kv.find("prefix_id");
        if (pit != kv.end() && !pit->second.empty() && pit->second != "null") {
            req.prefix_id = pit->second;
        }
        std::size_t prefix_tokens = 0;
        if (parse_number(kv, "prefix_tokens", prefix_tokens)) {
            req.prefix_tokens = prefix_tokens;
        }

        auto tit = kv.find("tenant_id");
        if (tit != kv.end() && !tit->second.empty() && tit->second != "null") {
            req.tenant_id = tit->second;
        }
        auto mit = kv.find("model_id");
        if (mit != kv.end() && !mit->second.empty() && mit->second != "null") {
            req.model_id = mit->second;
        }
        double slo = 0.0;
        if (parse_number(kv, "slo_ms", slo)) {
            req.slo_ms = slo;
        }

        out.push_back(std::move(req));
        local_stats.parsed++;
    }

    if (stats) {
        *stats = local_stats;
    }
    return out;
}

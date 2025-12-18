#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// event categories for timeline logging
enum class EventType {
    kAllocate,
    kReuse,
    kEvict,
    kTransfer,
    kNaiveWrite,
    kPrefillStep,
    kDecodeStep
};

struct Event {
    EventType type;
    std::string label;
    std::size_t batch_id = 0;
    std::size_t block_id = 0;
    std::size_t token_index = 0;
    std::size_t bytes = 0;
    int token_id = -1;
    std::string token_text;
    bool decode = false;
    std::chrono::steady_clock::time_point timestamp;
};

struct TokenEvent {
    std::string kind; // place or evict token
    std::size_t batch_id = 0;
    std::size_t block_id = 0;
    std::size_t block_offset = 0;
    std::size_t token_index = 0;
    int token_id = -1;
    std::string token_text;
    bool decode = false;
    std::chrono::steady_clock::time_point timestamp;
};

// attention snapshot for a single head at a given query index
struct AttentionEvent {
    std::size_t batch_id = 0;
    std::size_t query_index = 0;
    std::size_t head = 0;
    bool decode = false;
    std::vector<float> scores;
    std::chrono::steady_clock::time_point timestamp;
};

struct RunMeta {
    std::size_t block_size = 0;
    std::size_t max_blocks = 0;
    std::size_t hidden = 0;
    std::size_t num_heads = 0;
    std::size_t batch = 0;
    bool real_layer = false;
};

class Instrumentation {
  public:
    void set_run_meta(const RunMeta& meta) {
        std::lock_guard<std::mutex> guard(mu_);
        meta_ = meta;
    }

    void log(EventType type, const std::string& label, std::size_t batch_id,
             std::size_t block_id, std::size_t token_index, std::size_t bytes,
             int token_id = -1, const std::string& token_text = std::string(), bool decode = false) {
        std::lock_guard<std::mutex> guard(mu_);
        events_.push_back(
            Event{type, label, batch_id, block_id, token_index, bytes, token_id, token_text, decode,
                  std::chrono::steady_clock::now()});
    }

    void log_token_event(const std::string& kind, std::size_t batch_id, std::size_t block_id,
                         std::size_t block_offset, std::size_t token_index, int token_id,
                         const std::string& token_text, bool decode) {
        std::lock_guard<std::mutex> guard(mu_);
        token_events_.push_back(TokenEvent{kind,
                                           batch_id,
                                           block_id,
                                           block_offset,
                                           token_index,
                                           token_id,
                                           token_text,
                                           decode,
                                           std::chrono::steady_clock::now()});
    }

    void log_attention(std::size_t batch_id, std::size_t query_index, std::size_t head,
                       const std::vector<float>& scores, bool decode) {
        std::lock_guard<std::mutex> guard(mu_);
        attention_events_.push_back(
            AttentionEvent{batch_id, query_index, head, decode, scores, std::chrono::steady_clock::now()});
    }

    void dump(std::ostream& os) const {
        for (const auto& e : events_) {
            os << "[" << time_since_start(e.timestamp) << "us] "
               << event_name(e.type) << " batch=" << e.batch_id
               << " block=" << e.block_id << " token=" << e.token_index;
            if (!e.label.empty()) {
                os << " " << e.label;
            }
            if (e.bytes) {
                os << " bytes=" << e.bytes;
            }
            if (e.token_id >= 0) {
                os << " id=" << e.token_id;
                if (!e.token_text.empty()) {
                    os << " text=\"" << e.token_text << "\"";
                }
            }
            if (e.type == EventType::kPrefillStep || e.type == EventType::kDecodeStep) {
                os << " phase=" << (e.decode ? "decode" : "prefill");
            }
            os << "\n";
        }

        if (!token_events_.empty()) {
            os << "-- Token placement/eviction --\n";
            for (const auto& t : token_events_) {
                os << "[" << time_since_start(t.timestamp) << "us] TOKEN_" << t.kind
                   << " batch=" << t.batch_id << " block=" << t.block_id
                   << " slot=" << t.block_offset << " token=" << t.token_index
                   << " id=" << t.token_id << " text=\"" << t.token_text << "\""
                   << " phase=" << (t.decode ? "decode" : "prefill") << "\n";
            }
        }

        if (!attention_events_.empty()) {
            os << "-- Attention weights (per head) --\n";
            for (const auto& a : attention_events_) {
                os << "[" << time_since_start(a.timestamp) << "us] ATTN batch=" << a.batch_id
                   << " query=" << a.query_index << " head=" << a.head
                   << " phase=" << (a.decode ? "decode" : "prefill")
                   << " keys=" << a.scores.size() << "\n";
            }
        }
    }

    // summarize counts of cache events.
    std::string summary() const {
        std::size_t allocs = 0, reuse = 0, evicts = 0, transfers = 0, naive = 0;
        for (const auto& e : events_) {
            switch (e.type) {
            case EventType::kAllocate:
                ++allocs;
                break;
            case EventType::kReuse:
                ++reuse;
                break;
            case EventType::kEvict:
                ++evicts;
                break;
            case EventType::kTransfer:
                ++transfers;
                break;
            case EventType::kNaiveWrite:
                ++naive;
                break;
            default:
                break;
            }
        }
        std::ostringstream oss;
        oss << "alloc=" << allocs << " reuse=" << reuse << " evict=" << evicts
            << " transfers=" << transfers << " naive_writes=" << naive
            << " total_events=" << events_.size();
        return oss.str();
    }

    void write_json(const std::string& path) const {
        std::lock_guard<std::mutex> guard(mu_);
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("failed to open log file for writing: " + path);
        }

        out << "{";
        out << "\"meta\":{"
            << "\"block_size\":" << meta_.block_size << ","
            << "\"max_blocks\":" << meta_.max_blocks << ","
            << "\"hidden\":" << meta_.hidden << ","
            << "\"num_heads\":" << meta_.num_heads << ","
            << "\"batch\":" << meta_.batch << ","
            << "\"real_layer\":" << (meta_.real_layer ? "true" : "false")
            << "},";

        out << "\"events\":[";
        for (std::size_t i = 0; i < events_.size(); ++i) {
            const auto& e = events_[i];
            out << "{"
                << "\"type\":\"" << event_name(e.type) << "\","
                << "\"label\":\"" << escape(e.label) << "\","
                << "\"batch\":" << e.batch_id << ","
                << "\"block\":" << e.block_id << ","
                << "\"token_index\":" << e.token_index << ","
                << "\"bytes\":" << e.bytes << ","
                << "\"token_id\":" << e.token_id << ","
                << "\"token_text\":\"" << escape(e.token_text) << "\","
                << "\"decode\":" << (e.decode ? "true" : "false") << ","
                << "\"timestamp_us\":" << time_since_start(e.timestamp)
                << "}";
            if (i + 1 < events_.size()) {
                out << ",";
            }
        }
        out << "],";

        out << "\"token_events\":[";
        for (std::size_t i = 0; i < token_events_.size(); ++i) {
            const auto& t = token_events_[i];
            out << "{"
                << "\"kind\":\"" << t.kind << "\","
                << "\"batch\":" << t.batch_id << ","
                << "\"block\":" << t.block_id << ","
                << "\"block_offset\":" << t.block_offset << ","
                << "\"token_index\":" << t.token_index << ","
                << "\"token_id\":" << t.token_id << ","
                << "\"token_text\":\"" << escape(t.token_text) << "\","
                << "\"decode\":" << (t.decode ? "true" : "false") << ","
                << "\"timestamp_us\":" << time_since_start(t.timestamp)
                << "}";
            if (i + 1 < token_events_.size()) {
                out << ",";
            }
        }
        out << "],";

        out << "\"attention_events\":[";
        for (std::size_t i = 0; i < attention_events_.size(); ++i) {
            const auto& a = attention_events_[i];
            out << "{"
                << "\"batch\":" << a.batch_id << ","
                << "\"query_index\":" << a.query_index << ","
                << "\"head\":" << a.head << ","
                << "\"decode\":" << (a.decode ? "true" : "false") << ","
                << "\"timestamp_us\":" << time_since_start(a.timestamp) << ",";
            out << "\"scores\":[";
            for (std::size_t j = 0; j < a.scores.size(); ++j) {
                out << sanitize_number(static_cast<double>(a.scores[j]));
                if (j + 1 < a.scores.size()) {
                    out << ",";
                }
            }
            out << "]}";
            if (i + 1 < attention_events_.size()) {
                out << ",";
            }
        }
        out << "]";

        out << "}\n";
    }

  private:
    std::string event_name(EventType t) const {
        switch (t) {
        case EventType::kAllocate:
            return "ALLOC";
        case EventType::kReuse:
            return "REUSE";
        case EventType::kEvict:
            return "EVICT";
        case EventType::kTransfer:
            return "TRANSFER";
        case EventType::kNaiveWrite:
            return "NAIVE_WRITE";
        case EventType::kPrefillStep:
            return "PREFILL";
        case EventType::kDecodeStep:
            return "DECODE";
        }
        return "UNKNOWN";
    }

    double sanitize_number(double v) const {
        if (std::isfinite(v)) {
            return v;
        }
        return 0.0;
    }

    std::string escape(const std::string& s) const {
        std::ostringstream oss;
        for (char c : s) {
            switch (c) {
            case '\"':
                oss << "\\\"";
                break;
            case '\\':
                oss << "\\\\";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                oss << c;
                break;
            }
        }
        return oss.str();
    }

    long long time_since_start(std::chrono::steady_clock::time_point t) const {
        return std::chrono::duration_cast<std::chrono::microseconds>(t - start_).count();
    }

    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
    std::vector<Event> events_;
    std::vector<TokenEvent> token_events_;
    std::vector<AttentionEvent> attention_events_;
    RunMeta meta_;
    mutable std::mutex mu_;
};

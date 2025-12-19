#include "simulator.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>

namespace {

struct RequestState {
    TraceRequest req;
    RequestTimeline timeline;
    std::size_t prefill_remaining = 0;
    std::size_t decode_remaining = 0;
    bool prefill_enqueued = false;
    bool decode_enqueued = false;
};

struct EventCompare {
    bool operator()(const SimEvent& a, const SimEvent& b) const { return a.time_ms > b.time_ms; }
};

double tokens_to_ms(std::size_t tokens, double rate_tokens_per_s) {
    if (rate_tokens_per_s <= 0.0) {
        throw std::runtime_error("token rate must be > 0");
    }
    return (static_cast<double>(tokens) / rate_tokens_per_s) * 1000.0;
}

// naive prefix cache: unbounded map of prefix_id -> cached tokens.
std::size_t lookup_prefix(const TraceRequest& req, const std::unordered_map<std::string, std::size_t>& cache) {
    if (!req.prefix_id.has_value()) {
        return 0;
    }
    auto it = cache.find(*req.prefix_id);
    if (it == cache.end()) {
        return 0;
    }
    return std::min<std::size_t>(req.prompt_tokens, it->second);
}

void record_prefix(const TraceRequest& req, std::unordered_map<std::string, std::size_t>& cache) {
    if (!req.prefix_id.has_value()) {
        return;
    }
    cache[*req.prefix_id] = req.prefix_tokens.value_or(req.prompt_tokens);
}

int select_decode(std::deque<int>& decode_queue, double now_ms, const SimConfig& cfg,
                  const std::vector<RequestState>& state) {
    if (decode_queue.empty()) {
        return -1;
    }
    if (cfg.decode_policy == DecodePolicy::kFCFS) {
        int idx = decode_queue.front();
        decode_queue.pop_front();
        return idx;
    }

    double best = std::numeric_limits<double>::infinity();
    std::size_t best_pos = 0;
    for (std::size_t pos = 0; pos < decode_queue.size(); ++pos) {
        int idx = decode_queue[pos];
        const auto& r = state[static_cast<std::size_t>(idx)];
        double score = 0.0;
        switch (cfg.decode_policy) {
        case DecodePolicy::kSLO: {
            double deadline = r.req.slo_ms.has_value()
                                  ? (r.timeline.arrival_ms + *r.req.slo_ms)
                                  : std::numeric_limits<double>::infinity();
            score = deadline - now_ms; // lower slack → higher priority
            break;
        }
        case DecodePolicy::kSRPT:
            score = static_cast<double>(r.decode_remaining);
            break;
        case DecodePolicy::kFCFS:
            score = static_cast<double>(pos);
            break;
        }
        if (score < best) {
            best = score;
            best_pos = pos;
        }
    }
    int idx = decode_queue[best_pos];
    decode_queue.erase(decode_queue.begin() + static_cast<std::ptrdiff_t>(best_pos));
    return idx;
}

}

SimResult run_token_time_sim(const std::vector<TraceRequest>& raw_trace, const SimConfig& cfg) {
    if (raw_trace.empty()) {
        return {};
    }

    // sort by arrival for deterministic ordering
    std::vector<TraceRequest> trace = raw_trace;
    std::stable_sort(trace.begin(), trace.end(), [](const auto& a, const auto& b) {
        return a.arrival_ms < b.arrival_ms;
    });

    std::vector<RequestState> state(trace.size());
    std::unordered_map<std::string, std::size_t> prefix_cache;

    SimResult result;
    result.requests.resize(trace.size());

    for (std::size_t i = 0; i < trace.size(); ++i) {
        const auto& req = trace[i];
        std::size_t hit = lookup_prefix(req, prefix_cache);
        state[i].req = req;
        state[i].prefill_remaining = req.prompt_tokens - hit;
        state[i].decode_remaining = req.gen_tokens;
        state[i].timeline.arrival_ms = req.arrival_ms;
        state[i].timeline.prompt_tokens = req.prompt_tokens;
        state[i].timeline.gen_tokens = req.gen_tokens;
        state[i].timeline.cached_prefix_tokens = hit;
        result.total_prefill_tokens += req.prompt_tokens;
        result.total_decode_tokens += req.gen_tokens;
        result.total_prefill_saved += hit;
        record_prefix(req, prefix_cache); // seed cache for future arrivals
    }

    std::priority_queue<SimEvent, std::vector<SimEvent>, EventCompare> events;
    for (std::size_t i = 0; i < trace.size(); ++i) {
        events.push(SimEvent{trace[i].arrival_ms, SimEventType::kArrival, {static_cast<int>(i)}});
    }

    std::deque<int> prefill_queue;
    std::deque<int> decode_queue;
    bool prefill_busy = false;
    bool decode_busy = false;
    double now_ms = 0.0;

    auto schedule_prefill = [&](double t_ms) {
        if (prefill_busy || prefill_queue.empty()) {
            return;
        }
        std::vector<int> batch;
        std::size_t tokens = 0;
        while (!prefill_queue.empty() && batch.size() < cfg.max_batch) {
            int idx = prefill_queue.front();
            prefill_queue.pop_front();
            auto& r = state[static_cast<std::size_t>(idx)];
            if (r.prefill_remaining == 0) {
                // No prefill work; promote straight to decode.
                r.timeline.prefill_start_ms = r.timeline.arrival_ms;
                r.timeline.prefill_end_ms = r.timeline.arrival_ms;
                decode_queue.push_back(idx);
                continue;
            }
            batch.push_back(idx);
            tokens += r.prefill_remaining;
            if (r.timeline.prefill_start_ms < 0.0) {
                r.timeline.prefill_start_ms = t_ms;
            }
        }
        if (batch.empty()) {
            return;
        }
        double duration = tokens_to_ms(tokens, cfg.prefill_tokens_per_s);
        double done_time = t_ms + duration;
        events.push(SimEvent{done_time, SimEventType::kPrefillDone, batch});
        prefill_busy = true;
    };

    auto schedule_decode = [&](double t_ms) {
        if (decode_busy || decode_queue.empty()) {
            return;
        }
        std::vector<int> batch;
        std::size_t tokens = 0;
        while (!decode_queue.empty() && batch.size() < cfg.max_batch) {
            int idx = select_decode(decode_queue, t_ms, cfg, state);
            if (idx < 0) {
                break;
            }
            auto& r = state[static_cast<std::size_t>(idx)];
            if (r.decode_remaining == 0) {
                continue;
            }
            std::size_t chunk =
                std::min<std::size_t>(cfg.decode_chunk_tokens, r.decode_remaining);
            r.decode_remaining -= chunk;
            tokens += chunk;
            batch.push_back(idx);
            if (r.timeline.decode_start_ms < 0.0) {
                r.timeline.decode_start_ms = t_ms;
            }
        }
        if (batch.empty()) {
            return;
        }
        double duration = tokens_to_ms(tokens, cfg.decode_tokens_per_s);
        double done_time = t_ms + duration;
        events.push(SimEvent{done_time, SimEventType::kDecodeChunkDone, batch});
        decode_busy = true;
    };

    auto schedule_both = [&](double t_ms) {
        if (cfg.prefill_priority > 0.5) {
            schedule_prefill(t_ms);
            schedule_decode(t_ms);
        } else if (cfg.prefill_priority < 0.5) {
            schedule_decode(t_ms);
            schedule_prefill(t_ms);
        } else {
            schedule_prefill(t_ms);
            schedule_decode(t_ms);
        }
    };

    while (!events.empty()) {
        SimEvent ev = events.top();
        events.pop();
        now_ms = ev.time_ms;

        switch (ev.type) {
        case SimEventType::kArrival: {
            for (int idx : ev.req_indices) {
                prefill_queue.push_back(idx);
                state[static_cast<std::size_t>(idx)].prefill_enqueued = true;
            }
            schedule_both(now_ms);
            break;
        }
        case SimEventType::kPrefillDone: {
            prefill_busy = false;
            for (int idx : ev.req_indices) {
                auto& r = state[static_cast<std::size_t>(idx)];
                r.prefill_remaining = 0;
                r.timeline.prefill_end_ms = now_ms;
                decode_queue.push_back(idx);
            }
            schedule_both(now_ms);
            break;
        }
        case SimEventType::kDecodeChunkDone: {
            decode_busy = false;
            for (int idx : ev.req_indices) {
                auto& r = state[static_cast<std::size_t>(idx)];
                if (r.timeline.first_token_ms < 0.0) {
                    r.timeline.first_token_ms = now_ms;
                }
                if (r.decode_remaining == 0) {
                    r.timeline.completion_ms = now_ms;
                } else {
                    decode_queue.push_back(idx);
                }
            }
            schedule_both(now_ms);
            break;
        }
        }
    }

    double makespan = now_ms;
    for (std::size_t i = 0; i < state.size(); ++i) {
        result.requests[i] = state[i].timeline;
        makespan = std::max(makespan, state[i].timeline.completion_ms);
    }
    result.makespan_ms = makespan;
    return result;
}

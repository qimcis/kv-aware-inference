#pragma once

#include "trace.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

// token-time simulator structs
enum class SimEventType { kArrival, kPrefillDone, kDecodeChunkDone };

struct SimEvent {
    double time_ms = 0.0;
    SimEventType type = SimEventType::kArrival;
    std::vector<int> req_indices; // indices into the state vector affected by this event
};

enum class DecodePolicy { kFCFS, kSLO, kSRPT };

struct SimConfig {
    double prefill_tokens_per_s = 5000.0;
    double decode_tokens_per_s = 8000.0;
    std::size_t max_batch = 4;
    std::size_t decode_chunk_tokens = 16; // per-request chunk for decode steps
    double prefill_priority = 0.5;        // >0.5 favors prefill when both are idle
    DecodePolicy decode_policy = DecodePolicy::kFCFS;
};

struct RequestTimeline {
    double arrival_ms = 0.0;
    double prefill_start_ms = -1.0;
    double prefill_end_ms = -1.0;
    double decode_start_ms = -1.0;
    double first_token_ms = -1.0;
    double completion_ms = -1.0;
    std::size_t prompt_tokens = 0;
    std::size_t gen_tokens = 0;
    std::size_t cached_prefix_tokens = 0;
};

struct SimResult {
    std::vector<RequestTimeline> requests;
    double makespan_ms = 0.0;
    std::size_t total_prefill_tokens = 0;
    std::size_t total_decode_tokens = 0;
    std::size_t total_prefill_saved = 0;
};

// run a token-time simulation over a trace, producing per-request timelines.
SimResult run_token_time_sim(const std::vector<TraceRequest>& trace, const SimConfig& cfg);

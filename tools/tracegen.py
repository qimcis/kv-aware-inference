#!/usr/bin/env python3
"""
Generate synthetic JSONL traces with heavy-tail prompt/gen lengths and Zipfian prefix reuse.

Example:
  python tools/tracegen.py --requests 10000 --arrival-rate 40 --out traces/synth.jsonl
"""

import argparse
import json
import os
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Synthetic trace generator for KV-aware simulator."
    )
    p.add_argument(
        "--requests", type=int, default=1000, help="Number of requests to emit."
    )
    p.add_argument(
        "--arrival-rate",
        type=float,
        default=50.0,
        help="Average arrival rate in requests/sec (Poisson process).",
    )
    p.add_argument("--out", required=True, help="Output JSONL path.")
    p.add_argument("--seed", type=int, default=1234, help="RNG seed for repeatability.")
    p.add_argument(
        "--prompt-mean",
        type=float,
        default=4.5,
        help="Lognormal mean for prompt length.",
    )
    p.add_argument(
        "--prompt-sigma",
        type=float,
        default=1.2,
        help="Lognormal sigma for prompt length.",
    )
    p.add_argument(
        "--gen-mean",
        type=float,
        default=3.0,
        help="Lognormal mean for generation length.",
    )
    p.add_argument(
        "--gen-sigma",
        type=float,
        default=0.8,
        help="Lognormal sigma for generation length.",
    )
    p.add_argument(
        "--prefix-cardinality",
        type=int,
        default=500,
        help="Distinct prefixes available for reuse.",
    )
    p.add_argument(
        "--zipf-alpha",
        type=float,
        default=1.2,
        help="Zipf exponent controlling reuse skew (lower = flatter).",
    )
    p.add_argument(
        "--tenant-count", type=int, default=4, help="Number of tenants to tag."
    )
    p.add_argument(
        "--model-count", type=int, default=1, help="Number of models to tag."
    )
    p.add_argument(
        "--base-slo-ms",
        type=float,
        default=0.0,
        help="If >0, attach an SLO per request around this mean (ms).",
    )
    p.add_argument(
        "--slo-jitter-frac",
        type=float,
        default=0.2,
        help="Fractional jitter applied to base SLO (uniform).",
    )
    return p.parse_args()


def sample_arrivals(rng, count, rate_hz):
    """Return monotonically increasing arrival timestamps in ms."""
    if rate_hz <= 0:
        raise ValueError("arrival rate must be > 0")
    gap_mean_ms = 1000.0 / rate_hz
    t = 0.0
    out = []
    for _ in range(count):
        t += rng.exponential(gap_mean_ms)
        out.append(t)
    return out


def sample_lengths(rng, mean, sigma):
    return max(1, int(rng.lognormal(mean, sigma)))


def sample_prefix_id(rng, alpha, cardinality):
    idx = max(0, int(rng.zipf(alpha)) - 1)
    return f"pref_{idx % cardinality:05d}"


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    arrivals = sample_arrivals(rng, args.requests, args.arrival_rate)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for i, t_ms in enumerate(arrivals):
            prompt = sample_lengths(rng, args.prompt_mean, args.prompt_sigma)
            gen = sample_lengths(rng, args.gen_mean, args.gen_sigma)
            prefix_id = sample_prefix_id(rng, args.zipf_alpha, args.prefix_cardinality)
            prefix_tokens = max(1, int(prompt * rng.uniform(0.4, 1.0)))

            slo = None
            if args.base_slo_ms > 0:
                jitter = rng.uniform(-args.slo_jitter_frac, args.slo_jitter_frac)
                slo = max(1.0, args.base_slo_ms * (1.0 + jitter))

            record = {
                "request_id": f"req_{i:06d}",
                "t_arrival_ms": round(t_ms, 3),
                "prompt_tokens": int(prompt),
                "gen_tokens": int(gen),
                "prefix_id": prefix_id,
                "prefix_tokens": int(prefix_tokens),
                "tenant_id": f"tenant_{i % max(1, args.tenant_count)}",
                "model_id": f"model_{i % max(1, args.model_count)}",
            }
            if slo is not None:
                record["slo_ms"] = round(slo, 3)

            f.write(json.dumps(record, separators=(",", ":")))
            f.write("\n")

    print(f"Wrote {args.requests} requests to {args.out}")
    print(
        f"Prompt lengths ~ lognormal(mean={args.prompt_mean}, sigma={args.prompt_sigma})"
    )
    print(
        f"Generation lengths ~ lognormal(mean={args.gen_mean}, sigma={args.gen_sigma})"
    )
    print(
        f"Prefix reuse zipf_alpha={args.zipf_alpha}, cardinality={args.prefix_cardinality}"
    )
    if args.base_slo_ms > 0:
        print(
            f"SLOs around {args.base_slo_ms}ms with ±{args.slo_jitter_frac * 100:.1f}% jitter"
        )


if __name__ == "__main__":
    sys.exit(main())

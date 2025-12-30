#!/usr/bin/env python3
"""
Run a grid of simulator configs over a given trace and save metrics as CSV/JSONL.

Example:
  python tools/run_grid.py --trace traces/synth.jsonl --out-dir runs \
    --loads 0.5 0.7 0.9 --cache-budgets 0 32768 65536 --policies lru lfu cost --decode-policies fcfs slo
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List


def parse_args():
    p = argparse.ArgumentParser(description="Grid runner for token-time simulator.")
    p.add_argument("--trace", required=True, help="Path to JSONL trace.")
    p.add_argument(
        "--sim-bin",
        default="sim_main",
        help="Simulator binary path (default: sim_main).",
    )
    p.add_argument("--out-dir", required=True, help="Directory to write outputs.")
    p.add_argument(
        "--loads",
        nargs="+",
        type=float,
        default=[1.0],
        help="Load multipliers (scale arrival times by 1/load).",
    )
    p.add_argument("--prefill-rate", type=float, default=5000.0)
    p.add_argument("--decode-rate", type=float, default=8000.0)
    p.add_argument("--max-batch", type=int, default=4)
    p.add_argument("--decode-chunk", type=int, default=16)
    p.add_argument("--prefill-priority", nargs="+", type=float, default=[0.5])
    p.add_argument("--decode-policies", nargs="+", default=["fcfs"])
    p.add_argument("--policies", nargs="+", default=["lru"])
    p.add_argument("--cache-budgets", nargs="+", type=int, default=[0])
    p.add_argument("--cache-block", type=int, default=16)
    p.add_argument("--cache-decay", type=float, default=0.9)
    return p.parse_args()


def run_sim(sim_bin: str, trace_path: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        out_path = tmp.name
    cmd = [
        sim_bin,
        "--trace",
        trace_path,
        "--prefill-rate",
        str(cfg["prefill_rate"]),
        "--decode-rate",
        str(cfg["decode_rate"]),
        "--max-batch",
        str(cfg["max_batch"]),
        "--decode-chunk",
        str(cfg["decode_chunk"]),
        "--prefill-priority",
        str(cfg["prefill_priority"]),
        "--decode-policy",
        cfg["decode_policy"],
        "--cache-policy",
        cfg["cache_policy"],
        "--cache-block",
        str(cfg["cache_block"]),
        "--cache-capacity",
        str(cfg["cache_capacity"]),
        "--cache-decay",
        str(cfg["cache_decay"]),
        "--out",
        out_path,
    ]
    env = os.environ.copy()
    env["TRACE_LOAD_SCALE"] = str(cfg["load_scale"])
    subprocess.run(
        cmd, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    with open(out_path, "r", encoding="utf-8") as f:
        res = json.load(f)
    os.remove(out_path)
    return res


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    grid = list(
        itertools.product(
            args.loads,
            args.prefill_priority,
            args.decode_policies,
            args.policies,
            args.cache_budgets,
        )
    )

    rows: List[Dict[str, Any]] = []
    for load, pre_pri, dec_pol, cache_pol, budget in grid:
        cfg = {
            "load_scale": load,
            "prefill_rate": args.prefill_rate * load,
            "decode_rate": args.decode_rate * load,
            "max_batch": args.max_batch,
            "decode_chunk": args.decode_chunk,
            "prefill_priority": pre_pri,
            "decode_policy": dec_pol,
            "cache_policy": cache_pol,
            "cache_block": args.cache_block,
            "cache_capacity": budget,
            "cache_decay": args.cache_decay,
        }
        res = run_sim(args.sim_bin, args.trace, cfg)
        row = {
            "load": load,
            "prefill_priority": pre_pri,
            "decode_policy": dec_pol,
            "cache_policy": cache_pol,
            "cache_budget": budget,
            "makespan_ms": res.get("makespan_ms", 0.0),
            "lat_p50": res.get("latency_ms", {}).get("p50", 0.0),
            "lat_p90": res.get("latency_ms", {}).get("p90", 0.0),
            "lat_p95": res.get("latency_ms", {}).get("p95", 0.0),
            "lat_p99": res.get("latency_ms", {}).get("p99", 0.0),
            "ttft_p50": res.get("ttft_ms", {}).get("p50", 0.0),
            "tpot_p50": res.get("tpot_ms", {}).get("p50", 0.0),
            "throughput_rps": res.get("throughput", {}).get("rps", 0.0),
            "throughput_tokens_s": res.get("throughput", {}).get(
                "output_tokens_per_s", 0.0
            ),
            "prefill_hit_rate": res.get("prefill_hit_rate", 0.0),
        }
        rows.append(row)
        print(
            f"run load={load} prefill_pri={pre_pri} decode={dec_pol} cache={cache_pol} budget={budget} -> lat p99 {row['lat_p99']:.2f} ms"
        )

    csv_path = os.path.join(args.out_dir, "grid.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    jsonl_path = os.path.join(args.out_dir, "grid.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r))
            f.write("\n")
    print(f"Wrote {len(rows)} runs to {csv_path} and {jsonl_path}")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Plot grid results produced by tools/run_grid.py.

Examples:
  python tools/plot_grid.py --grid runs/grid.csv --out-dir runs/plots
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot simulator grid metrics.")
    p.add_argument("--grid", required=True, help="Path to grid CSV from run_grid.py.")
    p.add_argument("--out-dir", required=True, help="Directory to save PNGs.")
    p.add_argument(
        "--metric",
        default="lat_p99",
        help="Metric to plot vs load (default lat_p99). Options include lat_p50, lat_p90, lat_p95, lat_p99, throughput_rps, throughput_tokens_s, prefill_hit_rate.",
    )
    return p.parse_args()


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                    if parsed[k].is_integer():
                        parsed[k] = int(parsed[k])
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)
        return rows


def group_by(rows, keys):
    buckets = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in keys)
        buckets[key].append(r)
    return buckets


def plot_metric_vs_load(rows, metric, out_path):
    groups = group_by(rows, ["cache_policy", "decode_policy", "cache_budget"])
    plt.figure(figsize=(8, 5))
    for key, vals in groups.items():
        vals = sorted(vals, key=lambda x: x["load"])
        loads = [v["load"] for v in vals]
        ys = [v[metric] for v in vals]
        label = f"cache={key[0]} decode={key[1]} budget={key[2]}"
        plt.plot(loads, ys, marker="o", label=label)
    plt.xlabel("Load multiplier")
    plt.ylabel(metric)
    plt.title(f"{metric} vs load")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_hit_vs_budget(rows, out_path):
    groups = group_by(rows, ["load", "cache_policy"])
    plt.figure(figsize=(8, 5))
    for key, vals in groups.items():
        vals = sorted(vals, key=lambda x: x["cache_budget"])
        budgets = [v["cache_budget"] for v in vals]
        hits = [v["prefill_hit_rate"] for v in vals]
        label = f"load={key[0]} cache={key[1]}"
        plt.plot(budgets, hits, marker="o", label=label)
    plt.xlabel("Cache budget (tokens)")
    plt.ylabel("prefill_hit_rate")
    plt.title("Hit rate vs cache budget")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    rows = load_rows(args.grid)
    if not rows:
        raise SystemExit("no rows in grid CSV")
    os.makedirs(args.out_dir, exist_ok=True)
    plot_metric_vs_load(
        rows, args.metric, os.path.join(args.out_dir, f"{args.metric}_vs_load.png")
    )
    plot_hit_vs_budget(rows, os.path.join(args.out_dir, "hit_vs_budget.png"))
    print(f"Wrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()

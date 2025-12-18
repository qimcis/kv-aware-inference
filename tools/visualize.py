#!/usr/bin/env python3
"""
usage:
  python tools/visualize.py --log path/to/log.json --out-dir out/frames
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


def load_log(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def copy_state(state):
    return [list(row) for row in state]


def build_cache_states(log):
    """build per-event cache occupancy snapshots from token events."""
    meta = log.get("meta", {})
    block_size = int(meta.get("block_size", 0))
    max_blocks = int(meta.get("max_blocks", 0))
    events = sorted(log.get("token_events", []), key=lambda e: e.get("timestamp_us", 0))
    if block_size == 0 or max_blocks == 0:
        return [], [], meta
    state = [[None for _ in range(block_size)] for _ in range(max_blocks)]
    states = [copy_state(state)]
    for ev in events:
        block = int(ev.get("block", 0))
        slot = int(ev.get("block_offset", 0))
        if 0 <= block < max_blocks and 0 <= slot < block_size:
            kind = ev.get("kind")
            if kind == "place":
                state[block][slot] = ev
            elif kind in ("evict", "window_evict", "spill_cpu"):
                state[block][slot] = None
        states.append(copy_state(state))
    return events, states, meta


def build_attention_lookup(log):
    return {}, defaultdict(int), 0


def collect_tokens(log):
    """create a map from (batch, token_index) to token event for labels."""
    tokens = defaultdict(dict)
    for ev in log.get("token_events", []):
        if ev.get("kind") != "place":
            continue
        batch = int(ev.get("batch", 0))
        idx = int(ev.get("token_index", 0))
        tokens[batch][idx] = ev
    return tokens


def render_state(ax, state, title=None):
    """draw a block-slot grid with token labels for a given event index."""
    if not state:
        ax.clear()
        ax.text(0.5, 0.5, "No cache events", ha="center", va="center")
        ax.axis("off")
        return
    rows = len(state)
    cols = len(state[0])
    grid = np.full((rows, cols), -1.0)
    labels = [["" for _ in range(cols)] for _ in range(rows)]
    for b in range(rows):
        for s in range(cols):
            ev = state[b][s]
            if ev:
                grid[b, s] = ev.get("token_id", -1)
                labels[b][s] = ev.get("token_text") or f"id{ev.get('token_id', -1)}"
    ax.clear()
    base_cmap = plt.colormaps.get_cmap("viridis")
    cmap = colors.ListedColormap(["#f0f0f0"] + [base_cmap(i) for i in range(base_cmap.N)])
    im = ax.imshow(grid + 1, interpolation="nearest", cmap=cmap, aspect="auto")
    ax.set_xlabel("Block slot")
    ax.set_ylabel("Block id")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    for b in range(rows):
        for s in range(cols):
            if labels[b][s]:
                ax.text(s, b, labels[b][s], ha="center", va="center", fontsize=8)
    ax.set_title(title or "KV blocks")
    return im


def describe_token_event(ev):
    """description for plot titles."""
    if not ev:
        return "initial state"
    kind = ev.get("kind", "")
    token_text = ev.get("token_text", "")
    token_id = ev.get("token_id", -1)
    batch = ev.get("batch", 0)
    block = ev.get("block", 0)
    slot = ev.get("block_offset", 0)
    idx = ev.get("token_index", 0)
    phase = "decode" if ev.get("decode") else "prefill"
    return f"{kind} seq={batch} token_idx={idx} id={token_id} text=\"{token_text}\" block={block} slot={slot} phase={phase}"


def export_static_frames(states, token_events, meta, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, state in enumerate(states):
        ev = token_events[i - 1] if i > 0 and i - 1 < len(token_events) else None
        title = f"Event {i}/{len(states) - 1}: {describe_token_event(ev)}"
        fig, ax = plt.subplots(figsize=(10, 6))
        render_state(ax, state, title=title)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"event_{i:04d}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def render_attention(ax, seq, head, query, lookup, tokens_by_seq):
    ax.clear()
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description="Render KV cache blocks as PNGs.")
    parser.add_argument("--log", required=True, help="Path to log.json produced by kv_aware --log-json.")
    parser.add_argument("--out-dir", required=True, help="Write a PNG per cache event into this directory.")
    args = parser.parse_args()

    log = load_log(args.log)
    token_events, states, meta = build_cache_states(log)
    tokens_by_seq = collect_tokens(log)
    export_static_frames(states, token_events, meta, args.out_dir)
    print(f"Wrote {len(states)} PNGs to {args.out_dir}")


if __name__ == "__main__":
    main()

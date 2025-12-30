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
import numpy as np
from matplotlib import colors
from matplotlib.patches import Rectangle


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
    head_dim = int(meta.get("head_dim", 0))
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
    ax.clear()
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xlabel("Block slot")
    ax.set_ylabel("Block id")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    for b in range(rows):
        for s in range(cols):
            ev = state[b][s]
            if ev:
                phase = ev.get("decode")
                color = "#f08080" if phase else "#87cefa"  # coral vs light blue
                rect = Rectangle((s, b), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)
                label = ev.get("token_text") or f"id{ev.get('token_id', -1)}"
                ax.text(
                    s + 0.5,
                    b + 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                )
            else:
                rect = Rectangle((s, b), 1, 1, facecolor="#f0f0f0", edgecolor="black")
                ax.add_patch(rect)
    ax.invert_yaxis()
    ax.set_title(title or "KV blocks")


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
    return f'{kind} seq={batch} token_idx={idx} id={token_id} text="{token_text}" block={block} slot={slot} phase={phase}'


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


def render_attention_event(ax, ev, meta):
    scores = ev.get("scores", [])
    if not scores:
        ax.text(0.5, 0.5, "No attention scores", ha="center", va="center")
        ax.axis("off")
        return
    arr = np.array(scores, dtype=float)[None, :]
    im = ax.imshow(arr, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_yticks([])
    ax.set_xlabel("Key index")
    batch = ev.get("batch", 0)
    query = ev.get("query_index", 0)
    head = ev.get("head", 0)
    phase = "decode" if ev.get("decode") else "prefill"
    head_dim = meta.get("head_dim", 0)
    title = f"ATTN seq={batch} head={head} q={query} phase={phase}"
    if head_dim:
        title += f" head_dim={head_dim}"
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def export_attention_frames(attn_events, meta, out_dir):
    if not attn_events:
        return 0
    os.makedirs(out_dir, exist_ok=True)
    for i, ev in enumerate(attn_events):
        fig, ax = plt.subplots(figsize=(10, 3))
        render_attention_event(ax, ev, meta)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"attn_{i:04d}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return len(attn_events)


def compute_frequency_matrix(token_events, meta):
    block_size = int(meta.get("block_size", 0))
    max_blocks = int(meta.get("max_blocks", 0))
    if block_size == 0 or max_blocks == 0:
        return None
    freq = np.zeros((max_blocks, block_size), dtype=float)
    for ev in token_events:
        block = int(ev.get("block", -1))
        slot = int(ev.get("block_offset", ev.get("slot", -1)))
        if block < 0 or block >= max_blocks or slot < 0 or slot >= block_size:
            continue
        kind = ev.get("kind", "")
        if kind in ("place", "touch"):
            freq[block, slot] += 1.0
    return freq


def export_frequency_heatmap(token_events, meta, out_dir):
    freq = compute_frequency_matrix(token_events, meta)
    if freq is None or not freq.any():
        return False
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    im = plt.imshow(freq, aspect="auto", cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="touch count")
    plt.xlabel("Block slot")
    plt.ylabel("Block id")
    plt.title("Block-slot access frequency")
    out_path = os.path.join(out_dir, "frequency_heatmap.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def export_comparison_heatmap(lru_freq, lfu_freq, out_dir):
    if lru_freq is None or lfu_freq is None:
        return False
    if lru_freq.shape != lfu_freq.shape:
        return False
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, freq, title in zip(
        axes, [lru_freq, lfu_freq], ["LRU frequency", "LFU frequency"]
    ):
        im = ax.imshow(freq, aspect="auto", cmap="coolwarm")
        ax.set_xlabel("Block slot")
        ax.set_ylabel("Block id")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "frequency_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Render KV cache blocks as PNGs.")
    parser.add_argument(
        "--log", required=True, help="Path to log.json produced by kv_aware --log-json."
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Write a PNG per cache event into this directory.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("LRU_LOG", "LFU_LOG"),
        help="Optional: supply two logs to render side-by-side (policy comparison).",
    )
    args = parser.parse_args()

    def render_single(log_path, suffix):
        log = load_log(log_path)
        token_events, states, meta = build_cache_states(log)
        export_static_frames(
            states, token_events, meta, os.path.join(args.out_dir, suffix)
        )
        attn_events = sorted(
            log.get("attention_events", []), key=lambda e: e.get("timestamp_us", 0)
        )
        attn_written = export_attention_frames(
            attn_events, meta, os.path.join(args.out_dir, suffix)
        )
        freq_written = export_frequency_heatmap(
            token_events, meta, os.path.join(args.out_dir, suffix)
        )
        print(
            f"[{suffix}] wrote {len(states)} cache PNGs to {os.path.join(args.out_dir, suffix)}"
        )
        if attn_written:
            print(f"[{suffix}] wrote {attn_written} attention PNGs")
        if freq_written:
            print(f"[{suffix}] wrote frequency heatmap")
        return token_events, meta

    if args.compare:
        os.makedirs(args.out_dir, exist_ok=True)
        lru_tokens, lru_meta = render_single(args.compare[0], "lru")
        lfu_tokens, lfu_meta = render_single(args.compare[1], "lfu")
        lru_freq = compute_frequency_matrix(lru_tokens, lru_meta)
        lfu_freq = compute_frequency_matrix(lfu_tokens, lfu_meta)
        export_comparison_heatmap(lru_freq, lfu_freq, args.out_dir)
        print(
            f"Comparison complete. Outputs under {args.out_dir}/lru and {args.out_dir}/lfu."
        )
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        render_single(args.log, "")


if __name__ == "__main__":
    main()

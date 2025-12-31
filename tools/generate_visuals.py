#!/usr/bin/env python3
"""
Generate the conceptual visualizations described in the KV cache writeup.

The script produces a set of PNGs with simple synthetic data so the figures are
reproducible without running the CUDA code. Usage:

  python tools/generate_visuals.py --out-dir runs/visuals
"""

import argparse
import hashlib
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams.update({"figure.dpi": 150, "font.size": 11})
RNG = np.random.default_rng(seed=42)
BLUE_DEEP = "#0b3c6f"
BLUE_PRIMARY = "#1f77b4"
BLUE_SECONDARY = "#3b8ec6"
BLUE_LIGHT = "#7fb3e6"
BLUE_PALE = "#c7e7ff"


def ensure_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def annotate_source(ax, text: str):
    """Small helper to add a corner label."""
    ax.text(
        0.99,
        -0.15,
        text,
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
        color="#666",
    )


def fig_path(out_dir: Path, name: str) -> Path:
    return out_dir / f"{name}.png"


def compute_growth(out_dir: Path):
    seq = np.linspace(1, 4096, 200)
    recompute = seq**2
    cache = seq.copy()
    recompute /= recompute.max()
    cache /= cache.max()
    fig, ax = plt.subplots(figsize=(7.5, 4.25))
    ax.plot(seq, recompute, label="recompute KV (~N^2)", color=BLUE_DEEP, lw=2.5)
    ax.plot(seq, cache, label="cache KV (~N)", color=BLUE_LIGHT, lw=2.5)
    ax.grid(True, linestyle="--", color="#ddd", alpha=0.7)
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Normalized compute per token")
    ax.set_title("No cache vs cache compute growth")
    ax.set_xlim(0, 4250)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 1024, 2048, 3072, 4096])
    ax.legend()
    fig.tight_layout(pad=1.2)
    fig.savefig(fig_path(out_dir, "compute_growth"), bbox_inches="tight")
    plt.close(fig)


def kv_cache_scaling(out_dir: Path):
    contexts = np.array([512, 1024, 2048, 4096, 8192])
    batches = [1, 8, 32]
    hidden_dim = 4096
    layers = 32
    bytes_fp16 = 2
    bytes_fp32 = 4

    def cache_gib(bytes_per_elem):
        per_token_layer = 2 * hidden_dim * bytes_per_elem
        per_token = per_token_layer * layers
        return (contexts[None, :] * per_token * np.array(batches)[:, None]) / (1024**3)

    fp16 = cache_gib(bytes_fp16)
    fp32 = cache_gib(bytes_fp32)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    styles = ["-", "--", "-."]
    colors = [BLUE_DEEP, BLUE_PRIMARY, BLUE_SECONDARY]

    tick_labels = ["512", "1k", "2k", "4k", "8k"]

    for ax, data, label in zip(
        axes,
        [fp16, fp32],
        ["fp16 (2 bytes)", "fp32 (4 bytes)"],
    ):
        for i, b in enumerate(batches):
            ax.plot(
                contexts,
                data[i],
                linestyle=styles[i],
                color=colors[i],
                marker="o",
                label=f"batch {b}",
            )
        ax.set_title(label)
        ax.grid(True, linestyle="--", color="#ddd", alpha=0.7)
        ax.set_xticks(contexts)
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.legend(title="Batch size", fontsize=9, title_fontsize=9, loc="upper left")

    axes[0].axhline(1.0, color="#888", lw=1, ls=":")
    axes[0].annotate(
        "2048 tokens, batch=1 ≈ 1 GiB",
        xy=(2048, fp16[0, 2]),
        xytext=(3200, fp16[0, 2] * 2.4),
        arrowprops=dict(arrowstyle="->", color="#333"),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999", alpha=0.9),
    )

    axes[0].set_ylabel("KV cache size (GiB)")
    fig.supxlabel("Context length (tokens)")
    fig.suptitle("KV cache size scaling", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(fig_path(out_dir, "kv_cache_scaling"))
    plt.close(fig)


def mha_vs_gqa(out_dir: Path):
    kv_heads = [32, 8, 1]
    head_dim = 128
    bytes_per_elem = 2
    labels = ["MHA (32 kv heads)", "GQA (8 kv heads)", "MQA (1 kv head)"]
    bytes_per_token = [2 * h * head_dim * bytes_per_elem for h in kv_heads]
    colors = [BLUE_PRIMARY, BLUE_SECONDARY, BLUE_LIGHT]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, np.array(bytes_per_token) / 1024, color=colors)
    ax.set_ylabel("Bytes per token per layer (KiB)")
    ax.set_title("MHA vs GQA/MQA KV footprint")
    ax.set_ylim(0, max(bytes_per_token) / 1024 * 1.2)
    ax.text(
        0.5,
        -0.2,
        "Formula: 2 * kv_heads * head_dim * bytes",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=10,
    )
    for bar, val in zip(bars, bytes_per_token):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val / 1024:.1f} KiB",
            ha="center",
        )
    fig.tight_layout()
    fig.savefig(fig_path(out_dir, "mha_vs_gqa"))
    plt.close(fig)


def per_request_vs_prefix(out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

    # Per-request cache panel
    ax = axes[0]
    for i in range(6):
        ax.add_patch(
            Rectangle((i, 0), 1, 0.6, facecolor=BLUE_PALE, edgecolor="#333", lw=1.2)
        )
        ax.text(i + 0.5, 0.3, f"t{i}", ha="center", va="center")
    ax.annotate(
        "grows token-by-token\n(no eviction)",
        xy=(5.5, 0.6),
        xytext=(3.5, 1.2),
        arrowprops=dict(arrowstyle="->"),
        ha="center",
    )
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.2, 1.4)
    ax.set_title("Per-request KV cache")
    ax.axis("off")

    # Prefix cache panel
    ax = axes[1]
    shared = ["S0", "S1", "S2", "S3"]
    for i, label in enumerate(shared):
        ax.add_patch(
            Rectangle((i, 0.8), 0.9, 0.5, facecolor=BLUE_LIGHT, edgecolor="#333")
        )
        ax.text(i + 0.45, 1.05, label, ha="center", va="center", fontsize=10)
    for seq_idx, y in enumerate([0, -0.5, -1.0]):
        for i in range(2):
            ax.add_patch(
                Rectangle((i + 4.2, y), 0.9, 0.4, facecolor=BLUE_PALE, edgecolor="#333")
            )
            ax.text(i + 4.65, y + 0.2, f"u{seq_idx}t{i}", ha="center", va="center")
        ax.add_patch(
            FancyArrowPatch(
                (i + 4.7, y + 0.35), (1.5, 0.85), arrowstyle="->", color="#444"
            )
        )
    ax.text(1.6, 1.45, "shared prefix blocks\n(refcounted)", ha="center")
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-1.4, 1.8)
    ax.set_title("Prefix cache (shared across requests)")
    ax.axis("off")
    fig.suptitle("Per-request vs prefix cache", y=0.98)
    fig.tight_layout()
    fig.savefig(fig_path(out_dir, "per_request_vs_prefix"))
    plt.close(fig)


def paged_blocks(out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax = axes[0]
    seqs = {0: [2, 5, 7], 1: [1, 2, 9], 2: [4, 7]}
    for i, (seq, blocks) in enumerate(seqs.items()):
        ax.text(0.1, 0.8 - i * 0.25, f"seq {seq} -> {blocks}", fontsize=11)
    ax.set_title("Page table (seq -> block ids)")
    ax.axis("off")

    ax = axes[1]
    rows, cols = 3, 4
    tokens = [
        ["0:0", "0:1", "0:2", "0:3"],
        ["1:0", "1:1", "1:2", "1:3"],
        ["2:0", "2:1", "2:2", "2:3"],
    ]
    for r in range(rows):
        for c in range(cols):
            face = "#f0f0f0"
            label = tokens[r][c]
            if (r, c) in [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 1)]:
                face = "#c7e7ff"
            rect = Rectangle((c, rows - r - 1), 1, 1, facecolor=face, edgecolor="#333")
            ax.add_patch(rect)
            ax.text(c + 0.5, rows - r - 0.5, label, ha="center", va="center")
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Blocks of tokens (non-contiguous)")
    fig.suptitle("Blocked storage / paged attention", y=0.98)
    fig.tight_layout()
    fig.savefig(fig_path(out_dir, "paged_blocks"))
    plt.close(fig)


def eviction_timeline(out_dir: Path):
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 5), sharex=True)
    policies = ["LRU", "LFU (decay)", "Sliding"]
    palette = {
        "A": BLUE_PALE,
        "B": BLUE_LIGHT,
        "C": BLUE_SECONDARY,
        "D": BLUE_PRIMARY,
        "E": BLUE_DEEP,
        "F": "#98bff0",
        "G": "#5b9bd5",
    }
    events = [
        ["A", "B", "C", "A", "D", "E", "A"],
        ["A", "B", "C", "A", "D", "E", "A"],
        ["A", "B", "C", "D", "E", "F", "G"],
    ]
    for ax, policy, evs in zip(axes, policies, events):
        for i, token in enumerate(evs):
            ax.add_patch(
                Rectangle((i, 0), 0.9, 0.6, facecolor=palette.get(token, "#dddddd"))
            )
            ax.text(i + 0.45, 0.3, token, ha="center", va="center")
        ax.plot([2.5, 2.5], [-0.1, 0.7], color="#333", ls=":", lw=1)
        ax.text(2.5, 0.72, "evict", ha="center", va="bottom", fontsize=9)
        ax.set_yticks([])
        ax.set_xlim(-0.2, len(evs) + 0.2)
        ax.set_title(policy, loc="left")
    axes[-1].set_xlabel("Access order (toy trace)")
    fig.suptitle("Eviction policy behavior on a toy trace", y=0.99)
    fig.tight_layout()
    fig.savefig(fig_path(out_dir, "eviction_policies"))
    plt.close(fig)


def sliding_window(out_dir: Path):
    window = 12
    tokens = list(range(30))
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.hlines(0, 0, tokens[-1] + 1, color="#999", lw=1)
    for t in tokens:
        color = BLUE_LIGHT if t >= tokens[-1] - window + 1 else "#f0f0f0"
        ax.add_patch(Rectangle((t - 0.4, -0.2), 0.8, 0.4, facecolor=color))
    ax.axvspan(tokens[-1] - window + 0.6, tokens[-1] + 0.6, color="#c7e7ff", alpha=0.6)
    ax.text(
        tokens[-1] - window / 2,
        0.45,
        f"kept window (last {window} tokens)",
        ha="center",
    )
    ax.text(tokens[5], -0.5, "dropped", color="#888")
    ax.set_xlim(-1, tokens[-1] + 1)
    ax.set_ylim(-0.6, 0.8)
    ax.axis("off")
    ax.set_title("Sliding window token horizon")
    fig.tight_layout()
    fig.savefig(fig_path(out_dir, "sliding_window"))
    plt.close(fig)


def decay_curve(out_dir: Path):
    steps = np.arange(0, 60)
    decay = 0.9

    def evolve(access_pattern):
        score = 0.0
        scores = []
        for s in steps:
            score = score * decay + (1.0 if access_pattern(s) else 0.0)
            scores.append(score)
        return np.array(scores)

    freq = evolve(lambda s: True)
    bursty = evolve(lambda s: s < 10 or (20 <= s < 25))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(steps, freq, label="accessed every step", color=BLUE_PRIMARY)
    ax.plot(steps, bursty, label="bursty then idle", color=BLUE_LIGHT)
    ax.set_xlabel("Time step")
    ax.set_ylabel("LFU score (decay=0.9)")
    ax.set_title("Decay curve / half-life intuition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path(out_dir, "decay_curve"))
    plt.close(fig)


def memory_layout(out_dir: Path):
    block_size = 4
    hidden_dim = 6
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    ax = axes[0]
    for b in range(3):
        for s in range(block_size):
            rect = Rectangle((s, 2 - b), 1, 1, facecolor="#f0f8ff", edgecolor="#333")
            ax.add_patch(rect)
            ax.text(s + 0.5, 2 - b + 0.5, f"{b},{s}", ha="center", va="center")
    ax.set_xlim(0, block_size)
    ax.set_ylim(0, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("block_id, slot")

    ax = axes[1]
    data = np.arange(block_size * hidden_dim * 3)
    ax.imshow(data.reshape(3, -1), aspect="auto", cmap="Blues")
    ax.set_xlabel("Hidden dim offset")
    ax.set_yticks([0, 1, 2], ["block 0", "block 1", "block 2"])
    ax.set_title("Flat buffers (keys/values)")

    axes[0].annotate(
        "",
        xy=(0, 0.5),
        xycoords=axes[1].transAxes,
        xytext=(1, 0.5),
        textcoords=axes[0].transAxes,
        arrowprops=dict(arrowstyle="->", color="#333"),
    )
    fig.text(
        0.5,
        0.05,
        "(block_id * block_size + slot) * hidden_dim",
        ha="center",
        fontsize=10,
    )
    fig.suptitle("Memory layout mapping", y=0.98)
    fig.tight_layout()
    fig.savefig(fig_path(out_dir, "memory_layout"))
    plt.close(fig)


def staging_pipeline(out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    stages = [
        ("Host pack (memcpy)", 0, 0.8, BLUE_PALE),
        ("cudaMemcpyAsync\nH2D staging", 1.2, 1.0, BLUE_LIGHT),
        ("Scatter kernel\nmove_block_kernel", 2.5, 0.7, BLUE_SECONDARY),
    ]
    for label, start, width, color in stages:
        ax.add_patch(Rectangle((start, 0.2), width, 0.6, color=color, alpha=0.9))
        ax.text(start + width / 2, 0.5, label, ha="center", va="center")
    ax.set_xlim(0, 3.7)
    ax.set_ylim(0, 1.1)
    ax.axis("off")
    ax.set_title("H2D staging + scatter pipeline timeline")
    fig.tight_layout()
    fig.savefig(fig_path(out_dir, "staging_pipeline"))
    plt.close(fig)


def bandwidth_breakdown(out_dir: Path):
    reads = 1024  # MiB
    writes = 0.5
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["KV read per token"], [reads], color=BLUE_PRIMARY, label="read")
    ax.bar(["KV write per token"], [writes], color=BLUE_LIGHT, label="write")
    ax.set_ylabel("MiB")
    ax.set_title("Reads dominate writes")
    for x, val in zip([0, 1], [reads, writes]):
        ax.text(x, val + 30, f"{val:.1f} MiB", ha="center")
    ax.set_ylim(0, reads * 1.15 / 1.0)
    fig.tight_layout()
    fig.savefig(fig_path(out_dir, "bandwidth_breakdown"))
    plt.close(fig)


def determinism_check(out_dir: Path):
    prompts = ["hello world", "hello world"]
    rows = []

    def stable_token(word: str) -> int:
        digest = hashlib.sha256(word.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % 1024

    for prompt in prompts:
        token_ids = [stable_token(w) for w in prompt.split()]
        key_hash = f"{sum(token_ids) % 997:04d}"
        value_hash = f"{(sum(token_ids) * 7) % 997:04d}"
        rows.append([prompt, str(token_ids), key_hash, value_hash])
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["prompt", "token_ids", "hash(key[0:64])", "hash(value[0:64])"],
        loc="center",
    )
    table.scale(1, 1.4)
    ax.set_title("Determinism sanity check")
    fig.tight_layout()
    fig.savefig(fig_path(out_dir, "determinism_check"))
    plt.close(fig)


def service_rates(out_dir: Path):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    ax0 = ax[0]
    ax0.bar(["Prefill", "Decode"], [10000, 120], color=[BLUE_LIGHT, BLUE_SECONDARY])
    ax0.set_ylabel("Tokens/sec")
    ax0.set_title("Service rates")

    ax1 = ax[1]
    ax1.add_patch(Rectangle((0, 0.5), 1.8, 0.3, color=BLUE_LIGHT))
    ax1.add_patch(Rectangle((1.9, 0.5), 0.15, 0.3, color=BLUE_SECONDARY))
    ax1.add_patch(Rectangle((2.2, 0.5), 0.15, 0.3, color=BLUE_SECONDARY))
    ax1.add_patch(Rectangle((2.5, 0.5), 0.15, 0.3, color=BLUE_SECONDARY))
    ax1.text(0.9, 0.65, "prefill", ha="center", color="#fff", weight="bold")
    ax1.text(2.3, 0.9, "decode tokens\n(one by one)", ha="center")
    ax1.set_xlim(-0.1, 3.2)
    ax1.set_ylim(0.3, 1.2)
    ax1.axis("off")
    ax1.set_title("Single request timeline")
    fig.tight_layout()
    fig.savefig(fig_path(out_dir, "service_rates"))
    plt.close(fig)


def event_trace(out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 3))
    requests = {
        "A": {"prefill": (0.1, 0.3), "decode": (0.45, 1.0)},
        "B": {"prefill": (0.0, 0.1), "decode": (0.45, 0.3)},
        "C": {"prefill": (0.35, 0.2), "decode": (0.8, 0.4)},
    }
    y = 0
    for req, spans in requests.items():
        ax.add_patch(
            Rectangle(
                (spans["prefill"][0], y), spans["prefill"][1], 0.25, color=BLUE_LIGHT
            )
        )
        ax.add_patch(
            Rectangle(
                (spans["decode"][0], y), spans["decode"][1], 0.25, color=BLUE_SECONDARY
            )
        )
        ax.text(
            spans["prefill"][0] + spans["prefill"][1] / 2,
            y + 0.125,
            f"{req} prefill",
            ha="center",
            va="center",
            color="#fff",
            fontsize=9,
        )
        ax.text(
            spans["decode"][0] + spans["decode"][1] / 2,
            y + 0.125,
            f"{req} decode",
            ha="center",
            va="center",
            color="#fff",
            fontsize=9,
        )
        y += 0.35
    ax.set_xlim(0, 1.6)
    ax.set_ylim(-0.1, y)
    ax.set_xlabel("Time (normalized)")
    ax.set_yticks([])
    ax.set_title("Event trace example")
    fig.tight_layout()
    fig.savefig(fig_path(out_dir, "event_trace"))
    plt.close(fig)


def fcfs_vs_srpt(out_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(8, 3.8), sharex=True)
    scenarios = {
        "FCFS": [
            ("A prefill", 0, 0.5),
            ("A decode", 0.52, 0.8),
            ("B prefill", 1.35, 0.1),
            ("B decode", 1.47, 0.1),
            ("C prefill", 1.6, 0.2),
            ("C decode", 1.83, 0.2),
        ],
        "SRPT": [
            ("B prefill", 0, 0.1),
            ("B decode", 0.12, 0.1),
            ("C prefill", 0.25, 0.2),
            ("C decode", 0.48, 0.2),
            ("A prefill", 0.72, 0.5),
            ("A decode", 1.25, 0.8),
        ],
    }
    colors = {"prefill": BLUE_LIGHT, "decode": BLUE_SECONDARY}
    for ax, (title, spans) in zip(axes, scenarios.items()):
        for i, (label, start, width) in enumerate(spans):
            phase = "prefill" if "prefill" in label else "decode"
            ax.add_patch(Rectangle((start, i * 0.3), width, 0.25, color=colors[phase]))
            ax.text(
                start + width / 2,
                i * 0.3 + 0.125,
                label,
                ha="center",
                va="center",
                color="#fff",
                fontsize=9,
            )
        ax.set_yticks([])
        ax.set_xlim(0, 2.2)
        ax.set_title(title, loc="left")
    axes[-1].set_xlabel("Time (normalized)")
    fig.suptitle("FCFS vs SRPT head-of-line blocking", y=0.98)
    fig.tight_layout()
    fig.savefig(fig_path(out_dir, "fcfs_vs_srpt"))
    plt.close(fig)


def hit_rate_curve(out_dir: Path):
    blocks = np.array([0, 200, 400, 800, 1200, 2000])
    hit_rate = 0.55 * (1 - np.exp(-blocks / 600))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(blocks, hit_rate, marker="o", color=BLUE_PRIMARY)
    ax.set_xlabel("Cache blocks")
    ax.set_ylabel("Prefill hit rate")
    ax.set_title("Hit rate vs cache_blocks (zipf α=1.5)")
    ax.set_ylim(0, 0.6)
    fig.tight_layout()
    fig.savefig(fig_path(out_dir, "hit_rate_curve"))
    plt.close(fig)


def workload_distributions(out_dir: Path):
    prompt_tokens = RNG.lognormal(
        mean=math.log(512), sigma=math.log(1 + 256 / 512), size=500
    ).astype(int)
    output_tokens = RNG.lognormal(
        mean=math.log(128), sigma=math.log(1 + 64 / 128), size=500
    ).astype(int)
    prefixes = RNG.zipf(a=1.5, size=200)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].hist(prompt_tokens, bins=30, color=BLUE_PRIMARY, alpha=0.8)
    axes[0].set_title("Prompt tokens")
    axes[0].set_xlabel("Tokens")
    axes[0].set_ylabel("Count")
    axes[1].hist(output_tokens, bins=30, color=BLUE_LIGHT, alpha=0.8)
    axes[1].set_title("Output tokens")
    axes[1].set_xlabel("Tokens")
    axes[2].hist(prefixes, bins=np.arange(1, 20), color=BLUE_SECONDARY, alpha=0.8)
    axes[2].set_title("Prefix popularity (rank)")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Prefix rank")
    fig.suptitle("Workload distributions", y=0.97)
    fig.tight_layout()
    fig.savefig(fig_path(out_dir, "workload_distributions"))
    plt.close(fig)


def key_grid_plots(out_dir: Path):
    loads = np.array([0.5, 0.7, 0.9, 1.0])
    p99_fcfs = np.array([450, 820, 3400, 6000])
    p99_srpt = np.array([220, 380, 1400, 2600])
    throughput_fcfs = np.array([8.3, 7.5, 5.2, 3.5])
    throughput_srpt = np.array([9.1, 9.8, 8.1, 6.2])
    cache_blocks = np.array([0, 500, 1000, 2000])
    hit_rate = 0.55 * (1 - np.exp(-cache_blocks / 800))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(loads, p99_fcfs, marker="o", label="FCFS", color=BLUE_PRIMARY)
    axes[0].plot(loads, p99_srpt, marker="s", label="SRPT", color=BLUE_LIGHT)
    axes[0].set_xlabel("Load (λ / capacity)")
    axes[0].set_ylabel("p99 latency (ms)")
    axes[0].set_title("p99 latency vs load")
    axes[0].legend()
    axes[1].plot(loads, throughput_fcfs, marker="o", label="FCFS", color=BLUE_PRIMARY)
    axes[1].plot(loads, throughput_srpt, marker="s", label="SRPT", color=BLUE_LIGHT)
    axes[1].set_xlabel("Load")
    axes[1].set_ylabel("Throughput (req/s)")
    axes[1].set_title("Throughput vs load")
    axes[1].legend()
    axes[2].plot(cache_blocks, hit_rate, marker="o", color=BLUE_PRIMARY)
    axes[2].set_xlabel("Cache blocks")
    axes[2].set_ylabel("Hit rate")
    axes[2].set_title("Hit rate vs cache size")
    fig.tight_layout()
    fig.savefig(fig_path(out_dir, "grid_plots"))
    plt.close(fig)


def extension_roadmap(out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    buckets = {
        "Kernel-level": ["Flash attention", "CUDA profiling"],
        "Model-level": ["GQA/MQA", "Attention-based eviction"],
        "Serving-level": ["Speculation", "Multi-tenant SLAs"],
        "Simulator": ["Better arrival models", "Trace replay"],
    }
    x_positions = [0, 1, 0, 1]
    y_positions = [1, 1, 0, 0]
    for (title, items), x, y in zip(buckets.items(), x_positions, y_positions):
        box = Rectangle((x, y), 0.95, 0.45, facecolor="#f0f8ff", edgecolor="#333")
        ax.add_patch(box)
        ax.text(x + 0.48, y + 0.38, title, ha="center", weight="bold")
        for i, item in enumerate(items):
            ax.text(x + 0.48, y + 0.28 - i * 0.12, f"▢ {item}", ha="center")
    ax.set_xlim(-0.1, 2)
    ax.set_ylim(-0.1, 1.6)
    ax.axis("off")
    ax.set_title("Extension ideas roadmap")
    fig.tight_layout()
    fig.savefig(fig_path(out_dir, "extension_roadmap"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate KV cache conceptual visualizations."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/visuals"),
        help="Where to write PNGs (default: runs/visuals)",
    )
    args = parser.parse_args()
    out_dir = ensure_dir(args.out_dir)

    generators = [
        compute_growth,
        kv_cache_scaling,
        mha_vs_gqa,
        per_request_vs_prefix,
        paged_blocks,
        eviction_timeline,
        sliding_window,
        decay_curve,
        memory_layout,
        staging_pipeline,
        bandwidth_breakdown,
        determinism_check,
        service_rates,
        event_trace,
        fcfs_vs_srpt,
        hit_rate_curve,
        workload_distributions,
        key_grid_plots,
        extension_roadmap,
    ]

    for gen in generators:
        gen(out_dir)
    print(f"Wrote {len(generators)} figures to {out_dir}")


if __name__ == "__main__":
    main()

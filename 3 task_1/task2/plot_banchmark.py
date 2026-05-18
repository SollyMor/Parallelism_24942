#!/usr/bin/env python3
"""
Reads CSV from task2_benchmark and draws a comparison plot
for mean run time across server implementations.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

FIGSIZE = (9, 6)
COLOR_BAR = "#4e79a7"
COLOR_BEST = "#d62728"
GRID_KW = {"linestyle": "--", "alpha": 0.4}


def _impl_flag(delivery: str, container: str) -> str:
    d = delivery.strip().lower()
    c = container.strip()
    if d == "slot" and c == "unordered_map":
        return "slot-u"
    if d == "slot" and c in ("std::map", "map"):
        return "slot-o"
    if d == "promise" and c == "unordered_map":
        return "promise-u"
    if d == "promise" and c in ("std::map", "map"):
        return "promise-o"
    return f"{delivery}+{container}"


def load_benchmark_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    required = {"delivery", "container", "seconds_per_run_mean"}
    if not rows:
        print("CSV is empty.", file=sys.stderr)
        sys.exit(1)
    missing = required - set(rows[0].keys())
    if missing:
        print(f"CSV missing columns: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot benchmark comparison for server implementations."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "task2_benchmark_results.csv",
        help="Path to task2_benchmark_results.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "server_benchmark_task2.png",
        help="Output image path (png/pdf/svg)",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Custom chart title",
    )
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    raw = load_benchmark_csv(args.csv)
    labels: list[str] = []
    seconds: list[float] = []
    n_kind: str | None = None
    repeats: str | None = None

    for row in raw:
        labels.append(_impl_flag(row["delivery"], row["container"]))
        seconds.append(float(row["seconds_per_run_mean"]))
        if n_kind is None and row.get("n_per_kind"):
            n_kind = row["n_per_kind"]
        if repeats is None and row.get("repeats"):
            repeats = row["repeats"]

    order = sorted(range(len(seconds)), key=lambda i: seconds[i])
    labels = [labels[i] for i in order]
    seconds = [seconds[i] for i in order]
    best_idx = 0

    fig, ax = plt.subplots(figsize=FIGSIZE)
    y_pos = list(range(len(labels)))
    colors = [COLOR_BEST if i == best_idx else COLOR_BAR for i in range(len(labels))]
    bars = ax.barh(y_pos, seconds, color=colors, edgecolor="black", linewidth=0.5, alpha=0.9)
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(2.0)

    ax.set_yticks(y_pos, labels)
    ax.set_xlabel("Mean time per run, s (lower is better)")
    ax.set_ylabel("Implementation (delivery + map)")
    ax.invert_yaxis()

    title = args.title.strip()
    if not title:
        meta = []
        if n_kind:
            meta.append(f"n_per_kind={n_kind}")
        if repeats:
            meta.append(f"repeats={repeats}")
        title = "Task2 server: benchmark (mean seconds per run)"
        if meta:
            title += " (" + ", ".join(meta) + ")"
    ax.set_title(title, fontsize=13, weight="bold")

    xmax = max(seconds) * 1.18 if seconds else 1.0
    ax.set_xlim(0, xmax)

    best_sec = seconds[best_idx]
    for bar, sec in zip(bars, seconds):
        ax.text(
            bar.get_width() + xmax * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{sec:.6f}",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.axvline(best_sec, color=COLOR_BEST, linestyle=":", linewidth=1.4, alpha=0.85, zorder=0)
    ax.annotate(
        f"BEST: {labels[best_idx]}\n{best_sec:.6f} s",
        xy=(best_sec, best_idx),
        xytext=(best_sec + (xmax - best_sec) * 0.12, best_idx + 0.35),
        arrowprops={"arrowstyle": "->", "color": COLOR_BEST, "lw": 1.5},
        fontsize=11,
        color=COLOR_BEST,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": COLOR_BEST, "alpha": 0.95},
    )

    ax.grid(axis="x", **GRID_KW)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out.resolve()}")
    print(f"Best: {labels[best_idx]} ({seconds[best_idx]:.6f} s)")


if __name__ == "__main__":
    main()


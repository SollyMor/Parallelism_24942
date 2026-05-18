#!/usr/bin/env python3
"""
График ускорения, таблица T_n / S_n и график эффективности
из speedup_dgemv_stdthread_20000.csv и speedup_dgemv_stdthread_40000.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "threads": int(row["threads"]),
                    "T_serial": float(row["T_serial"]),
                    "T_parallel": float(row["T_parallel"]),
                    "speedup": float(row["speedup"]),
                    "efficiency": float(row["efficiency"]),
                }
            )
    rows.sort(key=lambda r: r["threads"])
    return rows


def union_threads(*row_lists: list[dict[str, float]]) -> list[int]:
    s: set[int] = set()
    for rows in row_lists:
        s.update(r["threads"] for r in rows)
    return sorted(s)


def best_efficiency_point(rows: list[dict[str, float]]) -> tuple[int, float]:
    best = max(rows, key=lambda r: r["efficiency"])
    return best["threads"], best["efficiency"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv20000",
        type=Path,
        default=Path(__file__).parent / "speedup_dgemv_stdthread_20000.csv",
    )
    parser.add_argument(
        "--csv40000",
        type=Path,
        default=Path(__file__).parent / "speedup_dgemv_stdthread_40000.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    r20 = read_csv(args.csv20000)
    r40 = read_csv(args.csv40000)

    threads_20 = [r["threads"] for r in r20]
    sp20 = [r["speedup"] for r in r20]
    tp20 = [r["T_parallel"] for r in r20]
    ef20 = [r["efficiency"] for r in r20]

    threads_40 = [r["threads"] for r in r40]
    sp40 = [r["speedup"] for r in r40]
    tp40 = [r["T_parallel"] for r in r40]
    ef40 = [r["efficiency"] for r in r40]

    max_t = max(max(threads_20), max(threads_40))
    ideal_x = list(range(1, max_t + 1))
    tick_threads = union_threads(r20, r40)
    ticks_note = (
        "По оси X только фактические n из CSV (шаг не 5, 10, 15…): "
        + ", ".join(str(t) for t in tick_threads)
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) График ускорения ---
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(threads_20, sp20, "o-", color="#1f77b4", linewidth=2, markersize=8, label="20000×20000")
    ax.plot(threads_40, sp40, "o-", color="#d62728", linewidth=2, markersize=8, label="40000×40000")
    ax.plot(ideal_x, ideal_x, "k--", linewidth=1.2, alpha=0.7, label="Линейное ускорение")

    ax.set_xlabel("Количество потоков", fontsize=12)
    ax.set_ylabel("Ускорение", fontsize=12)
    ax.set_title("Ускорение умножения матрицы на вектор", fontsize=13, fontweight="bold")
    ax.set_xticks(tick_threads)
    ax.set_xticklabels([str(t) for t in tick_threads])
    ax.set_xlim(0, max_t + 2)
    ax.set_ylim(0, max(max(sp20 + sp40), max_t) * 1.08)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left")

    # Подписи в конце линий (как на примере)
    if threads_20:
        ax.annotate(
            f"{sp20[-1]:.1f}x",
            xy=(threads_20[-1], sp20[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="yellow", alpha=0.85),
        )
    if threads_40:
        ax.annotate(
            f"{sp40[-1]:.1f}x",
            xy=(threads_40[-1], sp40[-1]),
            xytext=(8, -12),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="yellow", alpha=0.85),
        )

    fig.tight_layout(rect=[0, 0.11, 1, 0.98])
    fig.text(0.5, 0.055, ticks_note, ha="center", va="top", fontsize=9, transform=fig.transFigure)
    p_speed = out_dir / "dgemv_speedup_plot.png"
    fig.savefig(p_speed, dpi=200)
    plt.close(fig)
    print(f"Saved {p_speed}")

    # --- 2) Таблица ---
    # Колонки: только потоки > 1 как в примере (2,4,7,8,16,20,40) — берём пересечение
    common_threads = sorted(set(threads_20) & set(threads_40))
    if 1 in common_threads:
        common_threads = [t for t in common_threads if t != 1]

    def row_for(rows: list[dict], label: str) -> list[str]:
        by_t = {r["threads"]: r for r in rows}
        cells = [label]
        for t in common_threads:
            r = by_t[t]
            cells.append(f"{r['T_parallel']:.4f}")
            cells.append(f"{r['speedup']:.2f}")
        return cells

    col_labels = []
    for t in common_threads:
        col_labels.extend([f"T_{t}", f"S_{t}"])

    cell_text = [
        row_for(r20, "20000\n(~3 GiB)"),
        row_for(r40, "40000\n(~12 GiB)"),
    ]

    fig2, ax2 = plt.subplots(figsize=(14, 2.8))
    ax2.axis("off")
    table = ax2.table(
        cellText=cell_text,
        colLabels=["M = N"] + col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 2.0)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif col == 0:
            cell.set_facecolor("#E7E6E6")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#F2F2F2" if row % 2 else "white")

    fig2.tight_layout()
    p_table = out_dir / "dgemv_speedup_table.png"
    fig2.savefig(p_table, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {p_table}")

    # --- 3) График эффективности ---
    n20_best, e20_best = best_efficiency_point(r20)
    n40_best, e40_best = best_efficiency_point(r40)
    eff_caption = (
        f"Наибольшая эффективность: 20000×20000 — n={n20_best}, E={e20_best:.4f}; "
        f"40000×40000 — n={n40_best}, E={e40_best:.4f}"
    )

    fig3, ax3 = plt.subplots(figsize=(10, 6.4))
    ax3.plot(threads_20, ef20, "s-", color="#1f77b4", linewidth=2, markersize=7, label="20000×20000")
    ax3.plot(threads_40, ef40, "s-", color="#d62728", linewidth=2, markersize=7, label="40000×40000")
    ax3.axhline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.6, label="Идеальная эффективность (1.0)")

    ax3.set_xlabel("Количество потоков", fontsize=12)
    ax3.set_ylabel("Эффективность", fontsize=12)
    ax3.set_title("Эффективность умножения матрицы на вектор", fontsize=13, fontweight="bold")
    ax3.set_xticks(tick_threads)
    ax3.set_xticklabels([str(t) for t in tick_threads])
    ax3.set_xlim(0, max_t + 2)
    ymax = max(max(ef20 + ef40), 1.05) * 1.05
    ax3.set_ylim(0, ymax)
    ax3.grid(True, alpha=0.35)
    ax3.legend(loc="upper right")
    ax3.plot(
        n20_best,
        e20_best,
        "*",
        color="gold",
        markersize=22,
        markeredgecolor="darkgoldenrod",
        markeredgewidth=1.2,
        zorder=6,
        clip_on=False,
        label=None,
    )
    ax3.plot(
        n40_best,
        e40_best,
        "*",
        color="gold",
        markersize=22,
        markeredgecolor="darkgoldenrod",
        markeredgewidth=1.2,
        zorder=6,
        clip_on=False,
        label=None,
    )
    fig3.tight_layout(rect=[0, 0.16, 1, 0.98])
    fig3.text(0.5, 0.10, eff_caption, ha="center", va="top", fontsize=10, transform=fig3.transFigure)
    fig3.text(0.5, 0.045, ticks_note, ha="center", va="top", fontsize=9, transform=fig3.transFigure)
    p_eff = out_dir / "dgemv_efficiency_plot.png"
    fig3.savefig(p_eff, dpi=200)
    plt.close(fig3)
    print(f"Saved {p_eff}")


if __name__ == "__main__":
    main()

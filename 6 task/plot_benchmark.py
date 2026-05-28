#!/usr/bin/env python3
"""Plot benchmark tables for lab report.

Supports two CSV schemas:
1) benchmark_results.csv (variant/acc_mode/grid/time_sec/...)
2) report_tables.csv (mode/size/time_sec/error/iterations)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _plot_report_tables(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot graphs from report_tables.csv schema."""
    df = df.copy()
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    df = df.dropna(subset=["size", "time_sec"])
    df = df.sort_values("size")

    # CPU onecore vs multicore
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode, label in [
        ("CPU-onecore", "CPU-onecore"),
        ("CPU-multicore", "CPU-multicore"),
    ]:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        ax.plot(sub["size"].astype(int), sub["time_sec"], "o-", label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Размер сетки N")
    ax.set_ylabel("Время, с")
    ax.set_title("Сравнение CPU-onecore и CPU-multicore")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = out_dir / "time_cpu_onecore_multicore.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")

    # Bar chart in report style: CPU-onecore vs CPU-multicore
    one = df[df["mode"] == "CPU-onecore"][["size", "time_sec"]].set_index("size")
    multi = df[df["mode"] == "CPU-multicore"][["size", "time_sec"]].set_index("size")
    common_sizes = sorted(one.index.intersection(multi.index))
    if common_sizes:
        one_vals = one.loc[common_sizes, "time_sec"].values
        multi_vals = multi.loc[common_sizes, "time_sec"].values

        fig, ax = plt.subplots(figsize=(10, 6))
        x = list(range(len(common_sizes)))
        width = 0.28

        ax.bar([i - width / 2 for i in x], one_vals, width=width, label="Onecore")
        ax.bar(
            [i + width / 2 for i in x], multi_vals, width=width, label="Multicore"
        )

        ax.set_xticks(x, [f"{int(s)}*{int(s)}" for s in common_sizes])
        ax.set_yscale("log")
        ax.set_xlabel("Размер сетки")
        ax.set_ylabel("Время, с (log scale)")
        ax.set_title("Диаграмма сравнения времени работы CPU-one и CPU-multi")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        out = out_dir / "bar_cpu_onecore_multicore.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"Wrote {out}")

    # All available modes in one plot (CPU + GPU if present)
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode in sorted(df["mode"].unique()):
        sub = df[df["mode"] == mode]
        ax.plot(sub["size"].astype(int), sub["time_sec"], "o-", label=mode)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Размер сетки N")
    ax.set_ylabel("Время, с")
    ax.set_title("Сравнение режимов выполнения")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = out_dir / "time_all_modes.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


def _plot_benchmark_results(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot graphs from benchmark_results.csv schema."""
    df = df.copy()
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    df = df.dropna(subset=["grid", "time_sec"])
    df = df.sort_values("grid")

    for acc in df["acc_mode"].unique():
        sub = df[df["acc_mode"] == acc].copy()
        sub["label"] = sub["variant"].str.replace("baseline_", "").str.replace(
            "optimized_", "opt_"
        )
        pivot = sub.pivot_table(
            index="grid", columns="variant", values="time_sec", aggfunc="first"
        )
        base_cols = [c for c in pivot.columns if "baseline" in c]
        opt_cols = [c for c in pivot.columns if "optimized" in c]
        if not base_cols or not opt_cols:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        x = pivot.index.astype(int)
        ax.plot(x, pivot[base_cols[0]], "o-", label="baseline (до)")
        ax.plot(x, pivot[opt_cols[0]], "s-", label="optimized (после)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Размер сетки N")
        ax.set_ylabel("Время, с")
        ax.set_title(f"OpenACC -acc={acc}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = out_dir / f"time_{acc}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Wrote {out}")

    # Speedup optimized/baseline per mode
    fig, ax = plt.subplots(figsize=(8, 5))
    for acc in sorted(df["acc_mode"].unique()):
        sub = df[df["acc_mode"] == acc]
        b = sub[sub["variant"].str.contains("baseline")].set_index("grid")["time_sec"]
        o = sub[sub["variant"].str.contains("optimized")].set_index("grid")["time_sec"]
        common = b.index.intersection(o.index)
        speedup = (b.loc[common] / o.loc[common]).values
        ax.plot(common.astype(int), speedup, "o-", label=acc)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("Ускорение (baseline / optimized)")
    ax.set_title("Выигрыш от оптимизации")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "speedup_all.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "report_tables.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"Missing {args.csv}. Run scripts/benchmark.sh first.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    if {"mode", "size", "time_sec"}.issubset(df.columns):
        _plot_report_tables(df, args.out_dir)
    elif {"variant", "acc_mode", "grid", "time_sec"}.issubset(df.columns):
        _plot_benchmark_results(df, args.out_dir)
    else:
        raise SystemExit(
            "Unsupported CSV schema. Expected report_tables.csv "
            "(mode,size,time_sec,...) or benchmark_results.csv "
            "(variant,acc_mode,grid,time_sec,...)."
        )


if __name__ == "__main__":
    main()

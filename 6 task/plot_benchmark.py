#!/usr/bin/env python3
"""Plot benchmark_results.csv: baseline vs optimized, host/multicore/gpu."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_results.csv",
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
        out = args.out_dir / f"time_{acc}.png"
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
    out = args.out_dir / "speedup_all.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

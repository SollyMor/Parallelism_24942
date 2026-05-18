import csv
import math
import sys
from pathlib import Path


def read_rows(path: Path):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "threads": int(row["threads"]),
                    "t_parallel": float(row["T_parallel"]),
                    "speedup": float(row["speedup"]),
                    "efficiency": float(row["efficiency"]),
                }
            )
    return rows


def discover_input_csvs(base_dir: Path):
    csv_paths = sorted(base_dir.glob("speedup_dgemv_stdthread_*.csv"))
    return [p for p in csv_paths if p.name != "smoke_avg_500.csv"]


def score_threads(csv_paths, k=1.0):
    t_by_threads = {}
    for p in csv_paths:
        for row in read_rows(p):
            t_by_threads.setdefault(row["threads"], []).append(row["t_parallel"])

    scored = []
    for threads, values in sorted(t_by_threads.items()):
        mean_t = sum(values) / len(values)
        if len(values) > 1:
            var = sum((x - mean_t) ** 2 for x in values) / len(values)
            std_t = math.sqrt(var)
        else:
            std_t = 0.0
        j = mean_t + k * std_t
        scored.append((threads, mean_t, std_t, j))
    return scored


def choose_best_threads(scored_rows):
    return min(scored_rows, key=lambda x: x[3])


def plot_best_threads(scored, sorted_threads, out_metric):
    import matplotlib.pyplot as plt

    plot_threads = [row[0] for row in scored]
    plot_j = [row[3] for row in scored]
    best_threads, _best_mean_t, _best_std_t, best_j = choose_best_threads(scored)

    plt.figure(figsize=(11, 6))
    colors = ["#4e79a7"] * len(plot_threads)
    best_idx = plot_threads.index(best_threads)
    colors[best_idx] = "#d62728"

    bars = plt.bar(plot_threads, plot_j, color=colors, alpha=0.9, label="J(p)")
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(2.0)

    plt.axvline(best_threads, color="#d62728", linestyle="--", linewidth=1.6, alpha=0.8)
    plt.axhline(best_j, color="#d62728", linestyle=":", linewidth=1.4, alpha=0.8)
    plt.annotate(
        f"BEST = {best_threads} threads\nJ={best_j:.6f}",
        xy=(best_threads, best_j),
        xytext=(best_threads + 1, best_j * 1.06),
        arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.5},
        fontsize=11,
        color="#d62728",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#d62728", "alpha": 0.95},
    )
    plt.text(
        0.015,
        0.985,
        "Formula:\nJ(p) = mean(T_parallel) + k*std(T_parallel), k=1\nBest threads: p* = argmin_p J(p)",
        transform=plt.gca().transAxes,
        fontsize=10.5,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#fff8dc", "ec": "#444444", "alpha": 0.95},
    )

    plt.title("DGEMV std::thread: Best Thread Selection (Lower J is Better)", fontsize=13, weight="bold")
    plt.xlabel("Threads")
    plt.ylabel("J(p) = mean(T_parallel) + k*std(T_parallel)")
    xticks = sorted_threads if len(sorted_threads) <= 20 else list(range(min(sorted_threads), max(sorted_threads) + 1, 2))
    plt.xticks(xticks)
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_metric, dpi=150, bbox_inches="tight")
    plt.close()


def plot(csv_paths):
    import matplotlib.pyplot as plt

    series = []
    all_threads = set()
    for p in csv_paths:
        rows = read_rows(p)
        size = p.stem.split("_")[-1]
        threads = [r["threads"] for r in rows]
        speedup = [r["speedup"] for r in rows]
        efficiency = [r["efficiency"] for r in rows]
        series.append((size, threads, speedup, efficiency))
        all_threads.update(threads)

    if not series:
        raise RuntimeError("No CSV files provided.")

    sorted_threads = sorted(all_threads)
    out_speedup = csv_paths[0].parent / "speedup_task1_stdthread.png"
    out_eff = csv_paths[0].parent / "efficiency_task1_stdthread.png"

    plt.figure(figsize=(9, 6))
    for size, threads, speedup, _eff in series:
        plt.plot(threads, speedup, marker="o", linewidth=2, label=f"N={size}")
    plt.plot(sorted_threads, sorted_threads, linestyle="--", color="gray", label="Ideal x=y")
    plt.title("DGEMV std::thread: Speedup")
    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.grid(True, linestyle="--", alpha=0.4)
    xticks = sorted_threads if len(sorted_threads) <= 20 else list(range(min(sorted_threads), max(sorted_threads) + 1, 2))
    plt.xticks(xticks)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_speedup, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 6))
    for size, threads, _speedup, efficiency in series:
        plt.plot(threads, efficiency, marker="s", linewidth=2, label=f"N={size}")
    plt.title("DGEMV std::thread: Efficiency")
    plt.xlabel("Threads")
    plt.ylabel("Efficiency")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(xticks)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_eff, dpi=150, bbox_inches="tight")
    plt.close()

    return out_speedup, out_eff, sorted_threads


def main(argv):
    if len(argv) > 1:
        csv_paths = [Path(p) for p in argv[1:]]
    else:
        script_dir = Path(__file__).resolve().parent
        csv_paths = discover_input_csvs(script_dir)
        if not csv_paths:
            print(
                "Usage: python plot_speedup_task1.py [speedup_dgemv_stdthread_*.csv ...]\n"
                "No input CSVs provided and none were auto-discovered."
            )
            return 2

    missing = [p for p in csv_paths if not p.exists()]
    if missing:
        print("Missing CSV file(s):")
        for p in missing:
            print(f"- {p}")
        return 2

    out_speedup, out_eff, sorted_threads = plot(csv_paths)
    k = 1.0

    scored = score_threads(csv_paths, k=k)
    best_threads, best_mean_t, best_std_t, best_j = choose_best_threads(scored)
    out_metric = csv_paths[0].parent / "metric_task1_stdthread.png"
    plot_best_threads(scored, sorted_threads, out_metric)

    print(f"Wrote {out_speedup}")
    print(f"Wrote {out_eff}")
    print(f"Wrote {out_metric}")
    print("\nMetric: J(p) = mean(T_parallel) + k*std(T_parallel), k=1.0")
    print("threads,mean_t,std_t,J")
    for threads, mean_t, std_t, j in scored:
        print(f"{threads},{mean_t:.9f},{std_t:.9f},{j:.9f}")
    print(f"\nBest threads p* = {best_threads} (J={best_j:.9f}, mean={best_mean_t:.9f}, std={best_std_t:.9f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


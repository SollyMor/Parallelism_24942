import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot matrix-vector speedup.")
    parser.add_argument("--csv", default="../matrix_threads_summary.csv")
    parser.add_argument("--out", default="matrix_threads_speedup.png")
    args = parser.parse_args()

    with Path(args.csv).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    by_size = defaultdict(list)
    for row in rows:
        by_size[int(row["size"])].append(row)

    plt.figure(figsize=(9, 5))
    for size, items in sorted(by_size.items()):
        items.sort(key=lambda item: int(item["threads"]))
        threads = [int(item["threads"]) for item in items]
        speedup = [float(item["speedup"]) for item in items]
        plt.plot(threads, speedup, marker="o", label=f"{size}x{size}")

    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.title("Matrix-vector multiplication speedup")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

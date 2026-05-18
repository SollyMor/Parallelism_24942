import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser(description="Plot matrix-vector speedup.")
    parser.add_argument("--csv", default="../matrix_vector_summary.csv")
    parser.add_argument("--out", default="matrix_vector_speedup.png")
    parser.add_argument("--eff-out", default="matrix_vector_efficiency.png")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    rows = read_rows(csv_path)
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
    plt.close()

    plt.figure(figsize=(9, 5))
    for size, items in sorted(by_size.items()):
        items.sort(key=lambda item: int(item["threads"]))
        threads = [int(item["threads"]) for item in items]
        efficiency = [float(item["efficiency"]) for item in items]
        plt.plot(threads, efficiency, marker="o", label=f"{size}x{size}")

    plt.xlabel("Threads")
    plt.ylabel("Efficiency")
    plt.title("Matrix-vector multiplication efficiency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.eff_out, dpi=200)
    plt.close()
    print(f"Saved {args.out}")
    print(f"Saved {args.eff_out}")


if __name__ == "__main__":
    main()

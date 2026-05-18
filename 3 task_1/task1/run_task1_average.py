import argparse
import csv
import subprocess
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run task1 benchmark multiple times and average CSV results."
    )
    parser.add_argument(
        "--exe",
        default="build/task1/task1_stdthread",
        help="Path to task1 executable (default: build/task1/task1_stdthread)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="How many times to run the benchmark (default: 3)",
    )
    parser.add_argument(
        "--sizes",
        default="20000,40000",
        help="Sizes passed to task1 (default: 20000,40000)",
    )
    parser.add_argument(
        "--threads",
        default="1,2,4,7,8,16,20,40",
        help="Thread list passed to task1",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Inner repeats in task1 executable (default: 5)",
    )
    parser.add_argument(
        "--drop-max",
        type=int,
        default=1,
        help="Drop max samples in task1 executable (default: 1)",
    )
    parser.add_argument(
        "--output-prefix",
        default="speedup_dgemv_stdthread_avg",
        help="Output prefix for averaged CSV files",
    )
    parser.add_argument(
        "--workdir",
        default=".",
        help="Working directory where intermediate/final CSV are written",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep per-run CSV files",
    )
    return parser.parse_args()


def read_csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_avg_csv(path: Path, rows):
    fieldnames = ["threads", "T_serial", "T_parallel", "speedup", "efficiency"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def average_grouped_rows(grouped_values):
    out_rows = []
    for threads in sorted(grouped_values.keys()):
        cols = grouped_values[threads]
        count = len(cols["T_serial"])
        out_rows.append(
            {
                "threads": threads,
                "T_serial": f"{sum(cols['T_serial']) / count:.9f}",
                "T_parallel": f"{sum(cols['T_parallel']) / count:.9f}",
                "speedup": f"{sum(cols['speedup']) / count:.9f}",
                "efficiency": f"{sum(cols['efficiency']) / count:.9f}",
            }
        )
    return out_rows


def main():
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    workdir = Path(args.workdir).resolve()
    exe_path = Path(args.exe)
    if not exe_path.is_absolute():
        exe_path = (workdir / exe_path).resolve()

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    per_size_grouped = {size: defaultdict(lambda: defaultdict(list)) for size in sizes}
    intermediate_files = []

    for run_idx in range(1, args.runs + 1):
        run_prefix = f"tmp_run_{run_idx}_speedup_dgemv_stdthread"
        cmd = [
            str(exe_path),
            "--sizes",
            args.sizes,
            "--threads",
            args.threads,
            "--repeats",
            str(args.repeats),
            "--drop-max",
            str(args.drop_max),
            "--out-prefix",
            run_prefix,
        ]
        print(f"[run {run_idx}/{args.runs}] {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=workdir)

        for size in sizes:
            csv_path = workdir / f"{run_prefix}_{size}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing CSV after run {run_idx}: {csv_path}")
            intermediate_files.append(csv_path)
            for row in read_csv_rows(csv_path):
                threads = int(row["threads"])
                for key in ("T_serial", "T_parallel", "speedup", "efficiency"):
                    per_size_grouped[size][threads][key].append(float(row[key]))

    for size in sizes:
        averaged_rows = average_grouped_rows(per_size_grouped[size])
        out_path = workdir / f"{args.output_prefix}_{size}.csv"
        write_avg_csv(out_path, averaged_rows)
        print(f"Wrote averaged CSV: {out_path}")

    if not args.keep_intermediate:
        for path in intermediate_files:
            path.unlink(missing_ok=True)
        print("Intermediate per-run CSV files removed.")


if __name__ == "__main__":
    main()


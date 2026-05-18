import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def resolve_csv_path(csv_arg):
    given = Path(csv_arg)
    if given.exists():
        return given

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / csv_arg,
        script_dir.parent / "integration_summary.csv",
        script_dir.parent / "build" / "integration_summary.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    found = list(script_dir.parent.rglob("integration_summary.csv"))
    if found:
        return found[0]

    raise FileNotFoundError(
        "integration_summary.csv not found. Run integral first to generate it, "
        "or pass explicit path with --csv."
    )


def resolve_policy_csv_path(csv_arg):
    given = Path(csv_arg)
    if given.exists():
        return given

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / csv_arg,
        script_dir.parent / "integration_results.csv",
        script_dir.parent / "build" / "integration_results.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    found = list(script_dir.parent.rglob("integration_results.csv"))
    if found:
        return found[0]

    raise FileNotFoundError(
        "integration_results.csv not found. Generate binding-policy results first, "
        "or pass explicit path with --policy-csv."
    )


def draw_policy_table(ax, serial_row, policy_rows):
    ax.axis("off")
    rows_for_table = [serial_row] + policy_rows
    cell_text = []
    for row in rows_for_table:
        threads = "1 (serial)" if row["Policy"] == "serial" else row["Threads"]
        cell_text.append(
            [
                threads,
                f"{float(row['Time_Seconds']):.6f}",
                f"{float(row['Speedup']):.4f}",
            ]
        )

    table = ax.table(
        cellText=cell_text,
        colLabels=["Threads", "Time (sec)", "Speedup"],
        cellLoc="center",
        loc="center",
        colWidths=[0.35, 0.3, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for col in range(3):
        header = table[(0, col)]
        header.set_facecolor("#4CAF50")
        header.set_text_props(color="white", weight="bold")

    for col in range(3):
        serial_cell = table[(1, col)]
        serial_cell.set_facecolor("#c8d7e3")
        serial_cell.set_text_props(weight="bold")

    for row_idx in range(2, len(rows_for_table) + 1):
        shade = "#f2f2f2" if row_idx % 2 == 0 else "#e7e7e7"
        for col in range(3):
            table[(row_idx, col)].set_facecolor(shade)

def generate_policy_report(policy_csv_path, out_path):
    rows = read_rows(policy_csv_path)
    if not rows:
        raise ValueError(f"Policy CSV is empty: {policy_csv_path}")

    serial_rows = [row for row in rows if row["Policy"] == "serial"]
    if not serial_rows:
        raise ValueError("No serial row found in policy CSV.")
    serial_row = serial_rows[0]

    policy_order = ["none", "close", "spread"]
    colors = {"none": "#1f77b4", "close": "#d62728", "spread": "#2ca02c"}
    markers = {"none": "o", "close": "^", "spread": "s"}

    fig = plt.figure(figsize=(11.5, 12.5))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.7, 1.2, 1.2, 1.2], hspace=0.55)

    ax = fig.add_subplot(gs[0])
    all_points = []
    for policy in policy_order:
        items = [row for row in rows if row["Policy"] == policy]
        if not items:
            continue
        items.sort(key=lambda row: int(row["Threads"]))
        threads = [int(row["Threads"]) for row in items]
        speedup = [float(row["Speedup"]) for row in items]
        all_points.extend(zip(threads, speedup, [policy] * len(threads)))
        ax.plot(
            threads,
            speedup,
            marker=markers[policy],
            linewidth=2.0,
            color=colors[policy],
            label=policy,
        )

    max_thread = max(int(row["Threads"]) for row in rows if row["Policy"] != "serial")
    ax.plot([1, max_thread], [1, max_thread], "--", color="gray", linewidth=1.3, label="Ideal speedup")
    ax.set_title("Integration speedup\nBinding policy comparison", fontweight="bold")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    if all_points:
        best_thread, best_speedup, best_policy = max(all_points, key=lambda item: item[1])
        ax.annotate(
            f"{best_speedup:.1f}x ({best_policy}, T={best_thread})",
            xy=(best_thread, best_speedup),
            xytext=(20, -20),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "yellow", "alpha": 0.5},
            fontsize=9,
        )

    table_specs = [
        ("none", 1),
        ("close", 2),
        ("spread", 3),
    ]
    for policy, idx in table_specs:
        ax_table = fig.add_subplot(gs[idx])
        policy_rows = [row for row in rows if row["Policy"] == policy]
        policy_rows.sort(key=lambda row: int(row["Threads"]))
        draw_policy_table(ax_table, serial_row, policy_rows)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def generate_efficiency_from_policy_csv(policy_csv_path, eff_out_path, table_out_path):
    rows = read_rows(policy_csv_path)
    policy_rows = [row for row in rows if row["Policy"] != "serial"]
    if not policy_rows:
        raise ValueError("No policy rows found for efficiency plot.")

    policy_order = ["none", "close", "spread"]
    colors = {"none": "#1f77b4", "close": "#d62728", "spread": "#2ca02c"}
    markers = {"none": "o", "close": "^", "spread": "s"}

    # Build grouped series and compute efficiency = speedup / threads.
    by_policy = {}
    best_point = None
    for policy in policy_order:
        items = [row for row in policy_rows if row["Policy"] == policy]
        if not items:
            continue
        items.sort(key=lambda row: int(row["Threads"]))
        threads = [int(row["Threads"]) for row in items]
        speedup = [float(row["Speedup"]) for row in items]
        times = [float(row["Time_Seconds"]) for row in items]
        efficiency = [s / t for s, t in zip(speedup, threads)]
        by_policy[policy] = {
            "threads": threads,
            "speedup": speedup,
            "times": times,
            "efficiency": efficiency,
        }
        local_best_idx = max(range(len(threads)), key=lambda i: efficiency[i])
        candidate = (efficiency[local_best_idx], policy, threads[local_best_idx])
        if best_point is None or candidate[0] > best_point[0]:
            best_point = candidate

    if not by_policy:
        raise ValueError("No known policies (none/close/spread) found in policy CSV.")

    plt.figure(figsize=(10, 5))
    for policy in policy_order:
        if policy not in by_policy:
            continue
        series = by_policy[policy]
        label = f"{policy} (efficiency)"
        plt.plot(
            series["threads"],
            series["efficiency"],
            marker=markers[policy],
            linewidth=2.0,
            color=colors[policy],
            label=label,
        )
        for thread, value in zip(series["threads"], series["efficiency"]):
            plt.annotate(
                f"{value:.3f}",
                xy=(thread, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    if best_point is not None:
        best_eff, best_policy, best_thread = best_point
        plt.annotate(
            f"best: {best_eff:.3f} ({best_policy}, T={best_thread})",
            xy=(best_thread, best_eff),
            xytext=(12, 18),
            textcoords="offset points",
            fontsize=9,
            color="#d62728",
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": "#d62728"},
        )

    all_threads = sorted({t for series in by_policy.values() for t in series["threads"]})
    plt.xlabel("Threads")
    plt.ylabel("Efficiency")
    plt.title("Integration efficiency by thread count (binding policies)")
    plt.xticks(all_threads, [str(t) for t in all_threads])
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(eff_out_path, dpi=200)
    plt.close()

    # Comparative table for the same CSV (rows by policy and metric).
    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.axis("off")

    def values_for(policy, field):
        series = by_policy.get(policy)
        if series is None:
            return ["-" for _ in all_threads]
        mapping = {t: v for t, v in zip(series["threads"], series[field])}
        return [mapping.get(t, "-") for t in all_threads]

    table_rows = [
        [f"{v:.4f}" if isinstance(v, float) else v for v in values_for("none", "efficiency")],
        [f"{v:.4f}" if isinstance(v, float) else v for v in values_for("close", "efficiency")],
        [f"{v:.4f}" if isinstance(v, float) else v for v in values_for("spread", "efficiency")],
        [f"{v:.4f}" if isinstance(v, float) else v for v in values_for("none", "speedup")],
        [f"{v:.4f}" if isinstance(v, float) else v for v in values_for("close", "speedup")],
        [f"{v:.4f}" if isinstance(v, float) else v for v in values_for("spread", "speedup")],
    ]
    row_labels = [
        "none: efficiency",
        "close: efficiency",
        "spread: efficiency",
        "none: speedup",
        "close: speedup",
        "spread: speedup",
    ]

    table = ax.table(
        cellText=table_rows,
        rowLabels=row_labels,
        colLabels=[str(t) for t in all_threads],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.45)

    # Header colors.
    for col in range(len(all_threads)):
        table[(0, col)].set_facecolor("#e6eef8")
    for row in range(1, len(table_rows) + 1):
        table[(row, -1)].set_facecolor("#f0f0f0")
    for row in range(1, len(table_rows) + 1):
        shade = "#f8f8f8" if row % 2 else "#eeeeee"
        for col in range(len(all_threads)):
            table[(row, col)].set_facecolor(shade)

    ax.set_title("Comparative table from integration_results.csv")
    plt.tight_layout()
    plt.savefig(table_out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot integration speedup.")
    parser.add_argument("--csv", default="../integration_summary.csv")
    parser.add_argument("--out", default="integration_speedup.png")
    parser.add_argument("--eff-out", default="integration_efficiency.png")
    parser.add_argument("--table-out", default="integration_efficiency_table.png")
    parser.add_argument("--policy-csv", default="../integration_results.csv")
    parser.add_argument("--policy-out", default="integration_policy_report.png")
    args = parser.parse_args()

    csv_path = resolve_csv_path(args.csv)
    rows = [
        row for row in read_rows(csv_path)
        if row["method"] == "omp_atomic_local"
    ]
    if not rows:
        raise ValueError(
            f"No 'omp_atomic_local' rows found in {csv_path}. "
            "Check CSV format and content."
        )
    rows.sort(key=lambda row: int(row["threads"]))

    threads = [int(row["threads"]) for row in rows]
    speedup = [float(row["speedup"]) for row in rows]
    times = [float(row["avg_time_sec"]) for row in rows]
    efficiency = [float(row["efficiency"]) for row in rows]
    best = min(rows, key=lambda row: float(row["avg_time_sec"]))
    best_efficiency = max(
        (row for row in rows if int(row["threads"]) > 1),
        key=lambda row: float(row["efficiency"]),
    )
    best_eff_threads = int(best_efficiency["threads"])

    plt.figure(figsize=(9, 5))
    plt.plot(threads, speedup, marker="o", linewidth=2.0, label="Speedup (omp_atomic_local)")
    for thread, value in zip(threads, speedup):
        plt.annotate(
            f"T={thread}",
            xy=(thread, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.title("Integration exp(-x*x) speedup")
    plt.grid(True)
    plt.xticks(threads, [str(t) for t in threads])
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    colors = ["#1f77b4"] * len(threads)
    for idx, thread in enumerate(threads):
        if thread == best_eff_threads:
            colors[idx] = "#d62728"

    plt.figure(figsize=(10, 5))
    bars = plt.bar(threads, efficiency, color=colors, edgecolor="black", linewidth=0.7)
    plt.xlabel("Threads")
    plt.ylabel("Efficiency")
    plt.title("Integration efficiency by thread count")
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(threads, [str(t) for t in threads])
    plt.ylim(0, max(efficiency) * 1.2)
    for thread, value, bar in zip(threads, efficiency, bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + max(efficiency) * 0.02,
            f"E={value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        if thread == best_eff_threads:
            plt.annotate(
                "best efficiency",
                xy=(bar.get_x() + bar.get_width() / 2.0, value),
                xytext=(0, 34),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="#d62728",
                fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "#d62728"},
            )
    legend_items = [
        Patch(facecolor="#1f77b4", edgecolor="black", label="Regular thread count"),
        Patch(facecolor="#d62728", edgecolor="black", label=f"Best efficiency (T={best_eff_threads})"),
    ]
    plt.legend(handles=legend_items, loc="upper right")
    plt.tight_layout()
    plt.savefig(args.eff_out, dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")
    cell_text = [
        [f"{thread}" for thread in threads],
        [f"{eff:.4f}" for eff in efficiency],
        [f"{spd:.4f}" for spd in speedup],
        [f"{tm:.6f}" for tm in times],
    ]
    row_labels = ["Threads", "Efficiency", "Speedup", "Avg time, sec"]
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    best_col = threads.index(best_eff_threads)
    for col in range(len(threads)):
        color_strength = 0.15 + 0.45 * (efficiency[col] / (max(efficiency) + 1e-12))
        table[(1, col)].set_facecolor((0.75 - color_strength * 0.3, 1.0, 0.75 - color_strength * 0.3))
    for row in range(len(cell_text)):
        table[(row, best_col)].set_facecolor("#ffeb99")
    for col in range(len(threads)):
        table[(0, col)].set_facecolor("#e6eef8")
    for row in range(len(cell_text)):
        table[(row, -1)].set_facecolor("#f0f0f0")

    ax.set_title("Comparative table by threads (best efficiency highlighted)")
    plt.tight_layout()
    plt.savefig(args.table_out, dpi=200)
    plt.close()

    print(f"CSV source: {csv_path}")
    print(f"Saved {args.out}")
    print(f"Saved {args.eff_out}")
    print(f"Saved {args.table_out}")
    print(
        "Best time / speed: "
        f"{best['threads']} threads, {float(best['avg_time_sec']):.6f} sec, "
        f"speedup {float(best['speedup']):.4f}"
    )
    print(
        "Best efficiency: "
        f"{best_efficiency['threads']} threads, "
        f"efficiency {float(best_efficiency['efficiency']):.4f}, "
        f"speedup {float(best_efficiency['speedup']):.4f}, "
        f"time {float(best_efficiency['avg_time_sec']):.6f} sec"
    )
    print(f"Measured times: {list(zip(threads, times))}")
    print(f"Measured efficiency: {list(zip(threads, efficiency))}")

    try:
        policy_csv_path = resolve_policy_csv_path(args.policy_csv)
        generate_policy_report(policy_csv_path, args.policy_out)
        generate_efficiency_from_policy_csv(policy_csv_path, args.eff_out, args.table_out)
        print(f"Policy CSV source: {policy_csv_path}")
        print(f"Saved {args.policy_out}")
        print(f"Rebuilt {args.eff_out} from policy CSV")
        print(f"Rebuilt {args.table_out} from policy CSV")
    except FileNotFoundError as err:
        print(f"Skipped policy report: {err}")


if __name__ == "__main__":
    main()

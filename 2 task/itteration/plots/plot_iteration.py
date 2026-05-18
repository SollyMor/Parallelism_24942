import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_csv_rows_raw(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.reader(file))


POLICY_ORDER = ("static", "dynamic", "guided")
POLICY_SET = frozenset(POLICY_ORDER)


def chunk_label_sort_key(chunk: str) -> tuple:
    chunk = chunk.strip()
    if chunk.isdigit():
        return (0, int(chunk))
    return (1, chunk.lower())


def parse_legacy_schedule_speedup_rows(path: Path) -> list[dict]:
    """
    Rows: policy, chunk, time_serial, time_parallel, speedup (5 columns),
    or DictReader with config_description like 'static,4'.
    """
    parsed: list[dict] = []
    raw = read_csv_rows_raw(path)
    for row in raw:
        if not row:
            continue
        row = [c.strip() for c in row]
        if not row[0]:
            continue
        head = row[0].lower()
        if head in ("config_description", "schedule"):
            continue
        if len(row) >= 5 and head in POLICY_SET:
            policy, chunk, ts, tp, sp = row[0], row[1], row[2], row[3], row[4]
            try:
                parsed.append(
                    {
                        "policy": policy,
                        "chunk": chunk,
                        "config_description": f"{policy},{chunk}",
                        "time_serial": ts,
                        "time_parallel": tp,
                        "speedup": float(sp),
                    }
                )
            except ValueError:
                continue

    if parsed:
        return parsed

    try:
        for row in read_rows(path):
            desc = (row.get("config_description") or "").strip()
            if "," not in desc:
                continue
            policy, chunk = [p.strip() for p in desc.split(",", 1)]
            if policy.lower() not in POLICY_SET:
                continue
            parsed.append(
                {
                    "policy": policy,
                    "chunk": chunk,
                    "config_description": desc,
                    "time_serial": row.get("time_serial", ""),
                    "time_parallel": row.get("time_parallel", ""),
                    "speedup": float(row["speedup"]),
                }
            )
    except (KeyError, ValueError, OSError):
        pass

    return parsed


def resolve_input_path(user_path, fallback_names):
    given = Path(user_path)
    if given.exists():
        return given

    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir / user_path]
    for name in fallback_names:
        candidates.append(script_dir.parent / name)
        candidates.append(script_dir.parent / "build" / name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Input CSV not found: {user_path}")


def plot_scaling(summary_path, out_path, eff_out_path):
    rows = read_rows(summary_path)
    by_variant = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)

    plt.figure(figsize=(9, 5))
    for variant, items in sorted(by_variant.items()):
        items.sort(key=lambda item: int(item["threads"]))
        threads = [int(item["threads"]) for item in items]
        speedup = [float(item["speedup"]) for item in items]
        plt.plot(threads, speedup, marker="o", label=variant)

    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.title("Simple iteration solver speedup")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

    plt.figure(figsize=(9, 5))
    for variant, items in sorted(by_variant.items()):
        items.sort(key=lambda item: int(item["threads"]))
        threads = [int(item["threads"]) for item in items]
        efficiency = [float(item["efficiency"]) for item in items]
        plt.plot(threads, efficiency, marker="o", label=variant)

    plt.xlabel("Threads")
    plt.ylabel("Efficiency")
    plt.title("Simple iteration solver efficiency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eff_out_path, dpi=200)
    plt.close()
    print(f"Saved {eff_out_path}")


def plot_schedule(summary_path, out_path):
    rows = read_rows(summary_path)
    rows.sort(key=lambda row: float(row["avg_time_sec"]))

    labels = [row["schedule"] for row in rows]
    times = [float(row["avg_time_sec"]) for row in rows]
    colors = ["#d62728"] + ["#1f77b4"] * (len(labels) - 1)

    plt.figure(figsize=(10, 5))
    plt.bar(labels, times, color=colors)
    plt.xlabel("Schedule")
    plt.ylabel("Average time, sec")
    plt.title("OpenMP schedule comparison")
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
    print(f"Best schedule: {rows[0]['schedule']}, {float(rows[0]['avg_time_sec']):.6f} sec")


def plot_from_summary_table(summary_path, out_plot_path, out_table_path):
    rows = read_rows(summary_path)
    rows.sort(key=lambda row: int(row["num_threads"]))
    threads = [int(row["num_threads"]) for row in rows]
    speedup_1 = [float(row["speedup_1"]) for row in rows]
    speedup_2 = [float(row["speedup_2"]) for row in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(threads, speedup_1, marker="o", linewidth=2, color="#1f77b4", label="variant 1")
    plt.plot(threads, speedup_2, marker="^", linewidth=2, color="#d62728", label="variant 2")
    plt.plot([min(threads), max(threads)], [min(threads), max(threads)], "--", color="gray", label="ideal speedup")
    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.title("Simple iteration speedup (from summary.csv)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot_path, dpi=200)
    plt.close()
    print(f"Saved {out_plot_path}")

    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    ax.axis("off")
    cell_text = [
        [f"{int(row['num_threads'])}" for row in rows],
        [f"{float(row['time_parallel_1']):.6f}" for row in rows],
        [f"{float(row['speedup_1']):.4f}" for row in rows],
        [f"{float(row['time_parallel_2']):.6f}" for row in rows],
        [f"{float(row['speedup_2']):.4f}" for row in rows],
    ]
    row_labels = [
        "Threads",
        "Variant 1 time, sec",
        "Variant 1 speedup",
        "Variant 2 time, sec",
        "Variant 2 speedup",
    ]
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.12, 1.5)
    for col in range(len(rows)):
        table[(0, col)].set_facecolor("#e6eef8")
    for row_idx in range(1, len(cell_text) + 1):
        if (row_idx, -1) in table.get_celld():
            table[(row_idx, -1)].set_facecolor("#f0f0f0")
        shade = "#f8f8f8" if row_idx % 2 else "#eeeeee"
        for col in range(len(rows)):
            if (row_idx, col) in table.get_celld():
                table[(row_idx, col)].set_facecolor(shade)
    ax.set_title("Comparative table from summary.csv")
    plt.tight_layout()
    plt.savefig(out_table_path, dpi=200)
    plt.close()
    print(f"Saved {out_table_path}")


def plot_from_schedule_table(schedule_path, out_plot_path, out_table_path):
    rows = parse_legacy_schedule_speedup_rows(schedule_path)
    if not rows:
        print(f"Skipped schedule policy plots: no parseable rows in {schedule_path}")
        return

    rows.sort(key=lambda row: float(row["speedup"]), reverse=True)

    all_chunks = sorted({row["chunk"] for row in rows}, key=chunk_label_sort_key)
    by_policy: dict[str, dict[str, float]] = {p: {} for p in POLICY_ORDER}
    for row in rows:
        by_policy[row["policy"]][row["chunk"]] = float(row["speedup"])

    x_idx = list(range(len(all_chunks)))
    style = {
        "static": {"color": "#1f77b4", "marker": "o"},
        "dynamic": {"color": "#ff7f0e", "marker": "s"},
        "guided": {"color": "#2ca02c", "marker": "^"},
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    for policy in POLICY_ORDER:
        ys = [by_policy[policy].get(ch, float("nan")) for ch in all_chunks]
        st = style[policy]
        ax.plot(
            x_idx,
            ys,
            color=st["color"],
            marker=st["marker"],
            linewidth=2.0,
            markersize=9,
            markeredgecolor="black",
            markeredgewidth=0.6,
            label=policy,
        )

    best_row = max(rows, key=lambda row: float(row["speedup"]))
    best_x = all_chunks.index(best_row["chunk"])
    best_y = float(best_row["speedup"])
    ax.plot(
        [best_x],
        [best_y],
        linestyle="None",
        marker="*",
        markersize=22,
        color="#d62728",
        markeredgecolor="black",
        markeredgewidth=0.6,
        zorder=6,
        label=f"Best: {best_row['config_description']}",
    )

    ax.set_xticks(x_idx)
    ax.set_xticklabels(all_chunks)
    ax.set_xlabel("Chunk size")
    ax.set_ylabel("Speedup")
    ax.set_title("Schedule policy comparison by chunk (from summary_sc.scv)")

    all_y = [float(r["speedup"]) for r in rows]
    s_min, s_max = min(all_y), max(all_y)
    margin = max(0.02, (s_max - s_min) * 0.25)
    ax.set_ylim(s_min - margin, s_max + margin)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.grid(True, axis="y", alpha=0.35, which="major")
    ax.grid(True, axis="y", alpha=0.2, which="minor", linestyle=":")
    ax.grid(True, axis="x", alpha=0.15)

    ax.annotate(
        f"{best_row['config_description']}\n{best_y:.4f}",
        xy=(best_x, best_y),
        xytext=(0, 14),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#d62728",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#d62728", alpha=0.95),
    )
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_plot_path}")

    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.axis("off")
    cell_text = [
        [row["config_description"] for row in rows],
        [f"{float(row['time_parallel']):.6f}" for row in rows],
        [f"{float(row['speedup']):.4f}" for row in rows],
    ]
    row_labels = ["Config", "Parallel time, sec", "Speedup"]
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.12, 1.5)
    for col in range(len(rows)):
        table[(0, col)].set_facecolor("#e6eef8")
    for row_idx in range(1, len(cell_text) + 1):
        if (row_idx, -1) in table.get_celld():
            table[(row_idx, -1)].set_facecolor("#f0f0f0")
        shade = "#f8f8f8" if row_idx % 2 else "#eeeeee"
        for col in range(len(rows)):
            if (row_idx, col) in table.get_celld():
                table[(row_idx, col)].set_facecolor(shade)
    ax.set_title("Comparative table from summary_sc.scv")
    plt.tight_layout()
    plt.savefig(out_table_path, dpi=200)
    plt.close()
    print(f"Saved {out_table_path}")


def parse_policy_chunk_times(summary_sc_path):
    rows = read_csv_rows_raw(summary_sc_path)
    if len(rows) < 2:
        return {}

    by_policy = defaultdict(list)
    for row in rows[1:]:
        if len(row) < 4:
            continue
        policy = row[0].strip()
        chunk = row[1].strip()
        # Works for both 4/5-column variants; parallel time is before speedup when present.
        time_parallel = float(row[-2])
        by_policy[policy].append((chunk, time_parallel))

    # Keep chunks in natural order (numeric first, then other labels like N/T).
    def chunk_key(item):
        chunk = item[0]
        return (0, int(chunk)) if chunk.isdigit() else (1, chunk)

    for policy in by_policy:
        by_policy[policy] = sorted(by_policy[policy], key=chunk_key)
    return by_policy


def plot_chunk_bars(summary_sc_path, out_prefix):
    by_policy = parse_policy_chunk_times(summary_sc_path)
    if not by_policy:
        print(f"Skipped chunk charts: no valid rows in {summary_sc_path}")
        return

    for policy in ("static", "dynamic", "guided"):
        series = by_policy.get(policy)
        if not series:
            continue
        chunks = [chunk for chunk, _ in series]
        times = [time for _, time in series]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(chunks, times, color="#1f77b4", edgecolor="black", linewidth=0.6)
        ax.set_xlabel("Chunk size")
        ax.set_ylabel("Average time, sec")
        ax.set_title(f"Bar chart: time vs chunk — {policy}")

        t_min, t_max = min(times), max(times)
        pad = max(0.05, (t_max - t_min) * 0.15)
        ax.set_ylim(max(0.0, t_min - pad), t_max + pad)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.02))
        ax.grid(True, axis="y", alpha=0.35, which="major")
        ax.grid(True, axis="y", alpha=0.2, which="minor", linestyle=":")

        labels = [f"{t:.4f}" for t in times]
        ax.bar_label(
            bars,
            labels=labels,
            label_type="center",
            fontsize=10,
            color="white",
            fontweight="bold",
        )

        fig.tight_layout()
        out_path = f"{out_prefix}_{policy}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot iteration solver results.")
    parser.add_argument("--summary", default="../iteration_summary.csv")
    parser.add_argument("--schedule", default="../iteration_schedule_summary.csv")
    parser.add_argument("--speedup-out", default="iteration_speedup.png")
    parser.add_argument("--eff-out", default="iteration_efficiency.png")
    parser.add_argument("--schedule-out", default="iteration_schedule.png")
    parser.add_argument("--legacy-summary", default="../build/summary.csv")
    parser.add_argument("--legacy-schedule", default="../build/summary_sc.scv")
    parser.add_argument("--legacy-speedup-out", default="iteration_legacy_speedup.png")
    parser.add_argument("--legacy-table-out", default="iteration_legacy_table.png")
    parser.add_argument("--legacy-schedule-out", default="iteration_legacy_schedule.png")
    parser.add_argument("--legacy-schedule-table-out", default="iteration_legacy_schedule_table.png")
    parser.add_argument("--chunk-bars-out-prefix", default="iteration_chunk_time")
    args = parser.parse_args()

    summary_path = resolve_input_path(args.summary, ["iteration_summary.csv"])
    schedule_path = resolve_input_path(args.schedule, ["iteration_schedule_summary.csv"])
    plot_scaling(summary_path, args.speedup_out, args.eff_out)
    plot_schedule(schedule_path, args.schedule_out)

    try:
        legacy_summary_path = resolve_input_path(args.legacy_summary, ["summary.csv"])
        plot_from_summary_table(
            legacy_summary_path,
            args.legacy_speedup_out,
            args.legacy_table_out,
        )
    except FileNotFoundError as err:
        print(f"Skipped legacy summary table plots: {err}")

    try:
        legacy_schedule_path = resolve_input_path(args.legacy_schedule, ["summary_sc.scv", "summary_sc.csv"])
        plot_from_schedule_table(
            legacy_schedule_path,
            args.legacy_schedule_out,
            args.legacy_schedule_table_out,
        )
        plot_chunk_bars(legacy_schedule_path, args.chunk_bars_out_prefix)
    except FileNotFoundError as err:
        print(f"Skipped legacy schedule table plots: {err}")


if __name__ == "__main__":
    main()

# Ensure only ASCII characters are used in this script
"""
Processes NASA NPB benchmark log files (*.log) from two directories,
calculates statistics (mean +/- stdev) for each benchmark within each directory,
prints a comparison summary including percentage differences (Dir2 vs Dir1)
to stdout and ./summary.txt, and generates comparison plots saved as PDF files.
Uses only ASCII characters.
"""

import argparse
import collections

# import csv # No longer needed
import glob
import math  # Needed for checking isnan/isinf
import os
import re
import statistics
import sys

# Attempt to import plotting libraries, proceed without plotting if missing
try:
    import matplotlib.pyplot as plt
    import numpy as np

    PLOT_ENABLED = True
except ImportError:
    PLOT_ENABLED = False
    # Ensure print statements here are ASCII
    print(
        "Warning: matplotlib or numpy not found. Plotting will be disabled.",
        file=sys.stderr,
    )
    print("Install using: pip install matplotlib numpy", file=sys.stderr)


# --- Configuration ---
# Log file pattern to search for within the directory
LOG_FILE_PATTERN = "*.log"
# Key metrics to extract and aggregate
METRIC_KEYS = ["time_s", "mops_total"]
# Key metrics for the percentage difference plot
METRICS_FOR_PERCENT_DIFF_PLOT = ["time_s", "mops_total"]
# Output file for the text summary
SUMMARY_FILENAME = "summary.txt"
# Threshold for baseline mean close to zero to avoid division issues
ZERO_THRESHOLD = 1e-9


# --- NPB Log Parsing Function ---
# (parse_npb_log remains the same as the previous version)
def parse_npb_log(filepath):
    """Parses a single NPB log file to extract key metrics."""
    metrics = {
        "benchmark_name": None,
        "time_s": None,
        "mops_total": None,
        "class": None,
        "threads": None,
        "verification": None,
        "error": None,
        "filepath": filepath,
    }
    if not os.path.exists(filepath):
        metrics["error"] = f"Error: File not found at {filepath}"
        return metrics
    if not os.path.isfile(filepath):
        metrics["error"] = f"Error: Path is not a file {filepath}"
        return metrics
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        name_match = re.search(
            r"NAS Parallel Benchmarks.*-\s+(\w+)\s+Benchmark", content
        )
        if name_match:
            metrics["benchmark_name"] = name_match.group(1).upper()
        else:
            metrics["error"] = "Could not find benchmark name header."
            return metrics
        time_match = re.search(r"Time in seconds =\s*([\d.]+)", content)
        if time_match:
            metrics["time_s"] = float(time_match.group(1))
        mops_match = re.search(r"Mop/s total\s+=\s*([\d.]+)", content)
        if mops_match:
            metrics["mops_total"] = float(mops_match.group(1))
        class_match = re.search(r"Class\s+=\s*(\w+)", content)
        if class_match:
            metrics["class"] = class_match.group(1).upper()
        threads_match = re.search(r"Total threads\s+=\s*(\d+)", content)
        if threads_match:
            metrics["threads"] = int(threads_match.group(1))
        veri_match = re.search(r"Verification\s+=\s*(\w+)", content)
        if veri_match:
            metrics["verification"] = veri_match.group(1).upper()
        elif "VERIFICATION SUCCESSFUL" in content:
            metrics["verification"] = "SUCCESSFUL"
    except Exception as e:
        err_msg = str(e).encode("ascii", "replace").decode("ascii")
        metrics["error"] = f"An error occurred during parsing {filepath}: {err_msg}"
    if metrics["time_s"] is None or metrics["mops_total"] is None:
        if not metrics["error"]:
            metrics["error"] = (
                f"Could not extract key NPB metrics (Time/Mop/s) from '{os.path.basename(filepath)}'"
            )
    return metrics


# --- Aggregation Function ---
# (aggregate_npb_results remains the same as the previous version)
def aggregate_npb_results(all_run_data):
    """
    Aggregates results from multiple NPB runs (parsed dictionaries).
    Groups by benchmark name.
    """
    collected_data = collections.defaultdict(lambda: collections.defaultdict(list))
    valid_run_count_per_bench = collections.defaultdict(int)
    configs = collections.defaultdict(lambda: {"class": set(), "threads": set()})
    if not all_run_data:
        return None
    for run_data in all_run_data:
        if not run_data or not run_data.get("benchmark_name"):
            continue
        bench_name = run_data["benchmark_name"]
        valid_run_count_per_bench[bench_name] += 1
        for key in METRIC_KEYS:
            value = run_data.get(key)
            if value is not None:
                collected_data[bench_name][key].append(value)
        if run_data.get("class"):
            configs[bench_name]["class"].add(run_data["class"])
        if run_data.get("threads"):
            configs[bench_name]["threads"].add(run_data["threads"])
    aggregated_stats = collections.defaultdict(dict)
    for bench_name, metrics_data in collected_data.items():
        for metric_name, values in metrics_data.items():
            count = len(values)
            if count > 0:
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if count > 1 else 0.0
                aggregated_stats[bench_name][metric_name] = {
                    "mean": mean,
                    "stdev": stdev,
                    "count": count,
                }
        cfg = configs[bench_name]
        aggregated_stats[bench_name]["class"] = (
            list(cfg["class"])[0] if len(cfg["class"]) == 1 else "Mixed"
        )
        aggregated_stats[bench_name]["threads"] = (
            list(cfg["threads"])[0] if len(cfg["threads"]) == 1 else "Mixed"
        )
        aggregated_stats[bench_name]["run_count"] = valid_run_count_per_bench[
            bench_name
        ]
    if not aggregated_stats:
        return None
    return aggregated_stats


# --- Comparison Summary Generation Function (Modified) ---


def generate_comparison_summary(results1, results2, label1, label2):
    """Generates a comparison summary string including percentage difference."""
    if not results1 or not results2:
        return None  # Cannot generate if data is missing

    label1_short = os.path.basename(label1.rstrip("/\\")) or "Baseline"
    label2_short = os.path.basename(label2.rstrip("/\\")) or "Comparison"

    all_benchmarks = sorted(list(set(results1.keys()) | set(results2.keys())))

    summary_lines = []
    separator = "-" * 100  # Adjusted width
    header_line1 = "=" * 100
    header_line2 = "NPB Benchmark Comparison Summary"
    header_line3 = f"Baseline (Dir1): {label1_short}"
    header_line4 = f"Comparison (Dir2): {label2_short}"
    header_line5 = "=" * 100
    # Adjusted column widths
    col_header_format = "{:<10} | {:<18} | {:<25} | {:<25} | {:<12}"
    col_headers = col_header_format.format(
        "Benchmark",
        "Metric",
        f"{label1_short} (Mean+/-Stdev)",
        f"{label2_short} (Mean+/-Stdev)",
        "% Diff",
    )

    summary_lines.extend(
        [
            header_line1,
            header_line2,
            header_line3,
            header_line4,
            header_line5,
            col_headers,
            separator,
        ]
    )

    for benchmark in all_benchmarks:
        res1_bench = results1.get(benchmark, {})
        res2_bench = results2.get(benchmark, {})
        class_val = res1_bench.get("class", res2_bench.get("class", "N/A"))
        threads_val = res1_bench.get("threads", res2_bench.get("threads", "N/A"))

        # Print benchmark info once, aligned with the first column
        summary_lines.append(
            f"{benchmark:<10} (C:{str(class_val)}, T:{str(threads_val)})"
        )

        for metric in METRIC_KEYS:  # Iterate through Time and Mop/s
            res1 = res1_bench.get(metric)
            res2 = res2_bench.get(metric)

            def format_res(res):
                if res:
                    mean_str = f"{res['mean']:.2f}"
                    stdev_str = f"{res['stdev']:.2f}"
                    count = res["count"]
                    return f"{mean_str} +/- {stdev_str} (n={count})"
                else:
                    return "N/A"

            res1_str = format_res(res1)
            res2_str = format_res(res2)

            # Calculate Percentage Difference
            percent_diff_str = "N/A"
            if (
                res1
                and res2
                and res1.get("mean") is not None
                and res2.get("mean") is not None
            ):
                mean1 = res1["mean"]
                mean2 = res2["mean"]
                if abs(mean1) > ZERO_THRESHOLD:
                    percent_diff = ((mean2 - mean1) / mean1) * 100.0
                    if not math.isnan(percent_diff) and not math.isinf(percent_diff):
                        percent_diff_str = (
                            f"{percent_diff:+.2f}%"  # Add sign explicitly
                        )
                elif abs(mean2) < ZERO_THRESHOLD:
                    percent_diff_str = "0.00%"
                # else: leave as N/A

            metric_label = metric.replace("_", " ").title()
            # Print metric line with alignment - use empty first column
            summary_lines.append(
                col_header_format.format(
                    "", metric_label, res1_str, res2_str, percent_diff_str
                )
            )

        summary_lines.append(separator)  # Separator after each benchmark

    return "\n".join(summary_lines)


# --- Plotting Function (Modified) ---


def plot_npb_comparison(agg_data1, agg_data2, label1, label2):
    """
    Creates comparison plots for NPB Time, Mop/s, and Percentage Difference.
    Saves plots as PDF.
    """
    if not PLOT_ENABLED:
        print("\nPlotting disabled as matplotlib/numpy are not installed.")
        return
    if not agg_data1 or not agg_data2:
        print(
            "\nCannot generate plots due to missing aggregated data.", file=sys.stderr
        )
        return

    print("\nGenerating NPB comparison plots...")

    label1_short = os.path.basename(label1.rstrip("/\\")) or "Baseline"
    label2_short = os.path.basename(label2.rstrip("/\\")) or "Comparison"

    # Find common benchmarks
    common_benchmarks = sorted(list(set(agg_data1.keys()) & set(agg_data2.keys())))

    if not common_benchmarks:
        print(
            "Error: No common benchmarks found between the two directories. Cannot generate plots.",
            file=sys.stderr,
        )
        return

    print(f"Found {len(common_benchmarks)} common benchmarks for comparison.")

    # Create output directory if it doesn't exist
    output_plot_dir = f"./{label1_short}-plots"
    if not os.path.exists(output_plot_dir):
        try:
            os.makedirs(output_plot_dir)
            print(f"Created plot output directory: {output_plot_dir}")
        except OSError as e:
            print(
                f"Error: Could not create plot directory '{output_plot_dir}': {e}",
                file=sys.stderr,
            )
            output_plot_dir = "."  # Save in current dir as fallback

    # Helper function for saving plots
    def save_plot(fig, filename_base):
        plot_filename = os.path.join(output_plot_dir, f"{filename_base}.pdf")
        try:
            fig.tight_layout()
            plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
            print(f"   Plot saved to '{plot_filename}'")
        except Exception as e:
            err_msg = str(e).encode("ascii", "replace").decode("ascii")
            print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
        plt.close(fig)

    # --- Plot 1: Time Comparison (Absolute Values) ---
    metric_time = "time_s"
    print(f" - Plotting Absolute '{metric_time}'...")
    means1_t, stdevs1_t = [], []
    means2_t, stdevs2_t = [], []
    valid_bench_t = []
    for bench in common_benchmarks:
        d1_t = agg_data1.get(bench, {}).get(metric_time)
        d2_t = agg_data2.get(bench, {}).get(metric_time)
        if d1_t and d2_t:
            means1_t.append(d1_t["mean"])
            stdevs1_t.append(d1_t["stdev"])
            means2_t.append(d2_t["mean"])
            stdevs2_t.append(d2_t["stdev"])
            valid_bench_t.append(bench)
    if valid_bench_t:
        x_indices = np.arange(len(valid_bench_t))
        bar_width = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(valid_bench_t) * 1.0), 6))
        r1 = ax.bar(
            x_indices - bar_width / 2,
            means1_t,
            bar_width,
            yerr=stdevs1_t,
            label=label1_short,
            capsize=5,
            alpha=0.8,
        )
        r2 = ax.bar(
            x_indices + bar_width / 2,
            means2_t,
            bar_width,
            yerr=stdevs2_t,
            label=label2_short,
            capsize=5,
            alpha=0.8,
        )
        ax.set_ylabel("Time (seconds)")
        ax.set_title("NPB Benchmark Time Comparison")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(valid_bench_t, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_ylim(bottom=0)
        save_plot(fig, "npb_comparison_time")
    else:
        print(f"   Skipping Time plot: No common benchmarks with data.")

    # --- Plot 2: Mop/s Comparison (Absolute Values) ---
    metric_mops = "mops_total"
    print(f" - Plotting Absolute '{metric_mops}'...")
    means1_m, stdevs1_m = [], []
    means2_m, stdevs2_m = [], []
    valid_bench_m = []
    for bench in common_benchmarks:
        d1_m = agg_data1.get(bench, {}).get(metric_mops)
        d2_m = agg_data2.get(bench, {}).get(metric_mops)
        if d1_m and d2_m:
            means1_m.append(d1_m["mean"])
            stdevs1_m.append(d1_m["stdev"])
            means2_m.append(d2_m["mean"])
            stdevs2_m.append(d2_m["stdev"])
            valid_bench_m.append(bench)
    if valid_bench_m:
        x_indices = np.arange(len(valid_bench_m))
        bar_width = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(valid_bench_m) * 1.0), 6))
        r1 = ax.bar(
            x_indices - bar_width / 2,
            means1_m,
            bar_width,
            yerr=stdevs1_m,
            label=label1_short,
            capsize=5,
            alpha=0.8,
        )
        r2 = ax.bar(
            x_indices + bar_width / 2,
            means2_m,
            bar_width,
            yerr=stdevs2_m,
            label=label2_short,
            capsize=5,
            alpha=0.8,
        )
        ax.set_ylabel("Mop/s Total")
        ax.set_title("NPB Benchmark Mop/s Comparison")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(valid_bench_m, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_ylim(bottom=0)
        save_plot(fig, "npb_comparison_mops")
    else:
        print(f"   Skipping Mop/s plot: No common benchmarks with data.")

    # --- Plot 3: Percentage Difference ---
    print(f" - Plotting Percentage Difference ({label2_short} vs {label1_short})...")
    percent_diff_time = []
    percent_diff_mops = []
    plot_labels_p = []  # Benchmarks included in this plot

    for bench in common_benchmarks:
        d1_t = agg_data1.get(bench, {}).get(metric_time)
        d2_t = agg_data2.get(bench, {}).get(metric_time)
        d1_m = agg_data1.get(bench, {}).get(metric_mops)
        d2_m = agg_data2.get(bench, {}).get(metric_mops)

        # Calculate diffs only if both metrics are present in both datasets for this bench
        if d1_t and d2_t and d1_m and d2_m:
            mean1_t = d1_t["mean"]
            mean2_t = d2_t["mean"]
            diff_t = float("nan")
            if abs(mean1_t) > ZERO_THRESHOLD:
                diff = ((mean2_t - mean1_t) / mean1_t) * 100.0
                if not math.isnan(diff) and not math.isinf(diff):
                    diff_t = diff
            elif abs(mean2_t) < ZERO_THRESHOLD:
                diff_t = 0.0

            mean1_m = d1_m["mean"]
            mean2_m = d2_m["mean"]
            diff_m = float("nan")
            if abs(mean1_m) > ZERO_THRESHOLD:
                diff = ((mean2_m - mean1_m) / mean1_m) * 100.0
                if not math.isnan(diff) and not math.isinf(diff):
                    diff_m = diff
            elif abs(mean2_m) < ZERO_THRESHOLD:
                diff_m = 0.0

            # Only add if at least one valid diff was calculated
            if not (math.isnan(diff_t) and math.isnan(diff_m)):
                percent_diff_time.append(
                    diff_t if not math.isnan(diff_t) else 0
                )  # Plot NaN as 0? Or filter later
                percent_diff_mops.append(diff_m if not math.isnan(diff_m) else 0)
                plot_labels_p.append(bench)
        # else: skip benchmark if data is missing

    if plot_labels_p:
        x_indices = np.arange(len(plot_labels_p))
        bar_width = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(plot_labels_p) * 1.0), 6))

        # Plot bars, handle potential NaNs stored as 0 if needed (though filtering before plot is better)
        rects1 = ax.bar(
            x_indices - bar_width / 2,
            percent_diff_time,
            bar_width,
            label="Time % Diff",
            alpha=0.8,
        )
        rects2 = ax.bar(
            x_indices + bar_width / 2,
            percent_diff_mops,
            bar_width,
            label="Mop/s % Diff",
            alpha=0.8,
        )

        ax.set_ylabel("Percentage Difference (%)")
        ax.set_title(f"NPB Performance % Difference ({label2_short} vs {label1_short})")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(plot_labels_p, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.axhline(0, color="grey", linewidth=0.8)  # Line at 0%

        ax.bar_label(rects1, padding=3, fmt="%.1f%%")
        ax.bar_label(rects2, padding=3, fmt="%.1f%%")

        max_abs_diff = (
            max(
                abs(v)
                for v in percent_diff_time + percent_diff_mops
                if not math.isnan(v)
            )
            if any(not math.isnan(v) for v in percent_diff_time + percent_diff_mops)
            else 10
        )
        ax.set_ylim(
            bottom=min(0, -max_abs_diff * 1.15) - 5, top=max(0, max_abs_diff * 1.15) + 5
        )  # Add buffer

        save_plot(fig, "npb_comparison_percent_diff")
    else:
        print(
            f"   Skipping Percentage Difference plot: No common benchmarks with data for comparison."
        )


# --- Directory Processing Function ---
# (process_directory remains the same as the previous version)
def process_directory(directory_path):
    """Finds, parses, and aggregates NPB log files in a directory."""
    print(f"\nProcessing directory: {directory_path}")
    if not os.path.isdir(directory_path):
        print(
            f"Error: Provided path '{directory_path}' is not a valid directory.",
            file=sys.stderr,
        )
        return None, 0
    search_pattern = os.path.join(directory_path, LOG_FILE_PATTERN)
    log_files = glob.glob(search_pattern)
    if not log_files:
        print(
            f"Warning: No files matching pattern '{LOG_FILE_PATTERN}' found in directory '{directory_path}'.",
            file=sys.stderr,
        )
        return None, 0
    all_run_results = []
    parse_errors = 0
    files_found_count = len(log_files)
    print(
        f"Found {files_found_count} file(s) matching '{LOG_FILE_PATTERN}'. Parsing..."
    )
    for log_file in log_files:
        metrics = parse_npb_log(log_file)
        if metrics.get("error"):
            if "Could not extract key NPB metrics" not in metrics["error"]:
                print(
                    f"\n   Parse Error in '{os.path.basename(log_file)}': {metrics['error']}",
                    file=sys.stderr,
                )
            parse_errors += 1
        elif metrics and metrics.get("benchmark_name"):
            all_run_results.append(metrics)
    valid_files_parsed = len(all_run_results)
    print(f"Successfully parsed data from {valid_files_parsed} file(s).")
    if parse_errors > 0:
        print(
            f"Skipped or encountered errors in {parse_errors} file(s).", file=sys.stderr
        )
    if valid_files_parsed > 0:
        aggregated_data = aggregate_npb_results(all_run_results)
        return aggregated_data, valid_files_parsed
    else:
        print(f"No valid data collected from directory '{directory_path}'.")
        return None, 0


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare NASA NPB benchmark logs from two directories, print summaries with % diff, save summary, and plot comparisons (ASCII only)."
    )
    parser.add_argument(
        "dir1",
        metavar="BASELINE_DIR",
        type=str,
        help="Path to the baseline directory containing NPB benchmark *.log files.",
    )
    parser.add_argument(
        "dir2",
        metavar="COMPARISON_DIR",
        type=str,
        help="Path to the comparison directory containing NPB benchmark *.log files.",
    )
    args = parser.parse_args()

    agg_data1, count1 = process_directory(args.dir1)
    agg_data2, count2 = process_directory(args.dir2)

    summary_text = None
    if agg_data1 and agg_data2:
        summary_text = generate_comparison_summary(
            agg_data1, agg_data2, args.dir1, args.dir2
        )

    if summary_text:
        SUMMARY_FILENAME = f"./{args.dir1}-{SUMMARY_FILENAME}.txt"

        print("\n" + summary_text)
        try:
            # Ensure output directory exists for summary file
            summary_dir = os.path.dirname(SUMMARY_FILENAME)
            if summary_dir and not os.path.exists(summary_dir):
                os.makedirs(summary_dir)  # Create dir if needed

            with open(SUMMARY_FILENAME, "w", encoding="ascii") as f_summary:
                f_summary.write(summary_text)
            print(f"\nComparison summary saved to '{SUMMARY_FILENAME}'")
        except IOError as e:
            err_msg = str(e).encode("ascii", "replace").decode("ascii")
            print(
                f"\nError: Could not write summary to file '{SUMMARY_FILENAME}': {err_msg}",
                file=sys.stderr,
            )
        except OSError as e:
            err_msg = str(e).encode("ascii", "replace").decode("ascii")
            print(
                f"\nError: Could not create directory for summary file '{SUMMARY_FILENAME}': {err_msg}",
                file=sys.stderr,
            )
    else:
        print(
            "\nComparison summary cannot be generated due to missing data.",
            file=sys.stderr,
        )

    if agg_data1 and agg_data2:
        plot_npb_comparison(agg_data1, agg_data2, args.dir1, args.dir2)
    elif PLOT_ENABLED:
        print(
            "\nComparison plots cannot be generated as data from both directories is required.",
            file=sys.stderr,
        )

    print("\nScript finished.")

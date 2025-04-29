# Ensure only ASCII characters are used in this script
"""
Processes NASA NPB benchmark log files (*.log) from two directories,
calculates statistics (mean +/- stdev) for each benchmark within each file,
prints a comparison summary including percentage differences (Dir2 vs Dir1),
and generates comparison plots saved as PDF files.
Uses only ASCII characters.
"""

import argparse
import collections
import csv  # Although not reading CSV, keep for potential future use? No, remove.
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
        "verification": None,  # SUCCESSFUL or FAILED/Unknown
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
        # Use utf-8 with replace for robustness
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Benchmark Name (e.g., BT, CG, EP)
        name_match = re.search(
            r"NAS Parallel Benchmarks.*-\s+(\w+)\s+Benchmark", content
        )
        if name_match:
            metrics["benchmark_name"] = name_match.group(1).upper()
        else:
            metrics["error"] = "Could not find benchmark name header."
            return metrics  # Cannot proceed without benchmark name

        # Time in seconds
        time_match = re.search(r"Time in seconds =\s*([\d.]+)", content)
        if time_match:
            metrics["time_s"] = float(time_match.group(1))

        # Mop/s total
        mops_match = re.search(r"Mop/s total\s+=\s*([\d.]+)", content)
        if mops_match:
            metrics["mops_total"] = float(mops_match.group(1))

        # Class
        class_match = re.search(r"Class\s+=\s*(\w+)", content)
        if class_match:
            metrics["class"] = class_match.group(1).upper()

        # Threads
        threads_match = re.search(r"Total threads\s+=\s*(\d+)", content)
        if threads_match:
            metrics["threads"] = int(threads_match.group(1))

        # Verification
        veri_match = re.search(r"Verification\s+=\s*(\w+)", content)
        if veri_match:
            metrics["verification"] = veri_match.group(1).upper()
        elif "VERIFICATION SUCCESSFUL" in content:
            metrics["verification"] = "SUCCESSFUL"

    except Exception as e:
        err_msg = str(e).encode("ascii", "replace").decode("ascii")
        metrics["error"] = f"An error occurred during parsing {filepath}: {err_msg}"

    # Basic Validation
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

    # Step 1: Collect all values and config info, grouped by benchmark
    for run_data in all_run_data:
        if not run_data or not run_data.get("benchmark_name"):
            continue

        bench_name = run_data["benchmark_name"]
        valid_run_count_per_bench[bench_name] += 1

        for key in METRIC_KEYS:  # Use defined METRIC_KEYS
            value = run_data.get(key)
            if value is not None:
                collected_data[bench_name][key].append(value)

        if run_data.get("class"):
            configs[bench_name]["class"].add(run_data["class"])
        if run_data.get("threads"):
            configs[bench_name]["threads"].add(run_data["threads"])

    # Step 2: Calculate statistics
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


# --- Comparison Summary Printing Function (Modified) ---


def print_comparison_summary(results1, results2, label1, label2):
    """Prints a side-by-side comparison summary including percentage difference."""
    if not results1 or not results2:
        print("\nCannot print comparison summary due to missing data.")
        return

    label1_short = os.path.basename(label1.rstrip("/\\")) or "Baseline"
    label2_short = os.path.basename(label2.rstrip("/\\")) or "Comparison"

    all_benchmarks = sorted(list(set(results1.keys()) | set(results2.keys())))

    print("\n" + "=" * 95)
    print("NPB Benchmark Comparison Summary")
    print(f"Baseline (Dir1): {label1_short}")
    print(f"Comparison (Dir2): {label2_short}")
    print("=" * 95)
    # Header line
    header_format = "{:<10} | {:<15} | {:<25} | {:<25} | {:<12}"
    print(
        header_format.format(
            "Benchmark",
            "Metric",
            f"{label1_short} (Mean+/-Stdev)",
            f"{label2_short} (Mean+/-Stdev)",
            "% Diff",
        )
    )
    print("-" * 95)

    for benchmark in all_benchmarks:
        # Get config info (assume consistent or take from first dict)
        res1_bench = results1.get(benchmark, {})
        res2_bench = results2.get(benchmark, {})
        class_val = res1_bench.get("class", res2_bench.get("class", "N/A"))
        threads_val = res1_bench.get("threads", res2_bench.get("threads", "N/A"))

        print(
            f"{benchmark:<10} (C:{str(class_val)}, T:{str(threads_val)})"
        )  # Print benchmark info once

        for metric in METRIC_KEYS:
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
            if res1 and res2 and res1["mean"] is not None and res2["mean"] is not None:
                mean1 = res1["mean"]
                mean2 = res2["mean"]
                # Avoid division by zero or near-zero
                if abs(mean1) > ZERO_THRESHOLD:
                    percent_diff = ((mean2 - mean1) / mean1) * 100.0
                    # Check for NaN or Inf just in case, though abs() check should prevent
                    if not math.isnan(percent_diff) and not math.isinf(percent_diff):
                        percent_diff_str = (
                            f"{percent_diff:+.2f}%"  # Add sign explicitly
                        )
                elif abs(mean2) < ZERO_THRESHOLD:  # Both are near zero
                    percent_diff_str = "0.00%"
                # else: Mean1 is zero/small, Mean2 is not - difference is large/infinite, leave as N/A

            metric_label = metric.replace("_", " ").title()
            # Print metric line with alignment
            print(
                header_format.format(
                    "", metric_label, res1_str, res2_str, percent_diff_str
                )
            )

        print("-" * 95)


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
    plot_bench_labels = common_benchmarks  # Use benchmark names as labels

    # --- Plot 1: Time Comparison (Absolute Values) ---
    metric_time = "time_s"
    print(f" - Plotting '{metric_time}'...")
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
        rects1 = ax.bar(
            x_indices - bar_width / 2,
            means1_t,
            bar_width,
            yerr=stdevs1_t,
            label=label1_short,
            capsize=5,
            alpha=0.8,
        )
        rects2 = ax.bar(
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
        fig.tight_layout()
        plot_filename = "npb_comparison_time.pdf"
        try:
            plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
            print(f"   Plot saved to '{plot_filename}'")
        except Exception as e:
            err_msg = str(e).encode("ascii", "replace").decode("ascii")
            print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
        plt.close(fig)
    else:
        print(f"   Skipping Time plot: No common benchmarks with data.")

    # --- Plot 2: Mop/s Comparison (Absolute Values) ---
    metric_mops = "mops_total"
    print(f" - Plotting '{metric_mops}'...")
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
        rects1 = ax.bar(
            x_indices - bar_width / 2,
            means1_m,
            bar_width,
            yerr=stdevs1_m,
            label=label1_short,
            capsize=5,
            alpha=0.8,
        )
        rects2 = ax.bar(
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
        fig.tight_layout()
        plot_filename = "npb_comparison_mops.pdf"
        try:
            plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
            print(f"   Plot saved to '{plot_filename}'")
        except Exception as e:
            err_msg = str(e).encode("ascii", "replace").decode("ascii")
            print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
        plt.close(fig)
    else:
        print(f"   Skipping Mop/s plot: No common benchmarks with data.")

    # --- Plot 3: Percentage Difference ---
    print(f" - Plotting Percentage Difference ({label2_short} vs {label1_short})...")
    percent_diff_time = []
    percent_diff_mops = []
    plot_labels_p = []

    for bench in common_benchmarks:
        d1_t = agg_data1.get(bench, {}).get(metric_time)
        d2_t = agg_data2.get(bench, {}).get(metric_time)
        d1_m = agg_data1.get(bench, {}).get(metric_mops)
        d2_m = agg_data2.get(bench, {}).get(metric_mops)

        # Only include benchmark if both metrics are available in both files
        if d1_t and d2_t and d1_m and d2_m:
            # Calculate % diff for Time
            mean1_t = d1_t["mean"]
            mean2_t = d2_t["mean"]
            if abs(mean1_t) > ZERO_THRESHOLD:
                diff_t = ((mean2_t - mean1_t) / mean1_t) * 100.0
                percent_diff_time.append(
                    diff_t if not (math.isnan(diff_t) or math.isinf(diff_t)) else 0
                )  # Append 0 if invalid? Or skip?
            elif abs(mean2_t) < ZERO_THRESHOLD:
                percent_diff_time.append(0.0)  # Both near zero
            else:
                percent_diff_time.append(
                    np.nan
                )  # Cannot calculate meaningfully, use NaN to skip plotting

            # Calculate % diff for Mop/s
            mean1_m = d1_m["mean"]
            mean2_m = d2_m["mean"]
            if abs(mean1_m) > ZERO_THRESHOLD:
                diff_m = ((mean2_m - mean1_m) / mean1_m) * 100.0
                percent_diff_mops.append(
                    diff_m if not (math.isnan(diff_m) or math.isinf(diff_m)) else 0
                )
            elif abs(mean2_m) < ZERO_THRESHOLD:
                percent_diff_mops.append(0.0)
            else:
                percent_diff_mops.append(np.nan)

            # Only add label if at least one valid diff was calculated
            if not (
                math.isnan(percent_diff_time[-1]) and math.isnan(percent_diff_mops[-1])
            ):
                plot_labels_p.append(bench)
            else:  # Remove the NaNs if both failed
                percent_diff_time.pop()
                percent_diff_mops.pop()

    if plot_labels_p:
        x_indices = np.arange(len(plot_labels_p))
        bar_width = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(plot_labels_p) * 1.0), 6))

        # Filter out NaN values before plotting - this requires adjusting indices too.
        # Simpler approach: plot NaNs, they might just appear as gaps or zero depending on MPL version.
        # Let's proceed assuming NaNs might render as gaps or can be handled.
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

        # Adjust y limits to show positive and negative differences clearly
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
            bottom=min(0, -max_abs_diff * 1.15), top=max(0, max_abs_diff * 1.15)
        )

        fig.tight_layout()
        plot_filename = "npb_comparison_percent_diff.pdf"
        try:
            plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
            print(f"   Plot saved to '{plot_filename}'")
        except Exception as e:
            err_msg = str(e).encode("ascii", "replace").decode("ascii")
            print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
        plt.close(fig)
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
        description="Compare NASA NPB benchmark logs from two directories, print summaries with % diff, and plot comparisons (ASCII only)."
    )
    parser.add_argument(
        "dir1",
        metavar="BASELINE_DIR",  # Changed label
        type=str,
        help="Path to the baseline directory containing NPB benchmark *.log files.",
    )
    parser.add_argument(
        "dir2",
        metavar="COMPARISON_DIR",  # Changed label
        type=str,
        help="Path to the comparison directory containing NPB benchmark *.log files.",
    )
    args = parser.parse_args()

    # Process first directory (baseline)
    agg_data1, count1 = process_directory(args.dir1)

    # Process second directory (comparison)
    agg_data2, count2 = process_directory(args.dir2)

    # Print comparison summary if both directories yielded data
    if agg_data1 and agg_data2:
        print_comparison_summary(agg_data1, agg_data2, args.dir1, args.dir2)
        # Generate comparison plots
        plot_npb_comparison(agg_data1, agg_data2, args.dir1, args.dir2)
    else:
        print(
            "\nComparison summary and plots cannot be generated as data from both directories is required.",
            file=sys.stderr,
        )
        if PLOT_ENABLED:  # Check if plotting was supposed to run
            print("Plotting skipped.", file=sys.stderr)

    print("\nScript finished.")

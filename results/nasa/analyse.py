# Ensure only ASCII characters are used in this script
"""
Processes NASA NPB benchmark log files (*.log) from two directories,
prints an aggregated summary for each directory based on benchmark type,
and generates comparison plots saved as PDF files.
Uses only ASCII characters.
"""

import argparse
import collections
import glob
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


# --- NPB Log Parsing Function ---


def parse_npb_log(filepath):
    """Parses a single NPB log file to extract key metrics."""
    # Metrics dictionary for a single run (file)
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

        # --- Extracting Metrics using Regular Expressions ---

        # Benchmark Name (e.g., BT, CG, EP)
        name_match = re.search(
            r"NAS Parallel Benchmarks.*-\s+(\w+)\s+Benchmark", content
        )
        if name_match:
            metrics["benchmark_name"] = name_match.group(1).upper()
        else:
            metrics["error"] = "Could not find benchmark name header."
            # Try to guess from filename? Safer to error out.
            # filename_base = os.path.basename(filepath)
            # common_names = ['bt', 'cg', 'ep', 'ft', 'is', 'lu', 'mg', 'sp', 'ua']
            # for name in common_names:
            #      if name in filename_base.lower():
            #           metrics["benchmark_name"] = name.upper()
            #           print(f"   Warning: Guessed benchmark name '{metrics['benchmark_name']}' from filename for {filepath}", file=sys.stderr)
            #           break
            # if not metrics["benchmark_name"]:
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
        elif "VERIFICATION SUCCESSFUL" in content:  # Handle CG/MG format
            metrics["verification"] = "SUCCESSFUL"

    except Exception as e:
        # Ensure error message uses only ASCII characters if possible
        err_msg = str(e).encode("ascii", "replace").decode("ascii")
        metrics["error"] = f"An error occurred during parsing {filepath}: {err_msg}"

    # --- Basic Validation ---
    if metrics["time_s"] is None or metrics["mops_total"] is None:
        if not metrics["error"]:
            metrics["error"] = (
                f"Could not extract key NPB metrics (Time/Mop/s) from '{os.path.basename(filepath)}'"
            )

    return metrics


# --- Aggregation Function ---


def aggregate_npb_results(all_run_data):
    """
    Aggregates results from multiple NPB runs (parsed dictionaries).
    Groups by benchmark name.
    """
    # Structure: benchmark_name -> metric_name -> list_of_values
    collected_data = collections.defaultdict(lambda: collections.defaultdict(list))
    valid_run_count_per_bench = collections.defaultdict(int)
    configs = collections.defaultdict(lambda: {"class": set(), "threads": set()})

    if not all_run_data:
        return None

    # Step 1: Collect all values and config info, grouped by benchmark
    for run_data in all_run_data:
        if not run_data or not run_data.get("benchmark_name"):
            continue  # Skip if basic info is missing

        bench_name = run_data["benchmark_name"]
        valid_run_count_per_bench[bench_name] += 1

        # Collect metrics
        for key in METRIC_KEYS:
            value = run_data.get(key)
            if value is not None:
                collected_data[bench_name][key].append(value)

        # Collect config info for consistency check
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
        # Add config info to the aggregated stats for this benchmark
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

    if not aggregated_stats:  # Check if any stats were actually calculated
        return None

    return aggregated_stats


# --- Summary Printing Function ---


def print_npb_summary(agg_results, directory_path):
    """Prints the aggregated NPB summary (mean +/- stdev) for a directory."""
    if not agg_results:
        print(
            f"\n--- No aggregated results to display for directory: {directory_path} ---",
            file=sys.stderr,
        )
        return

    print("\n" + "=" * 75)
    print(f"NPB Aggregated Benchmark Summary")
    print(f"Directory Processed : {directory_path}")
    print("=" * 75)

    # Header
    print(
        f"{'Benchmark':<10} {'Class':<6} {'Threads':<8} | {'Metric':<15} | {'Mean +/- Stdev':<25} {'Runs (n)':<5}"
    )
    print("-" * 75)

    # Sort benchmarks for consistent output
    for benchmark_name in sorted(agg_results.keys()):
        bench_data = agg_results[benchmark_name]
        class_val = bench_data.get("class", "N/A")
        threads_val = bench_data.get("threads", "N/A")
        run_count = bench_data.get("run_count", 0)

        # Print benchmark info once
        print(
            f"{benchmark_name:<10} {str(class_val):<6} {str(threads_val):<8} |", end=""
        )

        # Print Time
        time_data = bench_data.get("time_s")
        if time_data:
            mean_str = f"{time_data['mean']:.2f}"
            stdev_str = f"{time_data['stdev']:.2f}"
            print(
                f" {'Time (s)':<15} | {mean_str:>10} +/- {stdev_str:<10} | {time_data['count']:<5}"
            )
        else:
            print(f" {'Time (s)':<15} | {'N/A':<25} | {'0':<5}")

        # Print Mop/s on the next line, aligned
        mops_data = bench_data.get("mops_total")
        if mops_data:
            mean_str = f"{mops_data['mean']:.2f}"
            stdev_str = f"{mops_data['stdev']:.2f}"
            # Use spaces for alignment
            print(
                f"{'':<10} {'':<6} {'':<8} | {'Mop/s total':<15} | {mean_str:>10} +/- {stdev_str:<10} | {mops_data['count']:<5}"
            )
        else:
            print(
                f"{'':<10} {'':<6} {'':<8} | {'Mop/s total':<15} | {'N/A':<25} | {'0':<5}"
            )

        print("-" * 75)


# --- Plotting Function ---


def plot_npb_comparison(agg_data1, agg_data2, label1, label2):
    """
    Creates separate comparison plots for NPB Time and Mop/s.
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

    label1_short = os.path.basename(label1.rstrip("/\\")) or "Dir 1"
    label2_short = os.path.basename(label2.rstrip("/\\")) or "Dir 2"

    # Find common benchmarks
    common_benchmarks = sorted(list(set(agg_data1.keys()) & set(agg_data2.keys())))

    if not common_benchmarks:
        print(
            "Error: No common benchmarks found between the two directories. Cannot generate plots.",
            file=sys.stderr,
        )
        return

    print(f"Found {len(common_benchmarks)} common benchmarks for comparison.")

    # --- Plot 1: Time Comparison ---
    metric_time = "time_s"
    print(f" - Plotting '{metric_time}'...")
    means1_t, stdevs1_t = [], []
    means2_t, stdevs2_t = [], []
    plot_labels_t = []

    for bench in common_benchmarks:
        d1_t = agg_data1.get(bench, {}).get(metric_time)
        d2_t = agg_data2.get(bench, {}).get(metric_time)
        if d1_t and d2_t:
            means1_t.append(d1_t["mean"])
            stdevs1_t.append(d1_t["stdev"])
            means2_t.append(d2_t["mean"])
            stdevs2_t.append(d2_t["stdev"])
            plot_labels_t.append(bench)

    if plot_labels_t:
        x_indices = np.arange(len(plot_labels_t))
        bar_width = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(plot_labels_t) * 1.0), 6))
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
        ax.set_xticklabels(plot_labels_t, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        # ax.bar_label(rects1, padding=3, fmt='%.1f') # Optional labels
        # ax.bar_label(rects2, padding=3, fmt='%.1f')
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

    # --- Plot 2: Mop/s Comparison ---
    metric_mops = "mops_total"
    print(f" - Plotting '{metric_mops}'...")
    means1_m, stdevs1_m = [], []
    means2_m, stdevs2_m = [], []
    plot_labels_m = []

    for bench in common_benchmarks:
        d1_m = agg_data1.get(bench, {}).get(metric_mops)
        d2_m = agg_data2.get(bench, {}).get(metric_mops)
        if d1_m and d2_m:
            means1_m.append(d1_m["mean"])
            stdevs1_m.append(d1_m["stdev"])
            means2_m.append(d2_m["mean"])
            stdevs2_m.append(d2_m["stdev"])
            plot_labels_m.append(bench)

    if plot_labels_m:
        x_indices = np.arange(len(plot_labels_m))
        bar_width = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(plot_labels_m) * 1.0), 6))
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
        ax.set_xticklabels(plot_labels_m, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        # ax.bar_label(rects1, padding=3, fmt='%.0f') # Optional labels
        # ax.bar_label(rects2, padding=3, fmt='%.0f')
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


# --- Directory Processing Function ---


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
            # Only print error if it's not the 'missing key metrics' one, unless verbose
            if "Could not extract key NPB metrics" not in metrics["error"]:
                print(
                    f"\n   Parse Error in '{os.path.basename(log_file)}': {metrics['error']}",
                    file=sys.stderr,
                )
            parse_errors += 1
        elif metrics and metrics.get(
            "benchmark_name"
        ):  # Check if metrics dict is valid and has name
            all_run_results.append(metrics)

    valid_files_parsed = len(all_run_results)
    print(f"Successfully parsed data from {valid_files_parsed} file(s).")
    if parse_errors > 0:
        print(
            f"Skipped or encountered errors in {parse_errors} file(s).", file=sys.stderr
        )

    if valid_files_parsed > 0:
        # Aggregate results grouped by benchmark name found in this directory
        aggregated_data = aggregate_npb_results(all_run_results)
        return aggregated_data, valid_files_parsed  # Return aggregated data and count
    else:
        print(f"No valid data collected from directory '{directory_path}'.")
        return None, 0


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare NASA NPB benchmark logs from two directories, print summaries, and plot comparisons (ASCII only)."
    )
    parser.add_argument(
        "dir1",
        metavar="DIRECTORY_1",
        type=str,
        help="Path to the first directory containing NPB benchmark *.log files.",
    )
    parser.add_argument(
        "dir2",
        metavar="DIRECTORY_2",
        type=str,
        help="Path to the second directory containing NPB benchmark *.log files.",
    )
    args = parser.parse_args()

    # Process first directory
    agg_data1, count1 = process_directory(args.dir1)

    # Process second directory
    agg_data2, count2 = process_directory(args.dir2)

    # Print summaries if data was aggregated
    if agg_data1:
        print_npb_summary(agg_data1, args.dir1)
    if agg_data2:
        print_npb_summary(agg_data2, args.dir2)

    # Generate comparison plot if both directories yielded data
    if agg_data1 and agg_data2:
        plot_npb_comparison(agg_data1, agg_data2, args.dir1, args.dir2)
    elif PLOT_ENABLED:
        print(
            "\nComparison plots cannot be generated as data from both directories is required.",
            file=sys.stderr,
        )

    print("\nScript finished.")

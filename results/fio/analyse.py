# Ensure only ASCII characters are used in this script
"""
Processes fio benchmark log files (*.log) from two directories,
prints an aggregated summary for each directory, and generates
comparison plots saved as PDF files. Uses only ASCII characters.
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
# Specific clat percentiles to extract and aggregate (use strings like '99.00')
PERCENTILES_TO_AGGREGATE = ["99.00", "99.90", "99.99"]


# --- Fio Log Parsing Function ---


def parse_fio_log(filepath):
    """Parses a single fio log file to extract key metrics."""
    metrics = {
        "iops": None,
        "bandwidth_mibps": None,
        "clat_avg_ms": None,
        "clat_max_ms": None,
        # Percentiles will be added dynamically like clat_p99_00_ms etc.
        "cpu_usr_pct": None,
        "cpu_sys_pct": None,
        "disk_util_pct": None,
        "duration_sec": None,  # Added from run summary
        "operation": None,  # 'read' or 'write'
        "error": None,
        "filepath": filepath,
    }
    # Temporary storage for percentiles before potential unit conversion
    clat_percentiles_us = {}

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
        # Determine operation (read/write) - prioritize specific summary lines
        op = None
        bw_match = re.search(
            r"\s*(read|write)\s*:\s*IOPS=([\d.]+),\s*BW=([\d.]+)MiB/s", content
        )
        if bw_match:
            op = bw_match.group(1).lower()
            metrics["operation"] = op
            metrics["iops"] = float(bw_match.group(2))
            metrics["bandwidth_mibps"] = float(bw_match.group(3))
        else:
            # Fallback to Run status group (less preferable as IOPS isn't there)
            bw_match_fallback = re.search(
                r"\s*(READ|WRITE)\s*:\s*bw=([\d.]+)MiB/s", content
            )
            if bw_match_fallback:
                op = bw_match_fallback.group(1).lower()
                metrics["operation"] = op
                metrics["bandwidth_mibps"] = float(bw_match_fallback.group(2))
                # Try finding IOPS separately if needed
                iops_match_fallback = re.search(
                    r"\s+iops\s+:\s+.*?avg=([\d.]+),", content
                )
                if iops_match_fallback:
                    metrics["iops"] = float(iops_match_fallback.group(1))

        if not op:
            # Could not determine operation type or find key stats
            if not metrics["error"]:  # Avoid overwriting previous error
                metrics["error"] = (
                    "Could not determine operation type (read/write) or find key BW/IOPS stats."
                )
            return metrics  # Cannot proceed reliably without operation type

        # Completion Latency (clat) stats - avg, max in usec
        clat_match = re.search(
            r"clat \(usec\):\s*min=\d+,\s*max=(\d+),\s*avg=([\d.]+),", content
        )
        if clat_match:
            metrics["clat_max_ms"] = (
                float(clat_match.group(1)) / 1000.0
            )  # convert usec to ms
            metrics["clat_avg_ms"] = (
                float(clat_match.group(2)) / 1000.0
            )  # convert usec to ms

        # Completion Latency Percentiles (usec)
        percentile_matches = re.findall(r"\|\s*([\d\.]+)th=\[(\d+)\],?", content)
        for p_match in percentile_matches:
            percentile_key = f"{float(p_match[0]):.2f}"  # Format like 5.00, 99.90 etc.
            clat_percentiles_us[percentile_key] = int(p_match[1])
            # Add specific percentile metrics after conversion
            if percentile_key in PERCENTILES_TO_AGGREGATE:
                metric_key = f"clat_p{percentile_key.replace('.', '_')}_ms"
                metrics[metric_key] = (
                    clat_percentiles_us[percentile_key] / 1000.0
                )  # convert usec to ms

        # CPU Usage
        cpu_match = re.search(r"cpu\s+:\s+usr=([\d\.]+)%,\s+sys=([\d\.]+)%,", content)
        if cpu_match:
            metrics["cpu_usr_pct"] = float(cpu_match.group(1))
            metrics["cpu_sys_pct"] = float(cpu_match.group(2))

        # Disk Utilization (find the last occurrence)
        util_matches = re.findall(r"util=([\d\.]+)%", content)
        if util_matches:
            metrics["disk_util_pct"] = float(util_matches[-1])

        # Duration from Run status group (msec)
        duration_match = re.search(r"run=(\d+)-(\d+)msec", content)
        if duration_match:
            metrics["duration_sec"] = int(duration_match.group(1)) / 1000.0

    except Exception as e:
        # Ensure error message uses only ASCII characters if possible
        err_msg = str(e).encode("ascii", "replace").decode("ascii")
        metrics["error"] = f"An error occurred during parsing {filepath}: {err_msg}"

    # --- Basic Validation ---
    if (
        metrics["iops"] is None
        or metrics["bandwidth_mibps"] is None
        or metrics["clat_avg_ms"] is None
    ):
        if not metrics["error"]:
            metrics["error"] = (
                f"Could not extract key fio metrics (IOPS/BW/clat) from '{os.path.basename(filepath)}'"
            )

    return metrics


# --- Aggregation Function ---


def aggregate_fio_results(all_metrics_data):
    """Aggregates results from multiple fio log files."""
    collected_data = collections.defaultdict(list)
    valid_run_count = len(all_metrics_data)

    if valid_run_count == 0:
        return None

    # Determine the set of metric keys dynamically based on parsed data + required percentiles
    keys_to_aggregate = set()
    for run_data in all_metrics_data:
        if run_data:
            keys_to_aggregate.update(
                k for k, v in run_data.items() if isinstance(v, (int, float))
            )

    # Ensure required percentile keys are included if needed later
    for p in PERCENTILES_TO_AGGREGATE:
        keys_to_aggregate.add(f"clat_p{p.replace('.', '_')}_ms")

    # Step 1: Collect all values
    operations = set()  # Keep track of operations found
    for run_data in all_metrics_data:
        if not run_data:
            continue
        op = run_data.get("operation")
        if op:
            operations.add(op)

        for key in keys_to_aggregate:
            value = run_data.get(key)
            if value is not None:
                collected_data[key].append(value)

    # Step 2: Calculate statistics
    aggregated_stats = {}
    for metric_name, values in collected_data.items():
        count = len(values)
        if count > 0:
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if count > 1 else 0.0
            aggregated_stats[metric_name] = {
                "mean": mean,
                "stdev": stdev,
                "count": count,
            }

    if not aggregated_stats:
        return None

    aggregated_stats["valid_run_count"] = valid_run_count
    # Add the dominant operation type (useful for titles/labels)
    aggregated_stats["operation"] = (
        operations.pop() if len(operations) == 1 else "mixed"
    )

    return aggregated_stats


# --- Summary Printing Function ---


def print_fio_summary(agg_results, directory_path):
    """Prints the aggregated fio summary (mean +/- stdev) for a directory."""
    if not agg_results:
        print(
            f"\n--- No aggregated results to display for directory: {directory_path} ---",
            file=sys.stderr,
        )
        return

    valid_runs = agg_results.get("valid_run_count", 0)
    operation = agg_results.get("operation", "N/A").upper()
    print("\n" + "-" * 65)
    print(f"Fio Aggregated Benchmark Summary ({operation})")
    print(f"Directory Processed : {directory_path}")
    print(f"Log Files Aggregated: {valid_runs}")
    print("-" * 65)

    # Helper to format and print a metric line (ASCII safe)
    def print_metric(label, unit, key, precision=2):
        data = agg_results.get(key)
        label_padded = f"{label:<25}"  # Pad label for alignment
        if data:
            mean = data["mean"]
            stdev = data["stdev"]
            count = data["count"]
            mean_str = f"{mean:.{precision}f}"
            stdev_str = f"{stdev:.{precision}f}"
            # Use +/- instead of the plus-minus symbol
            print(f"{label_padded}: {mean_str} +/- {stdev_str} {unit} (n={count})")
        else:
            print(f"{label_padded}: N/A (n=0)")

    # Print Throughput
    print("Throughput:")
    print_metric("  IOPS", "", "iops", precision=1)
    print_metric("  Bandwidth", "MiB/s", "bandwidth_mibps", precision=1)
    print_metric("  Duration", "s", "duration_sec")

    print("-" * 65)  # Separator

    # Print Latency (Completion Latency)
    print("Completion Latency (clat):")
    print_metric("  Average", "ms", "clat_avg_ms", precision=3)
    print_metric("  Maximum", "ms", "clat_max_ms", precision=3)
    # Print aggregated percentiles
    for p in PERCENTILES_TO_AGGREGATE:
        p_key = f"clat_p{p.replace('.', '_')}_ms"
        p_label = f"  P{p}"
        print_metric(p_label, "ms", p_key, precision=3)

    print("-" * 65)  # Separator

    # Print Resource Usage
    print("Resource Usage:")
    print_metric("  CPU User", "%", "cpu_usr_pct")
    print_metric("  CPU System", "%", "cpu_sys_pct")
    print_metric("  Disk Util", "%", "disk_util_pct")

    print("-" * 65)


# --- Plotting Function ---


def plot_fio_comparison(agg_data1, agg_data2, label1, label2):
    """
    Creates separate comparison plots for Fio Throughput, Latency, and Resources.
    Saves plots as PDF.
    """
    if not PLOT_ENABLED:
        print("\nPlotting disabled as matplotlib/numpy are not installed.")
        return

    print("\nGenerating Fio comparison plots...")

    label1_short = os.path.basename(label1.rstrip("/\\")) or "Dir 1"
    label2_short = os.path.basename(label2.rstrip("/\\")) or "Dir 2"
    operation = agg_data1.get("operation", "Mixed").upper()  # Assume same op for both

    # --- Plot 1: IOPS ---
    metric_iops = "iops"
    print(f" - Plotting '{metric_iops}'...")
    d1_iops = agg_data1.get(metric_iops)
    d2_iops = agg_data2.get(metric_iops)
    if d1_iops and d2_iops:
        metrics_plot = [metric_iops]
        means1 = [d1_iops["mean"]]
        stdevs1 = [d1_iops["stdev"]]
        means2 = [d2_iops["mean"]]
        stdevs2 = [d2_iops["stdev"]]

        x_indices = np.arange(len(metrics_plot))
        bar_width = 0.35

        fig, ax = plt.subplots(figsize=(6, 5))
        rects1 = ax.bar(
            x_indices - bar_width / 2,
            means1,
            bar_width,
            yerr=stdevs1,
            label=label1_short,
            capsize=5,
            alpha=0.8,
        )
        rects2 = ax.bar(
            x_indices + bar_width / 2,
            means2,
            bar_width,
            yerr=stdevs2,
            label=label2_short,
            capsize=5,
            alpha=0.8,
        )

        ax.set_ylabel("IOPS")
        ax.set_title(f"Fio IOPS Comparison ({operation})")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(["IOPS"])
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.bar_label(rects1, padding=3, fmt="%.0f")
        ax.bar_label(rects2, padding=3, fmt="%.0f")
        ax.set_ylim(bottom=0, top=max(means1 + means2) * 1.15)
        fig.tight_layout()

        plot_filename = f"fio_comparison_iops.pdf"
        try:
            plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
            print(f"   Plot saved to '{plot_filename}'")
        except Exception as e:
            err_msg = str(e).encode("ascii", "replace").decode("ascii")
            print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
        plt.close(fig)
    else:
        print(f"   Skipping IOPS plot: Missing data.")

    # --- Plot 2: Bandwidth ---
    metric_bw = "bandwidth_mibps"
    print(f" - Plotting '{metric_bw}'...")
    d1_bw = agg_data1.get(metric_bw)
    d2_bw = agg_data2.get(metric_bw)
    if d1_bw and d2_bw:
        metrics_plot = [metric_bw]
        means1 = [d1_bw["mean"]]
        stdevs1 = [d1_bw["stdev"]]
        means2 = [d2_bw["mean"]]
        stdevs2 = [d2_bw["stdev"]]

        x_indices = np.arange(len(metrics_plot))
        bar_width = 0.35

        fig, ax = plt.subplots(figsize=(6, 5))
        rects1 = ax.bar(
            x_indices - bar_width / 2,
            means1,
            bar_width,
            yerr=stdevs1,
            label=label1_short,
            capsize=5,
            alpha=0.8,
        )
        rects2 = ax.bar(
            x_indices + bar_width / 2,
            means2,
            bar_width,
            yerr=stdevs2,
            label=label2_short,
            capsize=5,
            alpha=0.8,
        )

        ax.set_ylabel("Bandwidth (MiB/s)")
        ax.set_title(f"Fio Bandwidth Comparison ({operation})")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(["Bandwidth"])
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.bar_label(rects1, padding=3, fmt="%.0f")
        ax.bar_label(rects2, padding=3, fmt="%.0f")
        ax.set_ylim(bottom=0, top=max(means1 + means2) * 1.15)
        fig.tight_layout()

        plot_filename = f"plots/fio_comparison_bandwidth.pdf"
        try:
            plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
            print(f"   Plot saved to '{plot_filename}'")
        except Exception as e:
            err_msg = str(e).encode("ascii", "replace").decode("ascii")
            print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
        plt.close(fig)
    else:
        print(f"   Skipping Bandwidth plot: Missing data.")

    # --- Plot 3: Latency ---
    latency_metrics = ["clat_avg_ms"] + [
        f"clat_p{p.replace('.', '_')}_ms" for p in PERCENTILES_TO_AGGREGATE
    ]
    print(f" - Plotting Latency ({', '.join(latency_metrics)})...")
    means1_lat, stdevs1_lat = [], []
    means2_lat, stdevs2_lat = [], []
    plot_latency_labels = []

    for metric in latency_metrics:
        d1_lat = agg_data1.get(metric)
        d2_lat = agg_data2.get(metric)
        if d1_lat and d2_lat:
            means1_lat.append(d1_lat["mean"])
            stdevs1_lat.append(d1_lat["stdev"])
            means2_lat.append(d2_lat["mean"])
            stdevs2_lat.append(d2_lat["stdev"])
            # Make labels nicer (e.g., "Avg", "P99.00")
            label = (
                metric.replace("clat_", "")
                .replace("_ms", "")
                .replace("p", "P")
                .replace("_", ".")
            )
            plot_latency_labels.append(label.title())
        else:
            print(f"   Skipping latency metric '{metric}' for plot: Missing data.")

    if plot_latency_labels:
        x_indices = np.arange(len(plot_latency_labels))
        bar_width = 0.35
        fig, ax = plt.subplots(figsize=(max(6, len(plot_latency_labels) * 1.5), 5))
        rects1 = ax.bar(
            x_indices - bar_width / 2,
            means1_lat,
            bar_width,
            yerr=stdevs1_lat,
            label=label1_short,
            capsize=5,
            alpha=0.8,
        )
        rects2 = ax.bar(
            x_indices + bar_width / 2,
            means2_lat,
            bar_width,
            yerr=stdevs2_lat,
            label=label2_short,
            capsize=5,
            alpha=0.8,
        )

        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"Fio Completion Latency Comparison ({operation})")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(plot_latency_labels)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.bar_label(rects1, padding=3, fmt="%.2f")
        ax.bar_label(rects2, padding=3, fmt="%.2f")
        ax.set_ylim(bottom=0, top=max(means1_lat + means2_lat) * 1.15)
        fig.tight_layout()

        plot_filename = f"fio_comparison_latency.pdf"
        try:
            plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
            print(f"   Plot saved to '{plot_filename}'")
        except Exception as e:
            err_msg = str(e).encode("ascii", "replace").decode("ascii")
            print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
        plt.close(fig)
    else:
        print(f"   Skipping Latency plot: No common latency metrics with data.")

    # --- Plot 4: Resource Usage ---
    resource_metrics = ["cpu_usr_pct", "cpu_sys_pct", "disk_util_pct"]
    print(f" - Plotting Resources ({', '.join(resource_metrics)})...")
    means1_res, stdevs1_res = [], []
    means2_res, stdevs2_res = [], []
    plot_resource_labels = []

    for metric in resource_metrics:
        d1_res = agg_data1.get(metric)
        d2_res = agg_data2.get(metric)
        if d1_res and d2_res:
            means1_res.append(d1_res["mean"])
            stdevs1_res.append(d1_res["stdev"])
            means2_res.append(d2_res["mean"])
            stdevs2_res.append(d2_res["stdev"])
            # Make labels nicer (e.g., "CPU Usr", "Disk Util")
            label = (
                metric.replace("_pct", "")
                .replace("cpu_", "CPU ")
                .replace("disk_", "Disk ")
                .replace("_", " ")
                .title()
            )
            plot_resource_labels.append(label)
        else:
            print(f"   Skipping resource metric '{metric}' for plot: Missing data.")

    if plot_resource_labels:
        x_indices = np.arange(len(plot_resource_labels))
        bar_width = 0.35
        fig, ax = plt.subplots(figsize=(max(6, len(plot_resource_labels) * 1.8), 5))
        rects1 = ax.bar(
            x_indices - bar_width / 2,
            means1_res,
            bar_width,
            yerr=stdevs1_res,
            label=label1_short,
            capsize=5,
            alpha=0.8,
        )
        rects2 = ax.bar(
            x_indices + bar_width / 2,
            means2_res,
            bar_width,
            yerr=stdevs2_res,
            label=label2_short,
            capsize=5,
            alpha=0.8,
        )

        ax.set_ylabel("Utilization (%)")
        ax.set_title(f"Fio Resource Usage Comparison ({operation})")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(plot_resource_labels)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.bar_label(rects1, padding=3, fmt="%.1f")
        ax.bar_label(rects2, padding=3, fmt="%.1f")
        ax.set_ylim(
            bottom=0, top=max(means1_res + means2_res + [10]) * 1.15
        )  # Ensure some space, max of 100 basically
        fig.tight_layout()

        plot_filename = f"fio_comparison_resources.pdf"
        try:
            plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
            print(f"   Plot saved to '{plot_filename}'")
        except Exception as e:
            err_msg = str(e).encode("ascii", "replace").decode("ascii")
            print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
        plt.close(fig)
    else:
        print(f"   Skipping Resource plot: No common resource metrics with data.")


# --- Directory Processing Function ---


def process_directory(directory_path):
    """Finds, parses, and aggregates fio log files in a directory."""
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

    all_results = []
    parse_errors = 0
    files_found_count = len(log_files)

    print(
        f"Found {files_found_count} file(s) matching '{LOG_FILE_PATTERN}'. Parsing..."
    )

    for log_file in log_files:
        metrics = parse_fio_log(log_file)

        if metrics.get("error"):
            if "Could not extract key fio metrics" not in metrics["error"]:
                print(
                    f"\n   Parse Error in '{os.path.basename(log_file)}': {metrics['error']}",
                    file=sys.stderr,
                )
            parse_errors += 1
        elif metrics:
            all_results.append(metrics)

    valid_files_parsed = len(all_results)
    print(f"Successfully parsed data from {valid_files_parsed} file(s).")
    if parse_errors > 0:
        print(
            f"Skipped or encountered errors in {parse_errors} file(s).", file=sys.stderr
        )

    if valid_files_parsed > 0:
        aggregated_data = aggregate_fio_results(all_results)
        return aggregated_data, valid_files_parsed
    else:
        print(f"No valid data collected from directory '{directory_path}'.")
        return None, 0


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Fio benchmark logs from two directories, print summaries, and plot comparisons (ASCII only)."
    )
    parser.add_argument(
        "dir1",
        metavar="DIRECTORY_1",
        type=str,
        help="Path to the first directory containing Fio benchmark *.log files.",
    )
    parser.add_argument(
        "dir2",
        metavar="DIRECTORY_2",
        type=str,
        help="Path to the second directory containing Fio benchmark *.log files.",
    )
    args = parser.parse_args()

    # Process first directory
    agg_data1, count1 = process_directory(args.dir1)

    # Process second directory
    agg_data2, count2 = process_directory(args.dir2)

    # Print summaries if data was aggregated
    if agg_data1:
        print_fio_summary(agg_data1, args.dir1)
    if agg_data2:
        print_fio_summary(agg_data2, args.dir2)

    # Generate comparison plot if both directories yielded data
    if agg_data1 and agg_data2:
        plot_fio_comparison(agg_data1, agg_data2, args.dir1, args.dir2)
    elif PLOT_ENABLED:
        print(
            "\nComparison plots cannot be generated as data from both directories is required.",
            file=sys.stderr,
        )

    print("\nScript finished.")

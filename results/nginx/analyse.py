# Ensure only ASCII characters are used in this script
"""
Processes wrk-style benchmark log files (*.log) from two directories,
prints an aggregated summary for each directory, and generates a
comparison plot saved as a PDF. Uses only ASCII characters.
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
# Metrics to create comparison plots for (only used if plotting enabled)
# Note: The example log doesn't show p99 latency directly.
# Using Avg Latency, Max Latency, and Overall Req/Sec instead.
METRICS_TO_PLOT = [
    "overall_req_sec",
    "latency_avg_ms",
    "latency_max_ms",
    "overall_transfer_mb_sec",
]


# --- Helper Functions for Unit Parsing (ASCII safe) ---


def parse_latency_value(value_str, unit_str):
    """Converts latency string (e.g., '1.33ms') to milliseconds."""
    try:
        value = float(value_str)
        unit = unit_str.lower()
        if unit == "us":
            return value / 1000.0
        elif unit == "s":
            return value * 1000.0
        elif unit == "ms":
            return value
        else:
            # ASCII safe warning
            print(
                f"Warning: Unknown latency unit '{unit_str}'. Assuming ms.",
                file=sys.stderr,
            )
            return value
    except (ValueError, TypeError):
        return None


def parse_req_sec_value(value_str, unit_str):
    """Converts Req/Sec string (e.g., '39.03k') to requests per second."""
    try:
        value = float(value_str)
        unit = unit_str.lower()
        if unit == "k":
            return value * 1000.0
        elif unit == "m":
            return value * 1000000.0
        elif not unit:  # No unit means raw number
            return value
        else:
            # ASCII safe warning
            print(
                f"Warning: Unknown Req/Sec unit '{unit_str}'. Assuming raw number.",
                file=sys.stderr,
            )
            return value
    except (ValueError, TypeError):
        return None


def parse_data_size(value_str, unit_str):
    """Converts data size string (e.g., '1.27GB') to Megabytes (MB)."""
    try:
        value = float(value_str)
        unit = unit_str.upper()
        if unit == "GB":
            return value * 1024.0
        elif unit == "MB":
            return value
        elif unit == "KB":
            return value / 1024.0
        elif unit == "B":
            return value / (1024.0 * 1024.0)
        else:
            # ASCII safe warning
            print(
                f"Warning: Unknown data size unit '{unit_str}'. Assuming MB.",
                file=sys.stderr,
            )
            return value
    except (ValueError, TypeError):
        return None


# --- Log Parsing Function ---


def parse_wrk_log(filepath):
    """Parses a single wrk-style log file to extract key metrics."""
    # Metrics dictionary - using names consistent with previous scripts where applicable
    metrics = {
        "threads": None,
        "connections": None,
        "latency_avg_ms": None,
        "latency_stdev_ms": None,
        "latency_max_ms": None,
        # "latency_stdev_pct": None, # +/- Stdev column, not typically aggregated directly
        "req_sec_avg_thread": None,  # Avg Req/Sec per thread
        "req_sec_stdev_thread": None,
        "req_sec_max_thread": None,
        # "req_sec_stdev_pct": None, # +/- Stdev column
        "total_requests": None,
        "duration_s": None,
        "total_data_read_mb": None,
        "overall_req_sec": None,  # Overall summary Requests/sec
        "overall_transfer_mb_sec": None,  # Overall summary Transfer/sec
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
        # Use utf-8 with replace for robustness against potential encoding issues
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # --- Extracting Metrics using Regular Expressions (ASCII safe patterns) ---

        # Threads and Connections
        match = re.search(r"(\d+)\s+threads and (\d+)\s+connections", content)
        if match:
            metrics["threads"] = int(match.group(1))
            metrics["connections"] = int(match.group(2))

        # Thread Stats - Latency (Avg, Stdev, Max)
        # Latency     1.33ms    0.85ms   22.28ms   92.82%
        match = re.search(
            r"Latency\s+(\d+\.?\d*)(ms|us|s)\s+(\d+\.?\d*)(ms|us|s)\s+(\d+\.?\d*)(ms|us|s)\s+(\d+\.?\d*)%",
            content,
            re.IGNORECASE,
        )
        if match:
            metrics["latency_avg_ms"] = parse_latency_value(
                match.group(1), match.group(2)
            )
            metrics["latency_stdev_ms"] = parse_latency_value(
                match.group(3), match.group(4)
            )
            metrics["latency_max_ms"] = parse_latency_value(
                match.group(5), match.group(6)
            )
            # metrics["latency_stdev_pct"] = float(match.group(7)) # Available if needed

        # Thread Stats - Req/Sec (Avg, Stdev, Max)
        # Req/Sec    39.03k     2.37k    47.74k    71.75%
        match = re.search(
            r"Req/Sec\s+(\d+\.?\d*)(k|m?)\s+(\d+\.?\d*)(k|m?)\s+(\d+\.?\d*)(k|m?)\s+(\d+\.?\d*)%",
            content,
            re.IGNORECASE,
        )
        if match:
            metrics["req_sec_avg_thread"] = parse_req_sec_value(
                match.group(1), match.group(2)
            )
            metrics["req_sec_stdev_thread"] = parse_req_sec_value(
                match.group(3), match.group(4)
            )
            metrics["req_sec_max_thread"] = parse_req_sec_value(
                match.group(5), match.group(6)
            )
            # metrics["req_sec_stdev_pct"] = float(match.group(7)) # Available if needed

        # Summary Line: Requests, Duration, Data Read
        # 4659800 requests in 30.01s, 1.27GB read
        match = re.search(
            r"(\d+)\s+requests in\s+(\d+\.?\d*)s,\s+(\d+\.?\d*)(GB|MB|KB|B)\s+read",
            content,
            re.IGNORECASE,
        )
        if match:
            metrics["total_requests"] = int(match.group(1))
            metrics["duration_s"] = float(match.group(2))
            metrics["total_data_read_mb"] = parse_data_size(
                match.group(3), match.group(4)
            )

        # Overall Requests/sec
        # Requests/sec: 155289.29
        match = re.search(r"Requests/sec:\s+(\d+\.?\d*)", content, re.IGNORECASE)
        if match:
            metrics["overall_req_sec"] = float(match.group(1))

        # Overall Transfer/sec
        # Transfer/sec:     43.24MB
        match = re.search(
            r"Transfer/sec:\s+(\d+\.?\d*)(GB|MB|KB|B)", content, re.IGNORECASE
        )
        if match:
            metrics["overall_transfer_mb_sec"] = parse_data_size(
                match.group(1), match.group(2)
            )

    except Exception as e:
        # Ensure error message uses only ASCII characters if possible
        err_msg = str(e).encode("ascii", "replace").decode("ascii")
        metrics["error"] = f"An error occurred during parsing {filepath}: {err_msg}"

    # --- Basic Validation ---
    if metrics["overall_req_sec"] is None and metrics["latency_avg_ms"] is None:
        if not metrics["error"]:
            metrics["error"] = f"Could not extract key wrk metrics from {filepath}"

    return metrics


# --- Aggregation Function ---


def aggregate_results(all_metrics_data):
    """Calculates mean and standard deviation for metrics across multiple runs."""
    # Simpler structure: metric_name -> list_of_values
    collected_data = collections.defaultdict(list)
    valid_run_count = len(all_metrics_data)

    if valid_run_count == 0:
        return None  # No data to aggregate

    # List of numeric metric keys to aggregate (must match keys in parse_wrk_log)
    keys_to_aggregate = [
        "threads",
        "connections",
        "latency_avg_ms",
        "latency_stdev_ms",
        "latency_max_ms",
        "req_sec_avg_thread",
        "req_sec_stdev_thread",
        "req_sec_max_thread",
        "total_requests",
        "duration_s",
        "total_data_read_mb",
        "overall_req_sec",
        "overall_transfer_mb_sec",
    ]

    # Step 1: Collect all values
    for run_data in all_metrics_data:
        if not run_data:
            continue  # Skip if a file failed parsing

        for key in keys_to_aggregate:
            value = run_data.get(key)
            if value is not None:  # Only aggregate valid numeric values
                collected_data[key].append(value)

    # Step 2: Calculate statistics
    aggregated_stats = {}
    for metric_name, values in collected_data.items():
        count = len(values)
        if count > 0:
            mean = statistics.mean(values)
            # Standard deviation requires at least 2 data points
            stdev = statistics.stdev(values) if count > 1 else 0.0
            aggregated_stats[metric_name] = {
                "mean": mean,
                "stdev": stdev,
                "count": count,
            }

    if not aggregated_stats:  # Check if any stats were actually calculated
        return None

    # Add the count of valid runs used for aggregation
    aggregated_stats["valid_run_count"] = valid_run_count
    return aggregated_stats


# --- Summary Printing Function ---


def print_aggregated_summary(agg_results, directory_path):
    """Prints the aggregated wrk/nginx summary (mean +/- stdev) for a directory."""
    if not agg_results:
        print(
            f"\n--- No aggregated results to display for directory: {directory_path} ---",
            file=sys.stderr,
        )
        return

    valid_runs = agg_results.get("valid_run_count", 0)
    print("\n" + "-" * 65)
    print(f"Aggregated Benchmark Summary")
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

    # Print configuration if consistent
    threads_data = agg_results.get("threads")
    connections_data = agg_results.get("connections")
    if (
        threads_data
        and threads_data["count"] == valid_runs
        and threads_data["stdev"] == 0.0
    ):
        print(f"{'Threads':<25}: {int(threads_data['mean'])}")
    else:
        print_metric("Avg Threads", "", "threads", precision=1)

    if (
        connections_data
        and connections_data["count"] == valid_runs
        and connections_data["stdev"] == 0.0
    ):
        print(f"{'Connections':<25}: {int(connections_data['mean'])}")
    else:
        print_metric("Avg Connections", "", "connections", precision=1)

    print("-" * 65)  # Separator

    # Print Latency Stats (Thread Avg)
    print("Latency Stats (Thread Avg):")
    print_metric("  Avg Latency", "ms", "latency_avg_ms", precision=3)
    print_metric("  Stdev Latency", "ms", "latency_stdev_ms", precision=3)
    print_metric("  Max Latency", "ms", "latency_max_ms", precision=3)

    print("-" * 65)  # Separator

    # Print Req/Sec Stats (Thread Avg)
    print("Req/Sec Stats (Thread Avg):")
    print_metric("  Avg Req/Sec", "req/s", "req_sec_avg_thread", precision=0)
    print_metric("  Stdev Req/Sec", "req/s", "req_sec_stdev_thread", precision=0)
    print_metric("  Max Req/Sec", "req/s", "req_sec_max_thread", precision=0)

    print("-" * 65)  # Separator

    # Print Overall Summary Stats
    print("Overall Summary Stats:")
    print_metric("  Total Requests", "reqs", "total_requests", precision=0)
    print_metric("  Duration", "s", "duration_s")
    print_metric("  Total Data Read", "MB", "total_data_read_mb")
    print_metric("  Overall Requests/sec", "req/s", "overall_req_sec")
    print_metric("  Overall Transfer/sec", "MB/s", "overall_transfer_mb_sec")

    print("-" * 65)


# --- Plotting Function ---

# --- Plotting Function (Modified) ---


def plot_comparison(agg_data1, agg_data2, label1, label2):
    """
    Creates separate comparison plots for RPS, Latency, and Transfer Speed
    between two aggregated datasets. Saves plots as PDF.
    """
    if not PLOT_ENABLED:
        print("\nPlotting disabled as matplotlib/numpy are not installed.")
        return

    print("\nGenerating comparison plots...")

    # Use directory names as labels
    label1_short = os.path.basename(label1.rstrip("/\\")) or "Dir 1"
    label2_short = os.path.basename(label2.rstrip("/\\")) or "Dir 2"

    # --- Plot 1: Requests per Second (Horizontal) ---
    metric_rps = "overall_req_sec"
    print(f" - Plotting '{metric_rps}'...")
    d1_rps_data = agg_data1.get(metric_rps)
    d2_rps_data = agg_data2.get(metric_rps)

    if d1_rps_data and d2_rps_data:
        means1 = [d1_rps_data["mean"]]
        stdevs1 = [d1_rps_data["stdev"]]
        means2 = [d2_rps_data["mean"]]
        stdevs2 = [d2_rps_data["stdev"]]

        y_pos = np.arange(len(means1))  # Only one metric group
        bar_height = 0.35  # Height for horizontal bars

        fig_rps, ax_rps = plt.subplots(figsize=(8, 4))  # Adjust size for horizontal

        # Note: For barh, error bars are passed via xerr
        rects1 = ax_rps.barh(
            y_pos + bar_height / 2,
            means1,
            bar_height,
            xerr=stdevs1,
            label=label1_short,
            capsize=4,
            alpha=0.8,
        )
        rects2 = ax_rps.barh(
            y_pos - bar_height / 2,
            means2,
            bar_height,
            xerr=stdevs2,
            label=label2_short,
            capsize=4,
            alpha=0.8,
        )

        ax_rps.set_xlabel("Requests per Second (Overall)")
        ax_rps.set_title("Comparison of Overall Requests per Second")
        ax_rps.set_yticks(y_pos)
        ax_rps.set_yticklabels(["Req/Sec"])  # Label the single category
        ax_rps.invert_yaxis()  # labels read top-to-bottom
        ax_rps.legend(loc="lower right")
        ax_rps.grid(axis="x", linestyle="--", alpha=0.6)

        # Add value labels on the right of the bars
        ax_rps.bar_label(rects1, padding=3, fmt="%.0f")
        ax_rps.bar_label(rects2, padding=3, fmt="%.0f")

        # Adjust x-axis limits if needed to make space for labels
        ax_rps.set_xlim(right=max(means1 + means2) * 1.15)  # Add 15% padding

        fig_rps.tight_layout()

        plot_filename = "comparison_requests_per_second.pdf"
        try:
            plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
            print(f"   Plot saved to '{plot_filename}'")
        except Exception as e:
            err_msg = str(e).encode("ascii", "replace").decode("ascii")
            print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
        plt.close(fig_rps)
    else:
        print(
            f"   Skipping plot for '{metric_rps}': Missing data in one or both directories."
        )

    # --- Plot 2: Latency Figures (Vertical Grouped) ---
    latency_metrics_to_plot = [
        "latency_avg_ms",
        "latency_max_ms",
    ]  # Choose latency metrics
    print(f" - Plotting Latency ({', '.join(latency_metrics_to_plot)})...")

    means1_lat, stdevs1_lat = [], []
    means2_lat, stdevs2_lat = [], []
    plot_latency_labels = []  # Latency metric names for x-axis

    for metric in latency_metrics_to_plot:
        d1_lat_data = agg_data1.get(metric)
        d2_lat_data = agg_data2.get(metric)

        if d1_lat_data and d2_lat_data:
            means1_lat.append(d1_lat_data["mean"])
            stdevs1_lat.append(d1_lat_data["stdev"])
            means2_lat.append(d2_lat_data["mean"])
            stdevs2_lat.append(d2_lat_data["stdev"])
            # Make labels nicer (e.g., "Avg Latency", "Max Latency")
            plot_latency_labels.append(
                metric.replace("latency_", "")
                .replace("_ms", "")
                .replace("_", " ")
                .title()
            )
        else:
            print(f"   Skipping latency metric '{metric}' for plot: Missing data.")

    if plot_latency_labels:  # Check if we have any latency data to plot
        x_indices_lat = np.arange(len(plot_latency_labels))
        bar_width = 0.35

        fig_lat, ax_lat = plt.subplots(
            figsize=(max(6, len(plot_latency_labels) * 2), 6)
        )  # Adjust width

        rects1_lat = ax_lat.bar(
            x_indices_lat - bar_width / 2,
            means1_lat,
            bar_width,
            yerr=stdevs1_lat,
            label=label1_short,
            capsize=5,
            alpha=0.8,
        )
        rects2_lat = ax_lat.bar(
            x_indices_lat + bar_width / 2,
            means2_lat,
            bar_width,
            yerr=stdevs2_lat,
            label=label2_short,
            capsize=5,
            alpha=0.8,
        )

        ax_lat.set_ylabel("Latency (ms)")
        ax_lat.set_title("Comparison of Latency Metrics")
        ax_lat.set_xticks(x_indices_lat)
        ax_lat.set_xticklabels(plot_latency_labels)
        ax_lat.legend()
        ax_lat.grid(axis="y", linestyle="--", alpha=0.6)

        # Add bar labels
        ax_lat.bar_label(rects1_lat, padding=3, fmt="%.2f")
        ax_lat.bar_label(rects2_lat, padding=3, fmt="%.2f")

        # Adjust y-axis limits if needed
        ax_lat.set_ylim(bottom=0, top=max(means1_lat + means2_lat) * 1.15)

        fig_lat.tight_layout()

        plot_filename = "comparison_latency.pdf"
        try:
            plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
            print(f"   Plot saved to '{plot_filename}'")
        except Exception as e:
            err_msg = str(e).encode("ascii", "replace").decode("ascii")
            print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
        plt.close(fig_lat)
    else:
        print(f"   Skipping latency plot: No common latency metrics with data found.")

    # --- Plot 3: Transfer Speed (Horizontal) ---
    metric_transfer = "overall_transfer_mb_sec"
    print(f" - Plotting '{metric_transfer}'...")
    d1_transfer_data = agg_data1.get(metric_transfer)
    d2_transfer_data = agg_data2.get(metric_transfer)

    if d1_transfer_data and d2_transfer_data:
        means1 = [d1_transfer_data["mean"]]
        stdevs1 = [d1_transfer_data["stdev"]]
        means2 = [d2_transfer_data["mean"]]
        stdevs2 = [d2_transfer_data["stdev"]]

        y_pos = np.arange(len(means1))
        bar_height = 0.35

        fig_transfer, ax_transfer = plt.subplots(figsize=(8, 4))

        rects1 = ax_transfer.barh(
            y_pos + bar_height / 2,
            means1,
            bar_height,
            xerr=stdevs1,
            label=label1_short,
            capsize=4,
            alpha=0.8,
        )
        rects2 = ax_transfer.barh(
            y_pos - bar_height / 2,
            means2,
            bar_height,
            xerr=stdevs2,
            label=label2_short,
            capsize=4,
            alpha=0.8,
        )

        ax_transfer.set_xlabel("Transfer Speed (MB/s)")
        ax_transfer.set_title("Comparison of Overall Transfer Speed")
        ax_transfer.set_yticks(y_pos)
        ax_transfer.set_yticklabels(["Transfer"])
        ax_transfer.invert_yaxis()
        ax_transfer.legend(loc="lower right")
        ax_transfer.grid(axis="x", linestyle="--", alpha=0.6)

        ax_transfer.bar_label(rects1, padding=3, fmt="%.1f")
        ax_transfer.bar_label(rects2, padding=3, fmt="%.1f")
        ax_transfer.set_xlim(right=max(means1 + means2) * 1.15)

        fig_transfer.tight_layout()

        plot_filename = "comparison_transfer_speed.pdf"
        try:
            plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
            print(f"   Plot saved to '{plot_filename}'")
        except Exception as e:
            err_msg = str(e).encode("ascii", "replace").decode("ascii")
            print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
        plt.close(fig_transfer)
    else:
        print(
            f"   Skipping plot for '{metric_transfer}': Missing data in one or both directories."
        )


# --- Directory Processing Function ---


def process_directory(directory_path):
    """Finds, parses, and aggregates log files in a directory."""
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
        metrics = parse_wrk_log(log_file)

        if metrics.get("error"):
            # Print specific error only if it's not the 'missing key metrics' warning
            if "Could not extract key wrk metrics" not in metrics["error"]:
                print(
                    f"\n   Parse Error in '{os.path.basename(log_file)}': {metrics['error']}",
                    file=sys.stderr,
                )
            # else: print(" Skipped (Missing key metrics).") # Optional verbose skip message
            parse_errors += 1
        elif metrics:  # Ensure metrics dictionary is not None/empty
            all_results.append(metrics)

    valid_files_parsed = len(all_results)
    print(f"Successfully parsed data from {valid_files_parsed} file(s).")
    if parse_errors > 0:
        print(
            f"Skipped or encountered errors in {parse_errors} file(s).", file=sys.stderr
        )

    if valid_files_parsed > 0:
        aggregated_data = aggregate_results(all_results)
        # Pass back the count of files *used* in aggregation
        return aggregated_data, valid_files_parsed
    else:
        print(f"No valid data collected from directory '{directory_path}'.")
        return None, 0


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare wrk/nginx benchmark logs from two directories, print summaries, and plot comparisons (ASCII only)."
    )
    parser.add_argument(
        "dir1",
        metavar="DIRECTORY_1",
        type=str,
        help="Path to the first directory containing benchmark *.log files.",
    )
    parser.add_argument(
        "dir2",
        metavar="DIRECTORY_2",
        type=str,
        help="Path to the second directory containing benchmark *.log files.",
    )
    args = parser.parse_args()

    # Process first directory
    agg_data1, count1 = process_directory(args.dir1)

    # Process second directory
    agg_data2, count2 = process_directory(args.dir2)

    # Print summaries if data was aggregated
    if agg_data1:
        # Pass count of files aggregated, not total found
        print_aggregated_summary(agg_data1, args.dir1)
    if agg_data2:
        print_aggregated_summary(agg_data2, args.dir2)

    # Generate comparison plot if both directories yielded data
    if agg_data1 and agg_data2:
        plot_comparison(agg_data1, agg_data2, args.dir1, args.dir2)
    elif PLOT_ENABLED:
        print(
            "\nComparison plot cannot be generated as data from both directories is required.",
            file=sys.stderr,
        )

    print("\nScript finished.")

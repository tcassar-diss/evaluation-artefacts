# -*- coding: utf-8 -*-
"""
Parses multiple wrk output log files found in a specified directory,
calculates mean and standard deviation for key metrics, and prints an
aggregated summary.
"""

import argparse
import glob
import math
import os
import re
import statistics
import sys

# --- Configuration ---
# Log file pattern to search for within the directory
LOG_FILE_PATTERN = "*.log"

# --- Helper Functions for Unit Parsing ---


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
            print(
                f"Warning: Unknown data size unit '{unit_str}'. Assuming MB.",
                file=sys.stderr,
            )
            return value
    except (ValueError, TypeError):
        return None


# --- Parsing Function ---


def parse_wrk_log(filepath):
    """Parses a single wrk log file to extract key metrics."""
    metrics = {
        "threads": None,
        "connections": None,
        "latency_avg_ms": None,
        "latency_stdev_ms": None,
        "latency_max_ms": None,
        "latency_stdev_pct": None,  # The +/- Stdev column
        "req_sec_avg": None,
        "req_sec_stdev": None,
        "req_sec_max": None,
        "req_sec_stdev_pct": None,  # The +/- Stdev column
        "total_requests": None,
        "duration_s": None,
        "total_data_read_mb": None,
        "overall_req_sec": None,
        "overall_transfer_mb_sec": None,
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
        with open(filepath, "r") as f:
            content = f.read()

        # --- Extracting Metrics using Regular Expressions ---

        # Threads and Connections
        match = re.search(r"(\d+)\s+threads and (\d+)\s+connections", content)
        if match:
            metrics["threads"] = int(match.group(1))
            metrics["connections"] = int(match.group(2))

        # Thread Stats - Latency
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
            metrics["latency_stdev_pct"] = float(match.group(7))

        # Thread Stats - Req/Sec
        # Req/Sec    39.03k     2.37k    47.74k    71.75%
        match = re.search(
            r"Req/Sec\s+(\d+\.?\d*)(k|m?)\s+(\d+\.?\d*)(k|m?)\s+(\d+\.?\d*)(k|m?)\s+(\d+\.?\d*)%",
            content,
            re.IGNORECASE,
        )
        if match:
            metrics["req_sec_avg"] = parse_req_sec_value(match.group(1), match.group(2))
            metrics["req_sec_stdev"] = parse_req_sec_value(
                match.group(3), match.group(4)
            )
            metrics["req_sec_max"] = parse_req_sec_value(match.group(5), match.group(6))
            metrics["req_sec_stdev_pct"] = float(match.group(7))

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
        metrics["error"] = f"An error occurred during parsing {filepath}: {e}"

    # --- Basic Validation ---
    # Check if essential metrics were found
    if metrics["overall_req_sec"] is None and metrics["latency_avg_ms"] is None:
        # Avoid adding empty results if parsing failed badly
        if not metrics["error"]:  # Add error if not already set
            metrics["error"] = f"Could not extract key wrk metrics from {filepath}"

    return metrics


# --- Aggregation Function ---


def calculate_aggregates(all_metrics_data):
    """Calculates mean and standard deviation for metrics across multiple runs."""
    aggregated_results = {}
    valid_run_count = len(all_metrics_data)

    if valid_run_count == 0:
        return {"error": "No valid metric data found to aggregate."}

    # List of metric keys to aggregate
    keys_to_aggregate = [
        "threads",
        "connections",
        "latency_avg_ms",
        "latency_stdev_ms",
        "latency_max_ms",
        "latency_stdev_pct",
        "req_sec_avg",
        "req_sec_stdev",
        "req_sec_max",
        "req_sec_stdev_pct",
        "total_requests",
        "duration_s",
        "total_data_read_mb",
        "overall_req_sec",
        "overall_transfer_mb_sec",
    ]

    for key in keys_to_aggregate:
        # Collect non-None values for the current key
        values = [
            run_data.get(key)
            for run_data in all_metrics_data
            if run_data.get(key) is not None
        ]
        count = len(values)

        if count > 0:
            mean = statistics.mean(values)
            # Standard deviation requires at least 2 data points
            stdev = statistics.stdev(values) if count > 1 else 0.0
            aggregated_results[key] = {"mean": mean, "stdev": stdev, "count": count}
        else:
            # Explicitly store that no data was found for this metric
            aggregated_results[key] = {"mean": None, "stdev": None, "count": 0}

    aggregated_results["valid_run_count"] = valid_run_count
    return aggregated_results


# --- Summary Printing Function ---


def print_aggregated_summary(agg_results, directory_path):
    """Prints the aggregated wrk summary (mean +/- stdev) for a directory."""
    if agg_results.get("error"):
        print(agg_results["error"], file=sys.stderr)
        return

    valid_runs = agg_results.get("valid_run_count", 0)
    print("-" * 65)
    print(f"WRK Aggregated Performance Summary")
    print(f"Directory Processed: {directory_path}")
    print(f"Valid Log Files Aggregated: {valid_runs}")
    print("-" * 65)

    # Helper to format and print a metric line
    def print_metric(label, unit, key, precision=2):
        data = agg_results.get(key)
        label_padded = f"{label:<25}"  # Pad label for alignment
        if data and data["mean"] is not None:
            mean = data["mean"]
            stdev = data["stdev"]
            count = data["count"]
            # Format mean and stdev with specified precision
            mean_str = f"{mean:.{precision}f}"
            stdev_str = f"{stdev:.{precision}f}"
            print(f"{label_padded}: {mean_str} ± {stdev_str} {unit} (n={count})")
        else:
            # Indicate if no data was found across runs for this metric
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
        print_metric(
            "Avg Threads", "", "threads", precision=1
        )  # Show avg if inconsistent

    if (
        connections_data
        and connections_data["count"] == valid_runs
        and connections_data["stdev"] == 0.0
    ):
        print(f"{'Connections':<25}: {int(connections_data['mean'])}")
    else:
        print_metric("Avg Connections", "", "connections", precision=1)

    print("-" * 65)  # Separator

    # Print Latency Stats
    print("Latency Stats:")
    print_metric("  Avg Latency", "ms", "latency_avg_ms")
    print_metric("  Stdev Latency", "ms", "latency_stdev_ms")
    print_metric("  Max Latency", "ms", "latency_max_ms")
    print_metric("  Avg Latency Stdev %", "%", "latency_stdev_pct")

    print("-" * 65)  # Separator

    # Print Req/Sec Stats
    print("Req/Sec Stats (Thread Avg):")
    print_metric("  Avg Req/Sec", "req/s", "req_sec_avg", precision=0)
    print_metric("  Stdev Req/Sec", "req/s", "req_sec_stdev", precision=0)
    print_metric("  Max Req/Sec", "req/s", "req_sec_max", precision=0)
    print_metric("  Avg Req/Sec Stdev %", "%", "req_sec_stdev_pct")

    print("-" * 65)  # Separator

    # Print Overall Summary Stats
    print("Overall Summary Stats:")
    print_metric("  Total Requests", "reqs", "total_requests", precision=0)
    print_metric("  Duration", "s", "duration_s")
    print_metric("  Total Data Read", "MB", "total_data_read_mb")
    print_metric("  Overall Requests/sec", "req/s", "overall_req_sec")
    print_metric("  Overall Transfer/sec", "MB/s", "overall_transfer_mb_sec")

    print("-" * 65)


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Aggregate wrk statistics from {LOG_FILE_PATTERN} files in a directory."
    )
    parser.add_argument(
        "log_directory",
        metavar="DIRECTORY",
        type=str,
        help=f"Path to the directory containing wrk {LOG_FILE_PATTERN} files.",
    )
    args = parser.parse_args()

    target_directory = args.log_directory

    if not os.path.isdir(target_directory):
        print(
            f"Error: Provided path '{target_directory}' is not a valid directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    search_pattern = os.path.join(target_directory, LOG_FILE_PATTERN)
    log_files_to_process = glob.glob(search_pattern)

    if not log_files_to_process:
        print(
            f"Error: No files matching pattern '{LOG_FILE_PATTERN}' found in directory '{target_directory}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    all_results = []
    parse_errors = 0
    files_processed_count = len(log_files_to_process)

    print(
        f"Found {files_processed_count} file(s) matching '{LOG_FILE_PATTERN}' in '{target_directory}'. Processing..."
    )

    for log_file in log_files_to_process:
        print(f" - Parsing {os.path.basename(log_file)}...", end="", flush=True)
        metrics = parse_wrk_log(log_file)

        if metrics.get("error"):
            # Print error only if it wasn't the 'key metrics missing' warning added at the end of parsing
            if "Could not extract key wrk metrics" not in metrics["error"]:
                print(f"\n   Parse Error: {metrics['error']}", file=sys.stderr)
            else:
                # Treat missing key metrics as a skippable file, not necessarily a hard error
                print(" Skipped (Missing key metrics).")
            parse_errors += 1
        else:
            all_results.append(metrics)
            print(" Done.")

    if len(all_results) > 0:
        aggregated_data = calculate_aggregates(all_results)
        print_aggregated_summary(aggregated_data, target_directory)
    else:
        print(
            "\nNo valid data collected from log files. Cannot generate summary.",
            file=sys.stderr,
        )

    if parse_errors > 0:
        skipped_count = parse_errors
        print(
            f"\nSkipped or encountered errors in {skipped_count} out of {files_processed_count} files.",
            file=sys.stderr,
        )

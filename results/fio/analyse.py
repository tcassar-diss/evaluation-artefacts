# -*- coding: utf-8 -*-
"""
Parses multiple fio output files found in a specified directory,
calculates mean and standard deviation for key metrics, and prints an
aggregated summary.
"""

import argparse  # Import argparse for command-line arguments
import glob  # Import glob for finding files
import os
import re
import statistics
import sys

# --- Configuration ---
# Specific percentiles to aggregate (remains configurable)
PERCENTILES_TO_AGGREGATE = ["95.00", "99.00", "99.90"]
# Log file pattern to search for within the directory
LOG_FILE_PATTERN = "*.log"

# --- Functions ---


# parse_fio_output function remains unchanged from the previous version
def parse_fio_output(filepath):
    """Parses a single fio output file to extract key metrics."""
    metrics = {
        "bandwidth_mibps": None,
        "iops": None,
        "clat_avg_us": None,
        "clat_percentiles_us": {},
        "cpu_usr_pct": None,
        "cpu_sys_pct": None,
        "disk_util_pct": None,
        "total_data_gib": None,
        "duration_sec": None,
        "jobs": None,
        "error": None,
        "filepath": filepath,  # Store filepath for reference
    }

    if not os.path.exists(filepath):
        metrics["error"] = f"Error: File not found at {filepath}"
        return metrics
    # Prevent trying to parse directories if pattern is too broad
    if not os.path.isfile(filepath):
        metrics["error"] = f"Error: Path is not a file {filepath}"
        return metrics

    try:
        with open(filepath, "r") as f:
            content = f.read()

        # --- Extracting Metrics using Regular Expressions ---
        # (Regex patterns are the same as before)

        # Jobs
        match = re.search(r"Jobs: (\d+)", content)
        if match:
            metrics["jobs"] = int(match.group(1))

        # Overall Write Summary (BW, IOPS, Data, Duration)
        match = re.search(
            r"write: IOPS=([\d.]+), BW=([\d.]+)MiB/s .*\((\d+\.?\d*)GiB/(\d+)msec\)",
            content,
        )
        if match:
            metrics["iops"] = float(match.group(1))
            metrics["bandwidth_mibps"] = float(match.group(2))
            metrics["total_data_gib"] = float(match.group(3))
            metrics["duration_sec"] = int(match.group(4)) / 1000.0
        else:
            # Fallback for Run status group
            match = re.search(
                r"WRITE: bw=([\d.]+)MiB/s .* io=([\d.]+)GiB .* run=(\d+)-(\d+)msec",
                content,
            )
            if match:
                metrics["bandwidth_mibps"] = float(match.group(1))
                metrics["total_data_gib"] = float(match.group(2))
                metrics["duration_sec"] = int(match.group(3)) / 1000.0
                # Try finding avg iops separately if needed
                match_iops_alt = re.search(
                    r"iops\s+:\s+min=\s*\d+,\s+max=\s*\d+,\s+avg=([\d.]+),", content
                )
                if match_iops_alt:
                    metrics["iops"] = float(match_iops_alt.group(1))

        # Average Completion Latency (clat avg)
        match = re.search(r"clat \(usec\):.*avg=([\d.]+),", content)
        if match:
            metrics["clat_avg_us"] = float(match.group(1))

        # Completion Latency Percentiles
        percentile_matches = re.findall(r"\|\s*([\d\.]+)th=\[(\d+)\],?", content)
        for p_match in percentile_matches:
            percentile_key = f"{float(p_match[0]):.2f}"
            metrics["clat_percentiles_us"][percentile_key] = int(p_match[1])

        # CPU Usage
        match = re.search(r"cpu\s+:\s+usr=([\d\.]+)%,\s+sys=([\d\.]+)%,", content)
        if match:
            metrics["cpu_usr_pct"] = float(match.group(1))
            metrics["cpu_sys_pct"] = float(match.group(2))

        # Disk Utilization
        util_matches = re.findall(r"util=([\d\.]+)%", content)
        if util_matches:
            metrics["disk_util_pct"] = float(util_matches[-1])

    except Exception as e:
        metrics["error"] = f"An error occurred during parsing {filepath}: {e}"

    return metrics


# calculate_aggregates function remains unchanged
def calculate_aggregates(all_metrics_data):
    """Calculates mean and standard deviation for metrics across multiple runs."""
    aggregated_results = {}
    valid_run_count = len(all_metrics_data)

    if valid_run_count == 0:
        return {"error": "No valid metric data found to aggregate."}

    keys_to_aggregate = [
        "bandwidth_mibps",
        "iops",
        "clat_avg_us",
        "cpu_usr_pct",
        "cpu_sys_pct",
        "disk_util_pct",
        "total_data_gib",
        "duration_sec",
        "jobs",
    ]

    for key in keys_to_aggregate:
        values = [
            run_data.get(key)
            for run_data in all_metrics_data
            if run_data.get(key) is not None
        ]
        count = len(values)
        if count > 0:
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if count > 1 else 0.0
            aggregated_results[key] = {"mean": mean, "stdev": stdev, "count": count}
        else:
            aggregated_results[key] = {"mean": None, "stdev": None, "count": 0}

    aggregated_results["clat_percentiles_us"] = {}
    for p_key in PERCENTILES_TO_AGGREGATE:
        p_values = []
        for run_data in all_metrics_data:
            percentile_value = run_data.get("clat_percentiles_us", {}).get(p_key)
            if percentile_value is not None:
                p_values.append(percentile_value)
        p_count = len(p_values)
        if p_count > 0:
            p_mean = statistics.mean(p_values)
            p_stdev = statistics.stdev(p_values) if p_count > 1 else 0.0
            aggregated_results["clat_percentiles_us"][p_key] = {
                "mean": p_mean,
                "stdev": p_stdev,
                "count": p_count,
            }
        else:
            aggregated_results["clat_percentiles_us"][p_key] = {
                "mean": None,
                "stdev": None,
                "count": 0,
            }

    aggregated_results["valid_run_count"] = valid_run_count
    return aggregated_results


# print_aggregated_summary is slightly modified to show the directory
def print_aggregated_summary(agg_results, directory_path):
    """Prints the aggregated summary (mean +/- stdev) for a directory."""
    if agg_results.get("error"):
        print(agg_results["error"], file=sys.stderr)
        return

    valid_runs = agg_results.get("valid_run_count", 0)
    print("-" * 60)
    print(f"FIO Aggregated Performance Summary")
    print(f"Directory Processed: {directory_path}")
    print(f"Valid Log Files Aggregated: {valid_runs}")
    print("-" * 60)

    def print_metric(label, unit, key, is_ms_also=False):
        data = agg_results.get(key)
        if data and data["mean"] is not None:
            mean = data["mean"]
            stdev = data["stdev"]
            count = data["count"]
            base_str = f"{label:<18}: {mean:.2f} ± {stdev:.2f} {unit} (n={count})"
            if is_ms_also and unit == "µs":
                mean_ms = mean / 1000.0
                stdev_ms = stdev / 1000.0
                base_str += f"  [{mean_ms:.2f} ± {stdev_ms:.2f} ms]"
            print(base_str)
        else:
            print(f"{label:<18}: N/A")

    jobs_data = agg_results.get("jobs")
    if jobs_data and jobs_data["count"] == valid_runs and jobs_data["stdev"] == 0.0:
        print(f"{'Jobs':<18}: {int(jobs_data['mean'])}")
    else:
        print_metric("Avg Jobs", "", "jobs")

    print_metric("Avg Duration", "s", "duration_sec")
    print_metric("Avg Total Data", "GiB", "total_data_gib")
    print_metric("Avg Bandwidth", "MiB/s", "bandwidth_mibps")
    print_metric("Avg IOPS", "", "iops")
    print_metric("Avg CLat", "µs", "clat_avg_us", is_ms_also=True)

    print("CLat Percentiles:")
    percentile_data = agg_results.get("clat_percentiles_us", {})
    for p_key in PERCENTILES_TO_AGGREGATE:
        p_agg = percentile_data.get(p_key)
        label = f"  - {p_key}th %-ile"
        if p_agg and p_agg["mean"] is not None:
            mean_us = p_agg["mean"]
            stdev_us = p_agg["stdev"]
            count = p_agg["count"]
            mean_ms = mean_us / 1000.0
            stdev_ms = stdev_us / 1000.0
            print(
                f"{label:<18}: {mean_us:.0f} ± {stdev_us:.0f} µs (n={count})  [{mean_ms:.2f} ± {stdev_ms:.2f} ms]"
            )
        else:
            print(f"{label:<18}: N/A")

    print_metric("Avg Disk Util", "%", "disk_util_pct")

    cpu_usr = agg_results.get("cpu_usr_pct")
    cpu_sys = agg_results.get("cpu_sys_pct")
    if (
        cpu_usr
        and cpu_usr["mean"] is not None
        and cpu_sys
        and cpu_sys["mean"] is not None
    ):
        print(
            f"{'Avg CPU Usage':<18}: {cpu_usr['mean']:.2f} ± {cpu_usr['stdev']:.2f}% usr / {cpu_sys['mean']:.2f} ± {cpu_sys['stdev']:.2f}% sys (n={min(cpu_usr['count'], cpu_sys['count'])})"
        )
    elif cpu_usr and cpu_usr["mean"] is not None:
        print_metric("Avg CPU User", "%", "cpu_usr_pct")
    elif cpu_sys and cpu_sys["mean"] is not None:
        print_metric("Avg CPU System", "%", "cpu_sys_pct")
    else:
        print(f"{'Avg CPU Usage':<18}: N/A")

    print("-" * 60)


# --- Main Execution ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description=f"Aggregate fio statistics from *.log files in a directory."
    )
    parser.add_argument(
        "log_directory",
        metavar="DIRECTORY",
        type=str,
        help="Path to the directory containing fio *.log files.",
    )
    args = parser.parse_args()

    target_directory = args.log_directory

    # --- Validate Directory ---
    if not os.path.isdir(target_directory):
        print(
            f"Error: Provided path '{target_directory}' is not a valid directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Find Log Files ---
    search_pattern = os.path.join(target_directory, LOG_FILE_PATTERN)
    log_files_to_process = glob.glob(search_pattern)

    if not log_files_to_process:
        print(
            f"Error: No files matching pattern '{LOG_FILE_PATTERN}' found in directory '{target_directory}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Processing ---
    all_results = []
    parse_errors = 0
    files_processed_count = len(log_files_to_process)

    print(
        f"Found {files_processed_count} file(s) matching '{LOG_FILE_PATTERN}' in '{target_directory}'. Processing..."
    )

    for log_file in log_files_to_process:
        # Print progress for potentially many files
        print(f" - Parsing {os.path.basename(log_file)}...", end="", flush=True)
        metrics = parse_fio_output(log_file)
        if metrics.get("error"):
            print(f"\n   {metrics['error']}", file=sys.stderr)  # Newline before error
            parse_errors += 1
        elif (
            metrics.get("bandwidth_mibps") is not None
            or metrics.get("iops") is not None
        ):
            all_results.append(metrics)
            print(" Done.")  # Indicate success
        else:
            print(
                f"\n   Warning: Could not extract key performance data from {os.path.basename(log_file)}. Skipping aggregation for this file.",
                file=sys.stderr,
            )
            parse_errors += 1

    # --- Aggregation and Output ---
    if len(all_results) > 0:
        aggregated_data = calculate_aggregates(all_results)
        # Pass the directory path to the summary function
        print_aggregated_summary(aggregated_data, target_directory)
    else:
        print(
            "\nNo valid data collected from log files. Cannot generate summary.",
            file=sys.stderr,
        )

    if parse_errors > 0:
        failed_count = parse_errors
        print(
            f"\nEncountered errors or missing data in {failed_count} out of {files_processed_count} files.",
            file=sys.stderr,
        )

# Ensure only ASCII characters are used in this script
"""
Processes wrk-style benchmark log files (*.log) from two directories,
prints a comparison summary including percentage differences (Dir2 vs Dir1)
to stdout and ./summary.txt, generates comparison plots for absolute values,
and a plot for percentage differences, all saved as PDF files.
Uses only ASCII characters.
"""

import argparse
import collections
import glob
import os
import re
import statistics
import sys
import math # Needed for checking isnan/isinf

# Attempt to import plotting libraries, proceed without plotting if missing
try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOT_ENABLED = True
except ImportError:
    PLOT_ENABLED = False
    # Ensure print statements here are ASCII
    print("Warning: matplotlib or numpy not found. Plotting will be disabled.", file=sys.stderr)
    print("Install using: pip install matplotlib numpy", file=sys.stderr)


# --- Configuration ---
# Log file pattern to search for within the directory
LOG_FILE_PATTERN = "*.log"
# Key metrics to extract, aggregate, and potentially plot
# Adding more keys here allows them to be included in the text summary
METRIC_KEYS = [
    "threads", "connections", "latency_avg_ms", "latency_stdev_ms",
    "latency_max_ms", "req_sec_avg_thread", "req_sec_stdev_thread",
    "req_sec_max_thread", "total_requests", "duration_s",
    "total_data_read_mb", "overall_req_sec", "overall_transfer_mb_sec"
]
# Metrics to create specific comparison plots for (absolute values & % diff)
METRICS_TO_PLOT = [
    "overall_req_sec",
    "latency_avg_ms",
    "latency_max_ms",
    "overall_transfer_mb_sec",
]
# Output file for the text summary
SUMMARY_FILENAME = "./summary.txt"
# Threshold for baseline mean close to zero to avoid division issues
ZERO_THRESHOLD = 1e-9


# --- Helper Functions for Unit Parsing (ASCII safe) ---
# (These functions remain the same as the provided script)
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
            print(f"Warning: Unknown latency unit '{unit_str}'. Assuming ms.", file=sys.stderr)
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
        elif not unit: # No unit means raw number
            return value
        else:
            print(f"Warning: Unknown Req/Sec unit '{unit_str}'. Assuming raw number.", file=sys.stderr)
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
            print(f"Warning: Unknown data size unit '{unit_str}'. Assuming MB.", file=sys.stderr)
            return value
    except (ValueError, TypeError):
        return None


# --- Log Parsing Function ---
# (parse_wrk_log remains the same as the provided script)
def parse_wrk_log(filepath):
    """Parses a single wrk-style log file to extract key metrics."""
    metrics = {
        "threads": None, "connections": None, "latency_avg_ms": None,
        "latency_stdev_ms": None, "latency_max_ms": None,
        "req_sec_avg_thread": None, "req_sec_stdev_thread": None,
        "req_sec_max_thread": None, "total_requests": None, "duration_s": None,
        "total_data_read_mb": None, "overall_req_sec": None,
        "overall_transfer_mb_sec": None, "error": None, "filepath": filepath,
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
        # Threads and Connections
        match = re.search(r"(\d+)\s+threads and (\d+)\s+connections", content)
        if match:
            metrics["threads"] = int(match.group(1))
            metrics["connections"] = int(match.group(2))
        # Latency
        match = re.search(r"Latency\s+(\d+\.?\d*)(ms|us|s)\s+(\d+\.?\d*)(ms|us|s)\s+(\d+\.?\d*)(ms|us|s)\s+(\d+\.?\d*)%", content, re.IGNORECASE)
        if match:
            metrics["latency_avg_ms"] = parse_latency_value(match.group(1), match.group(2))
            metrics["latency_stdev_ms"] = parse_latency_value(match.group(3), match.group(4))
            metrics["latency_max_ms"] = parse_latency_value(match.group(5), match.group(6))
        # Req/Sec (Thread)
        match = re.search(r"Req/Sec\s+(\d+\.?\d*)(k|m?)\s+(\d+\.?\d*)(k|m?)\s+(\d+\.?\d*)(k|m?)\s+(\d+\.?\d*)%", content, re.IGNORECASE)
        if match:
            metrics["req_sec_avg_thread"] = parse_req_sec_value(match.group(1), match.group(2))
            metrics["req_sec_stdev_thread"] = parse_req_sec_value(match.group(3), match.group(4))
            metrics["req_sec_max_thread"] = parse_req_sec_value(match.group(5), match.group(6))
        # Summary Line
        match = re.search(r"(\d+)\s+requests in\s+(\d+\.?\d*)s,\s+(\d+\.?\d*)(GB|MB|KB|B)\s+read", content, re.IGNORECASE)
        if match:
            metrics["total_requests"] = int(match.group(1))
            metrics["duration_s"] = float(match.group(2))
            metrics["total_data_read_mb"] = parse_data_size(match.group(3), match.group(4))
        # Overall Req/Sec
        match = re.search(r"Requests/sec:\s+(\d+\.?\d*)", content, re.IGNORECASE)
        if match:
            metrics["overall_req_sec"] = float(match.group(1))
        # Overall Transfer/Sec
        match = re.search(r"Transfer/sec:\s+(\d+\.?\d*)(GB|MB|KB|B)", content, re.IGNORECASE)
        if match:
            metrics["overall_transfer_mb_sec"] = parse_data_size(match.group(1), match.group(2))
    except Exception as e:
        err_msg = str(e).encode("ascii", "replace").decode("ascii")
        metrics["error"] = f"An error occurred during parsing {filepath}: {err_msg}"
    # Basic Validation
    if metrics["overall_req_sec"] is None and metrics["latency_avg_ms"] is None:
        if not metrics["error"]:
            metrics["error"] = f"Could not extract key wrk metrics from {filepath}"
    return metrics


# --- Aggregation Function ---
# (aggregate_results remains the same as the provided script)
def aggregate_results(all_metrics_data):
    """Calculates mean and standard deviation for metrics across multiple runs."""
    collected_data = collections.defaultdict(list)
    valid_run_count = len(all_metrics_data)
    if valid_run_count == 0:
        return None
    # Use METRIC_KEYS defined in config now
    keys_to_aggregate = [k for k in METRIC_KEYS if k not in ["filepath", "error"]]
    for run_data in all_metrics_data:
        if not run_data: continue
        for key in keys_to_aggregate:
            value = run_data.get(key)
            if value is not None:
                collected_data[key].append(value)
    aggregated_stats = {}
    for metric_name, values in collected_data.items():
        count = len(values)
        if count > 0:
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if count > 1 else 0.0
            # Check consistency for config items
            is_config = metric_name in ["threads", "connections"]
            consistent = is_config and (stdev < 1e-9 or count == 1)
            aggregated_stats[metric_name] = {
                "mean": mean, "stdev": stdev, "count": count, "consistent": consistent
            }
    if not aggregated_stats:
        return None
    aggregated_stats["valid_run_count"] = valid_run_count
    return aggregated_stats


# --- Comparison Summary Generation Function (NEW) ---

def generate_comparison_summary(results1, results2, label1, label2):
    """Generates a comparison summary string including percentage difference."""
    if not results1 or not results2:
        return None # Cannot generate if data is missing

    label1_short = os.path.basename(label1.rstrip('/\\')) or "Baseline"
    label2_short = os.path.basename(label2.rstrip('/\\')) or "Comparison"

    # Define order of metrics for the summary table
    summary_metric_order = [
         "overall_req_sec", "overall_transfer_mb_sec", "latency_avg_ms",
         "latency_stdev_ms", "latency_max_ms", "req_sec_avg_thread",
         "total_requests", "duration_s"
    ]

    summary_lines = []
    separator = "-" * 95
    header_line1 = "=" * 95
    header_line2 = "wrk/nginx Benchmark Comparison Summary"
    header_line3 = f"Baseline (Dir1): {label1_short}"
    header_line4 = f"Comparison (Dir2): {label2_short}"
    header_line5 = "=" * 95
    col_header_format = "{:<28} | {:<25} | {:<25} | {:<10}"
    col_headers = col_header_format.format("Metric", f"{label1_short} (Mean+/-Stdev)", f"{label2_short} (Mean+/-Stdev)", "% Diff")

    summary_lines.extend([header_line1, header_line2, header_line3, header_line4, header_line5, col_headers, separator])

    # Handle config items first (check consistency)
    config_items = ["threads", "connections"]
    for metric in config_items:
         res1 = results1.get(metric)
         res2 = results2.get(metric)
         res1_str = "N/A"
         res2_str = "N/A"
         if res1:
             res1_str = f"{int(res1['mean'])} (n={res1['count']})" + ("" if res1.get('consistent') else " (!)")
         if res2:
             res2_str = f"{int(res2['mean'])} (n={res2['count']})" + ("" if res2.get('consistent') else " (!)")
         summary_lines.append(col_header_format.format(metric.title(), res1_str, res2_str, "")) # No %diff for config

    summary_lines.append(separator)


    # Handle performance metrics
    for metric in summary_metric_order:
        if metric in config_items: continue # Skip config items already handled

        res1 = results1.get(metric)
        res2 = results2.get(metric)

        def format_res(res):
            if res:
                precision = 3 if "latency" in metric else (0 if "requests" in metric else 2)
                mean_str = f"{res['mean']:.{precision}f}"
                stdev_str = f"{res['stdev']:.{precision}f}"
                count = res['count']
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
            if abs(mean1) > ZERO_THRESHOLD:
                percent_diff = ((mean2 - mean1) / mean1) * 100.0
                if not math.isnan(percent_diff) and not math.isinf(percent_diff):
                     percent_diff_str = f"{percent_diff:+.2f}%"
            elif abs(mean2) < ZERO_THRESHOLD:
                 percent_diff_str = "0.00%"

        # Determine unit for display
        unit = ""
        if "_ms" in metric: unit = "ms"
        elif "req_sec" in metric: unit = "req/s"
        elif "_mb" in metric: unit = "MB"
        elif "transfer" in metric: unit = "MB/s"
        elif "_s" in metric: unit = "s"

        metric_label = f"{metric.replace('_', ' ').title()} ({unit})" if unit else metric.replace('_', ' ').title()
        summary_lines.append(col_header_format.format(metric_label, res1_str, res2_str, percent_diff_str))

    summary_lines.append(separator)
    return "\n".join(summary_lines)


# --- Plotting Function (Modified) ---

def plot_comparison(agg_data1, agg_data2, label1, label2):
    """
    Creates separate comparison plots for RPS, Latency, Transfer Speed,
    and Percentage Difference. Saves plots as PDF.
    """
    if not PLOT_ENABLED:
        print("\nPlotting disabled as matplotlib/numpy are not installed.")
        return
    if not agg_data1 or not agg_data2:
        print("\nCannot generate plots due to missing aggregated data.", file=sys.stderr)
        return

    print("\nGenerating comparison plots...")

    label1_short = os.path.basename(label1.rstrip('/\\')) or "Baseline"
    label2_short = os.path.basename(label2.rstrip('/\\')) or "Comparison"

    # Create output directory if it doesn't exist
    output_plot_dir = "./plots"
    if not os.path.exists(output_plot_dir):
         try:
             os.makedirs(output_plot_dir)
             print(f"Created plot output directory: {output_plot_dir}")
         except OSError as e:
             print(f"Error: Could not create plot directory '{output_plot_dir}': {e}", file=sys.stderr)
             output_plot_dir = "." # Save in current dir as fallback

    # --- Plot 1: Requests per Second (Horizontal) ---
    metric_rps = "overall_req_sec"
    print(f" - Plotting Absolute '{metric_rps}'...")
    d1_rps_data = agg_data1.get(metric_rps)
    d2_rps_data = agg_data2.get(metric_rps)
    if d1_rps_data and d2_rps_data:
        means1 = [d1_rps_data["mean"]]
        stdevs1 = [d1_rps_data["stdev"]]
        means2 = [d2_rps_data["mean"]]
        stdevs2 = [d2_rps_data["stdev"]]
        y_pos = np.arange(len(means1)); bar_height = 0.35
        fig, ax = plt.subplots(figsize=(8, 4))
        rects1 = ax.barh(y_pos + bar_height / 2, means1, bar_height, xerr=stdevs1, label=label1_short, capsize=4, alpha=0.8)
        rects2 = ax.barh(y_pos - bar_height / 2, means2, bar_height, xerr=stdevs2, label=label2_short, capsize=4, alpha=0.8)
        ax.set_xlabel('Requests per Second (Overall)'); ax.set_title('Comparison of Overall Requests per Second')
        ax.set_yticks(y_pos); ax.set_yticklabels(['Req/Sec']); ax.invert_yaxis()
        ax.legend(loc='lower right'); ax.grid(axis='x', linestyle='--', alpha=0.6)
        ax.bar_label(rects1, padding=3, fmt='%.0f'); ax.bar_label(rects2, padding=3, fmt='%.0f')
        ax.set_xlim(right=max(means1 + means2) * 1.15 if max(means1 + means2) > 0 else 1)
        fig.tight_layout()
        plot_filename = os.path.join(output_plot_dir, "wrk_comparison_requests_per_second.pdf")
        try:
            plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
            print(f"   Plot saved to '{plot_filename}'")
        except Exception as e:
            err_msg = str(e).encode('ascii', 'replace').decode('ascii')
            print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
        plt.close(fig)
    else: print(f"   Skipping plot for '{metric_rps}': Missing data.")

    # --- Plot 2: Latency Figures (Vertical Grouped) ---
    latency_metrics_to_plot = ["latency_avg_ms", "latency_max_ms"]
    print(f" - Plotting Absolute Latency ({', '.join(latency_metrics_to_plot)})...")
    means1_lat, stdevs1_lat = [], []; means2_lat, stdevs2_lat = [], []; plot_latency_labels = []
    for metric in latency_metrics_to_plot:
        d1_lat, d2_lat = agg_data1.get(metric), agg_data2.get(metric)
        if d1_lat and d2_lat:
            means1_lat.append(d1_lat["mean"]); stdevs1_lat.append(d1_lat["stdev"])
            means2_lat.append(d2_lat["mean"]); stdevs2_lat.append(d2_lat["stdev"])
            plot_latency_labels.append(metric.replace('latency_', '').replace('_ms', '').replace('_', ' ').title())
    if plot_latency_labels:
        x_indices_lat = np.arange(len(plot_latency_labels)); bar_width = 0.35
        fig, ax = plt.subplots(figsize=(max(6, len(plot_latency_labels) * 2), 6))
        rects1 = ax.bar(x_indices_lat - bar_width / 2, means1_lat, bar_width, yerr=stdevs1_lat, label=label1_short, capsize=5, alpha=0.8)
        rects2 = ax.bar(x_indices_lat + bar_width / 2, means2_lat, bar_width, yerr=stdevs2_lat, label=label2_short, capsize=5, alpha=0.8)
        ax.set_ylabel('Latency (ms)'); ax.set_title('Comparison of Latency Metrics')
        ax.set_xticks(x_indices_lat); ax.set_xticklabels(plot_latency_labels)
        ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.bar_label(rects1, padding=3, fmt='%.2f'); ax.bar_label(rects2, padding=3, fmt='%.2f')
        ax.set_ylim(bottom=0, top=max(means1_lat + means2_lat) * 1.15 if max(means1_lat + means2_lat) > 0 else 1)
        fig.tight_layout()
        plot_filename = os.path.join(output_plot_dir, "wrk_comparison_latency.pdf")
        try:
            plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
            print(f"   Plot saved to '{plot_filename}'")
        except Exception as e:
            err_msg = str(e).encode('ascii', 'replace').decode('ascii')
            print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
        plt.close(fig)
    else: print(f"   Skipping latency plot: No common latency metrics with data found.")

    # --- Plot 3: Transfer Speed (Horizontal) ---
    metric_transfer = "overall_transfer_mb_sec"
    print(f" - Plotting Absolute '{metric_transfer}'...")
    d1_transfer, d2_transfer = agg_data1.get(metric_transfer), agg_data2.get(metric_transfer)
    if d1_transfer and d2_transfer:
        means1 = [d1_transfer["mean"]]; stdevs1 = [d1_transfer["stdev"]]
        means2 = [d2_transfer["mean"]]; stdevs2 = [d2_transfer["stdev"]]
        y_pos = np.arange(len(means1)); bar_height = 0.35
        fig, ax = plt.subplots(figsize=(8, 4))
        rects1 = ax.barh(y_pos + bar_height / 2, means1, bar_height, xerr=stdevs1, label=label1_short, capsize=4, alpha=0.8)
        rects2 = ax.barh(y_pos - bar_height / 2, means2, bar_height, xerr=stdevs2, label=label2_short, capsize=4, alpha=0.8)
        ax.set_xlabel('Transfer Speed (MB/s)'); ax.set_title('Comparison of Overall Transfer Speed')
        ax.set_yticks(y_pos); ax.set_yticklabels(['Transfer'])
        ax.invert_yaxis(); ax.legend(loc='lower right'); ax.grid(axis='x', linestyle='--', alpha=0.6)
        ax.bar_label(rects1, padding=3, fmt='%.1f'); ax.bar_label(rects2, padding=3, fmt='%.1f')
        ax.set_xlim(right=max(means1 + means2) * 1.15 if max(means1 + means2) > 0 else 1)
        fig.tight_layout()
        plot_filename = os.path.join(output_plot_dir, "wrk_comparison_transfer_speed.pdf")
        try:
            plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
            print(f"   Plot saved to '{plot_filename}'")
        except Exception as e:
            err_msg = str(e).encode('ascii', 'replace').decode('ascii')
            print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
        plt.close(fig)
    else: print(f"   Skipping plot for '{metric_transfer}': Missing data.")

    # --- Plot 4: Percentage Difference ---
    print(f" - Plotting Percentage Difference ({label2_short} vs {label1_short})...")
    percent_diffs = []
    plot_labels_p = []

    for metric in METRICS_TO_PLOT: # Use the same metrics as the absolute plots
        res1 = agg_data1.get(metric)
        res2 = agg_data2.get(metric)

        if res1 and res2 and res1["mean"] is not None and res2["mean"] is not None:
            mean1 = res1["mean"]
            mean2 = res2["mean"]
            percent_diff = float('nan') # Default to NaN
            if abs(mean1) > ZERO_THRESHOLD:
                diff = ((mean2 - mean1) / mean1) * 100.0
                if not math.isnan(diff) and not math.isinf(diff):
                     percent_diff = diff
            elif abs(mean2) < ZERO_THRESHOLD:
                 percent_diff = 0.0

            if not math.isnan(percent_diff): # Only include if valid %diff calculated
                 percent_diffs.append(percent_diff)
                 plot_labels_p.append(metric.replace('_', ' ').replace(' ms', '').replace(' mb sec', '').title())
        else:
             print(f"   Skipping metric '{metric}' for % diff plot: Missing data.")

    if plot_labels_p:
        x_indices_p = np.arange(len(plot_labels_p))
        fig_p, ax_p = plt.subplots(figsize=(max(8, len(plot_labels_p) * 1.8), 6))

        colors = ['red' if x < 0 else 'green' for x in percent_diffs] # Color bars by sign
        rects_p = ax_p.bar(x_indices_p, percent_diffs, color=colors, alpha=0.8)

        ax_p.set_ylabel('Percentage Difference (%)')
        ax_p.set_title(f'Benchmark % Difference ({label2_short} vs {label1_short})')
        ax_p.set_xticks(x_indices_p)
        ax_p.set_xticklabels(plot_labels_p, rotation=45, ha="right")
        ax_p.grid(axis='y', linestyle='--', alpha=0.6)
        ax_p.axhline(0, color='grey', linewidth=0.8) # Line at 0%

        ax_p.bar_label(rects_p, padding=3, fmt='%.1f%%') # Add percent sign

        # Adjust y limits to show positive and negative differences clearly
        max_abs_diff = max(abs(v) for v in percent_diffs) if percent_diffs else 10
        ax_p.set_ylim(bottom=min(0, -max_abs_diff * 1.15), top=max(0, max_abs_diff * 1.15))

        fig_p.tight_layout()
        plot_filename = os.path.join(output_plot_dir, "wrk_comparison_percent_diff.pdf")
        try:
            plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
            print(f"   Percentage Difference plot saved to '{plot_filename}'")
        except Exception as e:
            err_msg = str(e).encode("ascii", "replace").decode("ascii")
            print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
        plt.close(fig_p)
    else:
        print(f"   Skipping Percentage Difference plot: No common metrics with data for comparison.")


# --- Directory Processing Function ---
# (process_directory remains the same as the provided script)
def process_directory(directory_path):
    """Finds, parses, and aggregates log files in a directory."""
    print(f"\nProcessing directory: {directory_path}")
    if not os.path.isdir(directory_path):
        print(f"Error: Provided path '{directory_path}' is not a valid directory.", file=sys.stderr)
        return None, 0
    search_pattern = os.path.join(directory_path, LOG_FILE_PATTERN)
    log_files = glob.glob(search_pattern)
    if not log_files:
        print(f"Warning: No files matching pattern '{LOG_FILE_PATTERN}' found in directory '{directory_path}'.", file=sys.stderr)
        return None, 0
    all_results = []
    parse_errors = 0
    files_found_count = len(log_files)
    print(f"Found {files_found_count} file(s) matching '{LOG_FILE_PATTERN}'. Parsing...")
    for log_file in log_files:
        metrics = parse_wrk_log(log_file)
        if metrics.get("error"):
            if "Could not extract key wrk metrics" not in metrics["error"]:
                 print(f"\n   Parse Error in '{os.path.basename(log_file)}': {metrics['error']}", file=sys.stderr)
            parse_errors += 1
        elif metrics:
            all_results.append(metrics)
    valid_files_parsed = len(all_results)
    print(f"Successfully parsed data from {valid_files_parsed} file(s).")
    if parse_errors > 0:
        print(f"Skipped or encountered errors in {parse_errors} file(s).", file=sys.stderr)
    if valid_files_parsed > 0:
        aggregated_data = aggregate_results(all_results)
        return aggregated_data, valid_files_parsed
    else:
        print(f"No valid data collected from directory '{directory_path}'.")
        return None, 0


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare wrk/nginx logs from two directories, print summaries with % diff, save summary, and plot comparisons (ASCII only)."
    )
    parser.add_argument(
        "dir1", metavar="BASELINE_DIR", type=str,
        help="Path to the baseline directory containing benchmark *.log files.",
    )
    parser.add_argument(
        "dir2", metavar="COMPARISON_DIR", type=str,
        help="Path to the comparison directory containing benchmark *.log files.",
    )
    args = parser.parse_args()

    agg_data1, count1 = process_directory(args.dir1)
    agg_data2, count2 = process_directory(args.dir2)

    summary_text = None
    if agg_data1 and agg_data2:
        summary_text = generate_comparison_summary(agg_data1, agg_data2, args.dir1, args.dir2)

    if summary_text:
        print("\n" + summary_text)
        try:
            # Ensure output directory exists for summary file
            summary_dir = os.path.dirname(SUMMARY_FILENAME)
            if summary_dir and not os.path.exists(summary_dir):
                 os.makedirs(summary_dir)

            with open(SUMMARY_FILENAME, 'w', encoding='ascii') as f_summary:
                f_summary.write(summary_text)
            print(f"\nComparison summary saved to '{SUMMARY_FILENAME}'")
        except IOError as e:
            err_msg = str(e).encode('ascii', 'replace').decode('ascii')
            print(f"\nError: Could not write summary to file '{SUMMARY_FILENAME}': {err_msg}", file=sys.stderr)
        except OSError as e:
             err_msg = str(e).encode('ascii', 'replace').decode('ascii')
             print(f"\nError: Could not create directory for summary file '{SUMMARY_FILENAME}': {err_msg}", file=sys.stderr)

    else:
        print("\nComparison summary cannot be generated due to missing data.", file=sys.stderr)

    if agg_data1 and agg_data2:
        plot_comparison(agg_data1, agg_data2, args.dir1, args.dir2)
    elif PLOT_ENABLED:
        print("\nComparison plots cannot be generated as data from both directories is required.", file=sys.stderr)

    print("\nScript finished.")
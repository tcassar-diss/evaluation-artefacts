# Ensure only ASCII characters are used in this script
"""
Processes fio benchmark log files (*.log) from two directories,
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
# Specific clat percentiles to extract and aggregate (use strings like '99.00')
PERCENTILES_TO_AGGREGATE = ["99.00", "99.90", "99.99"]
# Key metrics to include in the text summary table
SUMMARY_METRIC_ORDER = [
    "iops", "bandwidth_mibps", "clat_avg_ms", "clat_max_ms",
    "clat_p99_00_ms", "clat_p99_90_ms", "clat_p99_99_ms", # Dynamically generated keys
    "cpu_usr_pct", "cpu_sys_pct", "disk_util_pct", "duration_sec"
]
# Key metrics to include in the percentage difference plot
METRICS_FOR_PERCENT_DIFF_PLOT = [
    "iops", "bandwidth_mibps", "clat_avg_ms", "clat_p99_00_ms"
]
# Output file for the text summary
SUMMARY_FILENAME = "./summary.txt"
# Threshold for baseline mean close to zero to avoid division issues
ZERO_THRESHOLD = 1e-9


# --- Fio Log Parsing Function ---
# (parse_fio_log remains the same as the previous version)
def parse_fio_log(filepath):
    """Parses a single fio log file to extract key metrics."""
    metrics = {
        "iops": None, "bandwidth_mibps": None, "clat_avg_ms": None,
        "clat_max_ms": None, "cpu_usr_pct": None, "cpu_sys_pct": None,
        "disk_util_pct": None, "duration_sec": None, "operation": None,
        "error": None, "filepath": filepath
    }
    clat_percentiles_us = {}
    if not os.path.exists(filepath):
        metrics["error"] = f"Error: File not found at {filepath}"
        return metrics
    if not os.path.isfile(filepath):
        metrics["error"] = f"Error: Path is not a file {filepath}"
        return metrics
    try:
        with open(filepath, "r", encoding='utf-8', errors='replace') as f:
            content = f.read()
        op = None
        bw_match = re.search(r"\s*(read|write)\s*:\s*IOPS=([\d.]+),\s*BW=([\d.]+)MiB/s", content)
        if bw_match:
            op = bw_match.group(1).lower()
            metrics["operation"] = op
            metrics["iops"] = float(bw_match.group(2))
            metrics["bandwidth_mibps"] = float(bw_match.group(3))
        else:
             bw_match_fallback = re.search(r"\s*(READ|WRITE)\s*:\s*bw=([\d.]+)MiB/s", content)
             if bw_match_fallback:
                 op = bw_match_fallback.group(1).lower()
                 metrics["operation"] = op
                 metrics["bandwidth_mibps"] = float(bw_match_fallback.group(2))
                 iops_match_fallback = re.search(r"\s+iops\s+:\s+.*?avg=([\d.]+),", content)
                 if iops_match_fallback: metrics["iops"] = float(iops_match_fallback.group(1))
        if not op:
             if not metrics["error"]: metrics["error"] = "Could not determine operation type (read/write) or find key BW/IOPS stats."
             return metrics
        clat_match = re.search(r"clat \(usec\):\s*min=\d+,\s*max=(\d+),\s*avg=([\d.]+),", content)
        if clat_match:
            metrics["clat_max_ms"] = float(clat_match.group(1)) / 1000.0
            metrics["clat_avg_ms"] = float(clat_match.group(2)) / 1000.0
        percentile_matches = re.findall(r"\|\s*([\d\.]+)th=\[(\d+)\],?", content)
        for p_match in percentile_matches:
            percentile_key = f"{float(p_match[0]):.2f}"
            clat_percentiles_us[percentile_key] = int(p_match[1])
            if percentile_key in PERCENTILES_TO_AGGREGATE:
                metric_key = f"clat_p{percentile_key.replace('.', '_')}_ms"
                metrics[metric_key] = clat_percentiles_us[percentile_key] / 1000.0
        cpu_match = re.search(r"cpu\s+:\s+usr=([\d\.]+)%,\s+sys=([\d\.]+)%,", content)
        if cpu_match:
            metrics["cpu_usr_pct"] = float(cpu_match.group(1))
            metrics["cpu_sys_pct"] = float(cpu_match.group(2))
        util_matches = re.findall(r"util=([\d\.]+)%", content)
        if util_matches: metrics["disk_util_pct"] = float(util_matches[-1])
        duration_match = re.search(r"run=(\d+)-(\d+)msec", content)
        if duration_match: metrics["duration_sec"] = int(duration_match.group(1)) / 1000.0
    except Exception as e:
        err_msg = str(e).encode('ascii', 'replace').decode('ascii')
        metrics["error"] = f"An error occurred during parsing {filepath}: {err_msg}"
    if metrics["iops"] is None or metrics["bandwidth_mibps"] is None or metrics["clat_avg_ms"] is None:
         if not metrics["error"]: metrics["error"] = f"Could not extract key fio metrics (IOPS/BW/clat) from '{os.path.basename(filepath)}'"
    return metrics


# --- Aggregation Function ---
# (aggregate_fio_results remains the same as the previous version)
def aggregate_fio_results(all_metrics_data):
    """Aggregates results from multiple fio log files."""
    collected_data = collections.defaultdict(list)
    valid_run_count = len(all_metrics_data)
    if valid_run_count == 0: return None
    keys_to_aggregate = set()
    for run_data in all_metrics_data:
        if run_data: keys_to_aggregate.update(k for k, v in run_data.items() if isinstance(v, (int, float)))
    for p in PERCENTILES_TO_AGGREGATE: keys_to_aggregate.add(f"clat_p{p.replace('.', '_')}_ms")
    operations = set()
    for run_data in all_metrics_data:
        if not run_data: continue
        op = run_data.get("operation");
        if op: operations.add(op)
        for key in keys_to_aggregate:
             value = run_data.get(key)
             if value is not None: collected_data[key].append(value)
    aggregated_stats = {}
    for metric_name, values in collected_data.items():
        count = len(values)
        if count > 0:
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if count > 1 else 0.0
            aggregated_stats[metric_name] = {"mean": mean, "stdev": stdev, "count": count}
    if not aggregated_stats: return None
    aggregated_stats["valid_run_count"] = valid_run_count
    aggregated_stats["operation"] = operations.pop() if len(operations) == 1 else "mixed"
    return aggregated_stats


# --- Comparison Summary Generation Function (NEW) ---

def generate_comparison_summary(results1, results2, label1, label2):
    """Generates a comparison summary string including percentage difference."""
    if not results1 or not results2:
        return None # Cannot generate if data is missing

    label1_short = os.path.basename(label1.rstrip('/\\')) or "Baseline"
    label2_short = os.path.basename(label2.rstrip('/\\')) or "Comparison"
    operation = results1.get("operation", results2.get("operation", "N/A")).upper()

    # Define order, dynamically add found percentile keys
    summary_metrics = []
    base_summary_order = [
        "iops", "bandwidth_mibps", "clat_avg_ms", "clat_max_ms"
    ]
    percentile_summary_order = sorted([
        f"clat_p{p.replace('.', '_')}_ms" for p in PERCENTILES_TO_AGGREGATE
        if f"clat_p{p.replace('.', '_')}_ms" in results1 or f"clat_p{p.replace('.', '_')}_ms" in results2
    ])
    resource_summary_order = [
        "cpu_usr_pct", "cpu_sys_pct", "disk_util_pct", "duration_sec"
    ]
    summary_metrics.extend(base_summary_order)
    summary_metrics.extend(percentile_summary_order)
    summary_metrics.extend(resource_summary_order)


    summary_lines = []
    separator = "-" * 95
    header_line1 = "=" * 95
    header_line2 = f"Fio Benchmark Comparison Summary ({operation})"
    header_line3 = f"Baseline (Dir1): {label1_short}"
    header_line4 = f"Comparison (Dir2): {label2_short}"
    header_line5 = "=" * 95
    col_header_format = "{:<28} | {:<25} | {:<25} | {:<10}"
    col_headers = col_header_format.format("Metric", f"{label1_short} (Mean+/-Stdev)", f"{label2_short} (Mean+/-Stdev)", "% Diff")

    summary_lines.extend([header_line1, header_line2, header_line3, header_line4, header_line5, col_headers, separator])

    for metric in summary_metrics:
        res1 = results1.get(metric)
        res2 = results2.get(metric)

        def format_res(res):
            if res:
                # Determine precision based on metric type
                if "ms" in metric: precision = 3
                elif "pct" in metric or "util" in metric: precision = 2
                elif "iops" in metric: precision = 1
                elif "mibps" in metric: precision = 1
                else: precision = 2 # Default

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
                     percent_diff_str = f"{percent_diff:+.2f}%" # Add sign explicitly
            elif abs(mean2) < ZERO_THRESHOLD:
                 percent_diff_str = "0.00%"
            # else: leave as N/A

        # Format metric name nicely
        metric_label = metric.replace('_', ' ').replace('clat p', 'P').replace(' pct', '%').replace(' mibps', ' (MiB/s)').replace(' ms', ' (ms)').replace(' sec', ' (s)')
        metric_label = metric_label.replace(' iops', ' IOPS').replace(' cpu', ' CPU').replace(' disk', ' Disk').title()

        summary_lines.append(col_header_format.format(metric_label, res1_str, res2_str, percent_diff_str))

    summary_lines.append(separator)
    return "\n".join(summary_lines)


# --- Plotting Function (Modified) ---

def plot_fio_comparison(agg_data1, agg_data2, label1, label2):
    """
    Creates separate comparison plots for Fio Throughput, Latency, Resources,
    and Percentage Difference. Saves plots as PDF.
    """
    if not PLOT_ENABLED:
        print("\nPlotting disabled as matplotlib/numpy are not installed.")
        return
    if not agg_data1 or not agg_data2:
        print("\nCannot generate plots due to missing aggregated data.", file=sys.stderr)
        return

    print("\nGenerating Fio comparison plots...")

    label1_short = os.path.basename(label1.rstrip('/\\')) or "Baseline"
    label2_short = os.path.basename(label2.rstrip('/\\')) or "Comparison"
    operation = agg_data1.get("operation", "Mixed").upper()

    # Create output directory if it doesn't exist
    output_plot_dir = "./plots"
    if not os.path.exists(output_plot_dir):
         try:
             os.makedirs(output_plot_dir)
             print(f"Created plot output directory: {output_plot_dir}")
         except OSError as e:
             print(f"Error: Could not create plot directory '{output_plot_dir}': {e}", file=sys.stderr)
             output_plot_dir = "." # Save in current dir as fallback


    # Helper for saving plots
    def save_plot(fig, filename_base):
        plot_filename = os.path.join(output_plot_dir, f"{filename_base}.pdf")
        try:
            fig.tight_layout()
            plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
            print(f"   Plot saved to '{plot_filename}'")
        except Exception as e:
            err_msg = str(e).encode('ascii', 'replace').decode('ascii')
            print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
        plt.close(fig)

    # --- Plot 1: IOPS ---
    metric_key = "iops"; title = f"Fio IOPS Comparison ({operation})"; ylabel = "IOPS"
    print(f" - Plotting '{metric_key}'...")
    d1_data = agg_data1.get(metric_key); d2_data = agg_data2.get(metric_key)
    if d1_data and d2_data:
        means1=[d1_data["mean"]]; stdevs1=[d1_data["stdev"]]; means2=[d2_data["mean"]]; stdevs2=[d2_data["stdev"]]
        x_indices=np.arange(1); bar_width=0.35
        fig, ax = plt.subplots(figsize=(6, 5))
        r1=ax.bar(x_indices - bar_width/2, means1, bar_width, yerr=stdevs1, label=label1_short, capsize=5, alpha=0.8)
        r2=ax.bar(x_indices + bar_width/2, means2, bar_width, yerr=stdevs2, label=label2_short, capsize=5, alpha=0.8)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.set_xticks(x_indices); ax.set_xticklabels(['IOPS'])
        ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.6); ax.bar_label(r1, padding=3, fmt='%.0f'); ax.bar_label(r2, padding=3, fmt='%.0f')
        ax.set_ylim(bottom=0, top=max(means1 + means2) * 1.15 if max(means1 + means2) > 0 else 1)
        save_plot(fig, "fio_comparison_iops")
    else: print(f"   Skipping {ylabel} plot: Missing data.")

    # --- Plot 2: Bandwidth ---
    metric_key = "bandwidth_mibps"; title = f"Fio Bandwidth Comparison ({operation})"; ylabel = "Bandwidth (MiB/s)"
    print(f" - Plotting '{metric_key}'...")
    d1_data = agg_data1.get(metric_key); d2_data = agg_data2.get(metric_key)
    if d1_data and d2_data:
        means1=[d1_data["mean"]]; stdevs1=[d1_data["stdev"]]; means2=[d2_data["mean"]]; stdevs2=[d2_data["stdev"]]
        x_indices=np.arange(1); bar_width=0.35
        fig, ax = plt.subplots(figsize=(6, 5))
        r1=ax.bar(x_indices - bar_width/2, means1, bar_width, yerr=stdevs1, label=label1_short, capsize=5, alpha=0.8)
        r2=ax.bar(x_indices + bar_width/2, means2, bar_width, yerr=stdevs2, label=label2_short, capsize=5, alpha=0.8)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.set_xticks(x_indices); ax.set_xticklabels(['Bandwidth'])
        ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.6); ax.bar_label(r1, padding=3, fmt='%.0f'); ax.bar_label(r2, padding=3, fmt='%.0f')
        ax.set_ylim(bottom=0, top=max(means1 + means2) * 1.15 if max(means1 + means2) > 0 else 1)
        save_plot(fig, "fio_comparison_bandwidth")
    else: print(f"   Skipping {ylabel} plot: Missing data.")

    # --- Plot 3: Latency ---
    latency_metrics = ["clat_avg_ms"] + sorted([f"clat_p{p.replace('.', '_')}_ms" for p in PERCENTILES_TO_AGGREGATE])
    print(f" - Plotting Latency ({', '.join(latency_metrics)})...")
    means1_lat, stdevs1_lat = [], []; means2_lat, stdevs2_lat = [], []; plot_latency_labels = []
    for metric in latency_metrics:
        d1_lat = agg_data1.get(metric); d2_lat = agg_data2.get(metric)
        if d1_lat and d2_lat:
            means1_lat.append(d1_lat["mean"]); stdevs1_lat.append(d1_lat["stdev"])
            means2_lat.append(d2_lat["mean"]); stdevs2_lat.append(d2_lat["stdev"])
            label = metric.replace('clat_', '').replace('_ms', '').replace('p', 'P').replace('_', '.').title()
            plot_latency_labels.append(label)
        else: print(f"   Skipping latency metric '{metric}' for plot: Missing data.")
    if plot_latency_labels:
        x_indices = np.arange(len(plot_latency_labels)); bar_width = 0.35
        fig, ax = plt.subplots(figsize=(max(6, len(plot_latency_labels) * 1.5), 5))
        r1 = ax.bar(x_indices - bar_width/2, means1_lat, bar_width, yerr=stdevs1_lat, label=label1_short, capsize=5, alpha=0.8)
        r2 = ax.bar(x_indices + bar_width/2, means2_lat, bar_width, yerr=stdevs2_lat, label=label2_short, capsize=5, alpha=0.8)
        ax.set_ylabel('Latency (ms)'); ax.set_title(f'Fio Completion Latency Comparison ({operation})')
        ax.set_xticks(x_indices); ax.set_xticklabels(plot_latency_labels); ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.6); ax.bar_label(r1, padding=3, fmt='%.2f'); ax.bar_label(r2, padding=3, fmt='%.2f')
        ax.set_ylim(bottom=0, top=max(means1_lat + means2_lat) * 1.15 if max(means1_lat + means2_lat) > 0 else 1)
        save_plot(fig, "fio_comparison_latency")
    else: print(f"   Skipping Latency plot: No common latency metrics with data.")

    # --- Plot 4: Resource Usage ---
    resource_metrics = ["cpu_usr_pct", "cpu_sys_pct", "disk_util_pct"]
    print(f" - Plotting Resources ({', '.join(resource_metrics)})...")
    means1_res, stdevs1_res = [], []; means2_res, stdevs2_res = [], []; plot_resource_labels = []
    for metric in resource_metrics:
        d1_res = agg_data1.get(metric); d2_res = agg_data2.get(metric)
        if d1_res and d2_res:
            means1_res.append(d1_res["mean"]); stdevs1_res.append(d1_res["stdev"])
            means2_res.append(d2_res["mean"]); stdevs2_res.append(d2_res["stdev"])
            label = metric.replace('_pct', '').replace('cpu_', 'CPU ').replace('disk_', 'Disk ').replace('_', ' ').title()
            plot_resource_labels.append(label)
        else: print(f"   Skipping resource metric '{metric}' for plot: Missing data.")
    if plot_resource_labels:
        x_indices = np.arange(len(plot_resource_labels)); bar_width = 0.35
        fig, ax = plt.subplots(figsize=(max(6, len(plot_resource_labels) * 1.8), 5))
        r1 = ax.bar(x_indices - bar_width/2, means1_res, bar_width, yerr=stdevs1_res, label=label1_short, capsize=5, alpha=0.8)
        r2 = ax.bar(x_indices + bar_width/2, means2_res, bar_width, yerr=stdevs2_res, label=label2_short, capsize=5, alpha=0.8)
        ax.set_ylabel('Utilization (%)'); ax.set_title(f'Fio Resource Usage Comparison ({operation})')
        ax.set_xticks(x_indices); ax.set_xticklabels(plot_resource_labels); ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.6); ax.bar_label(r1, padding=3, fmt='%.1f'); ax.bar_label(r2, padding=3, fmt='%.1f')
        ax.set_ylim(bottom=0, top=max(means1_res + means2_res + [10]) * 1.15) # Ensure some space, max maybe 100
        save_plot(fig, "fio_comparison_resources")
    else: print(f"   Skipping Resource plot: No common resource metrics with data.")

    # --- Plot 5: Percentage Difference ---
    print(f" - Plotting Percentage Difference ({label2_short} vs {label1_short})...")
    percent_diffs = []
    plot_labels_p = []

    # Use key metrics defined for this plot
    for metric in METRICS_FOR_PERCENT_DIFF_PLOT:
        # Handle dynamically generated percentile keys
        if ".ms" in metric and "_p" in metric: # Assume it's a percentile key
             p_num_str = metric.split("_p")[-1].split("_ms")[0].replace("_",".")
             if p_num_str not in PERCENTILES_TO_AGGREGATE:
                 # print(f"   Skipping {metric} in % diff plot, not in PERCENTILES_TO_AGGREGATE")
                 continue # Skip if this percentile wasn't configured for aggregation

        res1 = agg_data1.get(metric)
        res2 = agg_data2.get(metric)

        if res1 and res2 and res1["mean"] is not None and res2["mean"] is not None:
            mean1 = res1["mean"]
            mean2 = res2["mean"]
            percent_diff = float('nan')
            if abs(mean1) > ZERO_THRESHOLD:
                diff = ((mean2 - mean1) / mean1) * 100.0
                if not math.isnan(diff) and not math.isinf(diff): percent_diff = diff
            elif abs(mean2) < ZERO_THRESHOLD: percent_diff = 0.0

            if not math.isnan(percent_diff):
                percent_diffs.append(percent_diff)
                # Format label nicely
                label = metric.replace('_mibps', ' BW').replace('_ms',' Lat').replace('clat_p', 'P').replace('_', '.').replace('pct', '%').replace(' iops', ' IOPS')
                label = label.replace(' cpu', ' CPU').replace(' disk', ' Disk').replace(' usr', ' User').replace(' sys', ' System').title()
                plot_labels_p.append(label)
        else:
             print(f"   Skipping metric '{metric}' for % diff plot: Missing data.")

    if plot_labels_p:
        x_indices_p = np.arange(len(plot_labels_p))
        fig_p, ax_p = plt.subplots(figsize=(max(8, len(plot_labels_p) * 1.8), 6))
        colors = ['red' if x < 0 else 'green' for x in percent_diffs]
        rects_p = ax_p.bar(x_indices_p, percent_diffs, color=colors, alpha=0.8)
        ax_p.set_ylabel('Percentage Difference (%)')
        ax_p.set_title(f'Fio Performance % Difference ({label2_short} vs {label1_short})')
        ax_p.set_xticks(x_indices_p); ax_p.set_xticklabels(plot_labels_p, rotation=45, ha="right")
        ax_p.grid(axis='y', linestyle='--', alpha=0.6); ax_p.axhline(0, color='grey', linewidth=0.8)
        ax_p.bar_label(rects_p, padding=3, fmt='%.1f%%')
        max_abs_diff = max(abs(v) for v in percent_diffs) if percent_diffs else 10
        ax_p.set_ylim(bottom=min(0, -max_abs_diff * 1.15), top=max(0, max_abs_diff * 1.15))
        save_plot(fig_p, "fio_comparison_percent_diff")
    else:
         print(f"   Skipping Percentage Difference plot: No common metrics with data.")


# --- Directory Processing Function ---
# (process_directory remains the same as the previous version)
def process_directory(directory_path):
    """Finds, parses, and aggregates fio log files in a directory."""
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
        metrics = parse_fio_log(log_file)
        if metrics.get("error"):
            if "Could not extract key fio metrics" not in metrics["error"]:
                 print(f"\n   Parse Error in '{os.path.basename(log_file)}': {metrics['error']}", file=sys.stderr)
            parse_errors += 1
        elif metrics:
            all_results.append(metrics)
    valid_files_parsed = len(all_results)
    print(f"Successfully parsed data from {valid_files_parsed} file(s).")
    if parse_errors > 0:
        print(f"Skipped or encountered errors in {parse_errors} file(s).", file=sys.stderr)
    if valid_files_parsed > 0:
        aggregated_data = aggregate_fio_results(all_results)
        return aggregated_data, valid_files_parsed
    else:
        print(f"No valid data collected from directory '{directory_path}'.")
        return None, 0


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Fio benchmark logs from two directories, print summaries with % diff, save summary, and plot comparisons (ASCII only)."
    )
    parser.add_argument(
        "dir1", metavar="BASELINE_DIR", type=str,
        help="Path to the baseline directory containing Fio benchmark *.log files.",
    )
    parser.add_argument(
        "dir2", metavar="COMPARISON_DIR", type=str,
        help="Path to the comparison directory containing Fio benchmark *.log files.",
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
        plot_fio_comparison(agg_data1, agg_data2, args.dir1, args.dir2)
    elif PLOT_ENABLED:
        print("\nComparison plots cannot be generated as data from both directories is required.", file=sys.stderr)

    print("\nScript finished.")
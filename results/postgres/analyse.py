# Ensure only ASCII characters are used in this script
"""
Processes pgbench benchmark log files (*.log) from two directories,
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
# Key metrics to include in the text summary table and plots
# Note: Order here influences the summary table rows
SUMMARY_METRIC_ORDER = [
    "tps", "latency_avg_ms", "transactions_processed",
    "transactions_failed_count", "transactions_failed_pct",
    "initial_conn_time_ms"
]
# Key metrics specifically for the percentage difference plot
METRICS_FOR_PERCENT_DIFF_PLOT = ["tps", "latency_avg_ms"]
# Output file for the text summary
SUMMARY_FILENAME = "./summary.txt"
# Threshold for baseline mean close to zero to avoid division issues
ZERO_THRESHOLD = 1e-9


# --- pgbench Log Parsing Function ---
# (parse_pgbench_log remains the same as the previous version)
def parse_pgbench_log(filepath):
    """Parses a single pgbench log file to extract key metrics."""
    metrics = {
        "scaling_factor": None, "query_mode": None, "clients": None,
        "threads": None, "duration_s": None, "transactions_processed": None,
        "transactions_failed_count": None, "transactions_failed_pct": None,
        "latency_avg_ms": None, "initial_conn_time_ms": None, "tps": None,
        "error": None, "filepath": filepath
    }
    if not os.path.exists(filepath):
        metrics["error"] = f"Error: File not found at {filepath}"
        return metrics
    if not os.path.isfile(filepath):
        metrics["error"] = f"Error: Path is not a file {filepath}"
        return metrics
    try:
        with open(filepath, "r", encoding='utf-8', errors='replace') as f:
            content = f.read()
        match = re.search(r"scaling factor:\s+(\d+)", content);
        if match: metrics["scaling_factor"] = int(match.group(1))
        match = re.search(r"query mode:\s+(\w+)", content);
        if match: metrics["query_mode"] = match.group(1)
        match = re.search(r"number of clients:\s+(\d+)", content);
        if match: metrics["clients"] = int(match.group(1))
        match = re.search(r"number of threads:\s+(\d+)", content);
        if match: metrics["threads"] = int(match.group(1))
        match = re.search(r"duration:\s+(\d+)\s+s", content);
        if match: metrics["duration_s"] = int(match.group(1))
        match = re.search(r"number of transactions actually processed:\s+(\d+)", content);
        if match: metrics["transactions_processed"] = int(match.group(1))
        match = re.search(r"number of failed transactions:\s+(\d+)\s+\(([\d.]+)%\)", content);
        if match:
            metrics["transactions_failed_count"] = int(match.group(1))
            metrics["transactions_failed_pct"] = float(match.group(2))
        match = re.search(r"latency average\s*=\s*([\d.]+)\s+ms", content);
        if match: metrics["latency_avg_ms"] = float(match.group(1))
        match = re.search(r"initial connection time\s*=\s*([\d.]+)\s+ms", content);
        if match: metrics["initial_conn_time_ms"] = float(match.group(1))
        match = re.search(r"tps\s*=\s*([\d.]+)\s+\(w", content);
        if match: metrics["tps"] = float(match.group(1))
    except Exception as e:
        err_msg = str(e).encode('ascii', 'replace').decode('ascii')
        metrics["error"] = f"An error occurred during parsing {filepath}: {err_msg}"
    if metrics["tps"] is None or metrics["latency_avg_ms"] is None:
         if not metrics["error"]:
              metrics["error"] = f"Could not extract key pgbench metrics (TPS/Latency) from '{os.path.basename(filepath)}'"
    return metrics


# --- Aggregation Function ---
# (aggregate_pgbench_results remains the same as the previous version)
def aggregate_pgbench_results(all_metrics_data):
    """Calculates mean and standard deviation for metrics across multiple runs."""
    aggregated_results = {}
    valid_run_count = len(all_metrics_data)
    if valid_run_count == 0: return None
    query_modes = set(run_data.get("query_mode") for run_data in all_metrics_data if run_data and run_data.get("query_mode"))
    if len(query_modes) == 1: aggregated_results["query_mode"] = {"value": query_modes.pop(), "consistent": True, "count": valid_run_count}
    elif len(query_modes) > 1:
        mode_count = sum(1 for run_data in all_metrics_data if run_data and run_data.get("query_mode"))
        aggregated_results["query_mode"] = {"value": list(query_modes), "consistent": False, "count": mode_count}
    else: aggregated_results["query_mode"] = {"value": None, "consistent": False, "count": 0}
    numeric_keys = [
        "scaling_factor", "clients", "threads", "duration_s",
        "transactions_processed", "transactions_failed_count", "transactions_failed_pct",
        "latency_avg_ms", "initial_conn_time_ms", "tps"
    ]
    collected_data = collections.defaultdict(list)
    for run_data in all_metrics_data:
        if not run_data: continue
        for key in numeric_keys:
            value = run_data.get(key)
            if value is not None: collected_data[key].append(value)
    for metric_name, values in collected_data.items():
        count = len(values)
        is_config = metric_name in ["scaling_factor", "clients", "threads", "duration_s"]
        if count > 0:
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if count > 1 else 0.0
            consistent = is_config and (stdev < 1e-9 or count == 1)
            aggregated_results[metric_name] = {"mean": mean, "stdev": stdev, "count": count, "consistent": consistent}
    if not any(key in aggregated_results for key in ["tps", "latency_avg_ms"]): return None
    aggregated_results["valid_run_count"] = valid_run_count
    return aggregated_results


# --- Comparison Summary Generation Function (NEW) ---

def generate_comparison_summary(results1, results2, label1, label2):
    """Generates a comparison summary string including percentage difference."""
    if not results1 or not results2:
        return None

    label1_short = os.path.basename(label1.rstrip('/\\')) or "Baseline"
    label2_short = os.path.basename(label2.rstrip('/\\')) or "Comparison"

    summary_lines = []
    separator = "-" * 100 # Increased width
    header_line1 = "=" * 100
    header_line2 = "pgbench Benchmark Comparison Summary"
    header_line3 = f"Baseline (Dir1): {label1_short}"
    header_line4 = f"Comparison (Dir2): {label2_short}"
    header_line5 = "=" * 100
    col_header_format = "{:<30} | {:<25} | {:<25} | {:<12}"
    col_headers = col_header_format.format("Configuration / Metric", f"{label1_short} (Mean+/-Stdev)", f"{label2_short} (Mean+/-Stdev)", "% Diff")

    summary_lines.extend([header_line1, header_line2, header_line3, header_line4, header_line5, col_headers, separator])

    # --- Configuration Section ---
    summary_lines.append("Configuration:")
    config_items = ["scaling_factor", "query_mode", "clients", "threads", "duration_s"]
    for metric in config_items:
         res1 = results1.get(metric)
         res2 = results2.get(metric)
         res1_str = "N/A"; res2_str = "N/A"; count1 = 0; count2 = 0;
         consistent1 = False; consistent2 = False;
         is_str_val = metric == "query_mode"

         if res1:
             count1 = res1['count']
             consistent1 = res1.get('consistent', False) if not is_str_val else res1.get('consistent', False)
             if is_str_val:
                 res1_str = f"{res1['value']} (n={count1})" + (" (consistent)" if consistent1 else (" (!)" if not consistent1 and count1 > 0 else ""))
             elif res1.get('mean') is not None:
                 is_int = isinstance(res1['mean'], int) or metric in ["clients", "threads", "scaling_factor", "duration_s"]
                 precision = 0 if is_int else 2
                 if consistent1: res1_str = f"{res1['mean']:.{precision}f} (n={count1}, consistent)"
                 else: res1_str = f"{res1['mean']:.{precision}f} +/- {res1['stdev']:.{precision}f} (n={count1})"

         if res2:
             count2 = res2['count']
             consistent2 = res2.get('consistent', False) if not is_str_val else res2.get('consistent', False)
             if is_str_val:
                 res2_str = f"{res2['value']} (n={count2})" + (" (consistent)" if consistent2 else (" (!)" if not consistent2 and count2 > 0 else ""))
             elif res2.get('mean') is not None:
                 is_int = isinstance(res2['mean'], int) or metric in ["clients", "threads", "scaling_factor", "duration_s"]
                 precision = 0 if is_int else 2
                 if consistent2: res2_str = f"{res2['mean']:.{precision}f} (n={count2}, consistent)"
                 else: res2_str = f"{res2['mean']:.{precision}f} +/- {res2['stdev']:.{precision}f} (n={count2})"

         summary_lines.append(col_header_format.format(f"  {metric.replace('_',' ').title()}", res1_str, res2_str, "")) # No %diff for config

    summary_lines.append(separator)

    # --- Results Section ---
    summary_lines.append("Results:")
    for metric in SUMMARY_METRIC_ORDER:
        res1 = results1.get(metric)
        res2 = results2.get(metric)

        def format_res(res):
            if res and res.get("mean") is not None:
                is_int = metric in ["transactions_processed", "transactions_failed_count"]
                precision = 0 if is_int else (3 if "latency" in metric or "pct" in metric else 2)
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
        if res1 and res2 and res1.get("mean") is not None and res2.get("mean") is not None:
            mean1 = res1["mean"]
            mean2 = res2["mean"]
            if abs(mean1) > ZERO_THRESHOLD:
                percent_diff = ((mean2 - mean1) / mean1) * 100.0
                if not math.isnan(percent_diff) and not math.isinf(percent_diff):
                     percent_diff_str = f"{percent_diff:+.2f}%"
            elif abs(mean2) < ZERO_THRESHOLD:
                 percent_diff_str = "0.00%"

        # Format metric name nicely
        unit = ""
        if "_ms" in metric: unit = "ms"
        elif "_s" in metric: unit = "s"
        elif "_pct" in metric: unit = "%"
        metric_label = metric.replace('_ms','').replace('_s','').replace('_pct','').replace('_',' ').title()
        if unit: metric_label += f" ({unit})"

        summary_lines.append(col_header_format.format(f"  {metric_label}", res1_str, res2_str, percent_diff_str))

    summary_lines.append(separator)
    return "\n".join(summary_lines)


# --- Plotting Function (Modified) ---

def plot_pgbench_comparison(agg_data1, agg_data2, label1, label2):
    """
    Creates comparison plots for pgbench TPS, Latency, Transactions,
    Connection Time, and Percentage Difference. Saves plots as PDF.
    """
    if not PLOT_ENABLED:
        print("\nPlotting disabled as matplotlib/numpy are not installed.")
        return
    if not agg_data1 or not agg_data2:
        print("\nCannot generate plots due to missing aggregated data.", file=sys.stderr)
        return

    print("\nGenerating pgbench comparison plots...")

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

    # Helper function for saving plots
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

    # Helper function for creating individual bar plots
    def create_single_metric_plot(metric_key, title, ylabel, filename_suffix, precision=2):
        print(f" - Plotting Absolute '{metric_key}'...")
        d1_data = agg_data1.get(metric_key)
        d2_data = agg_data2.get(metric_key)
        if d1_data and d2_data:
            means1 = [d1_data["mean"]]; stdevs1 = [d1_data["stdev"]]
            means2 = [d2_data["mean"]]; stdevs2 = [d2_data["stdev"]]
            x_indices = np.arange(1); bar_width = 0.35
            fig, ax = plt.subplots(figsize=(6, 5))
            r1 = ax.bar(x_indices - bar_width/2, means1, bar_width, yerr=stdevs1, label=label1_short, capsize=5, alpha=0.8)
            r2 = ax.bar(x_indices + bar_width/2, means2, bar_width, yerr=stdevs2, label=label2_short, capsize=5, alpha=0.8)
            ax.set_ylabel(ylabel); ax.set_title(title); ax.set_xticks(x_indices)
            label_text = metric_key.replace('_', ' ').replace(' ms', '').title()
            ax.set_xticklabels([label_text])
            ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.6)
            fmt_str = f'%.{precision}f'
            ax.bar_label(r1, padding=3, fmt=fmt_str); ax.bar_label(r2, padding=3, fmt=fmt_str)
            ax.set_ylim(bottom=0, top=max(means1 + means2) * 1.15 if max(means1 + means2) > 0 else 1)
            save_plot(fig, f"pgbench_comparison_{filename_suffix}")
        else: print(f"   Skipping plot for '{metric_key}': Missing data.")

    # --- Create the individual absolute value plots ---
    create_single_metric_plot("tps", "Comparison of Transactions Per Second", "TPS", "tps", precision=1)
    create_single_metric_plot("latency_avg_ms", "Comparison of Average Latency", "Latency (ms)", "latency", precision=3)
    create_single_metric_plot("initial_conn_time_ms", "Comparison of Initial Connection Time", "Time (ms)", "conn_time", precision=3)

    # --- Plot 4: Transaction Counts (Grouped Bar) ---
    transaction_metrics = ["transactions_processed", "transactions_failed_count"]
    print(f" - Plotting Absolute Transactions ({', '.join(transaction_metrics)})...")
    means1_txn, stdevs1_txn = [], []; means2_txn, stdevs2_txn = [], []; plot_txn_labels = []
    for metric in transaction_metrics:
        d1_txn = agg_data1.get(metric); d2_txn = agg_data2.get(metric)
        if d1_txn and d2_txn:
            means1_txn.append(d1_txn["mean"]); stdevs1_txn.append(d1_txn["stdev"])
            means2_txn.append(d2_txn["mean"]); stdevs2_txn.append(d2_txn["stdev"])
            plot_txn_labels.append(metric.replace('transactions_', '').replace('_count','').replace('_',' ').title())
        else: print(f"   Skipping transaction metric '{metric}' for plot: Missing data.")
    if plot_txn_labels:
        x_indices_txn = np.arange(len(plot_txn_labels)); bar_width = 0.35
        fig_txn, ax_txn = plt.subplots(figsize=(max(6, len(plot_txn_labels) * 2.5), 6))
        r1 = ax_txn.bar(x_indices_txn - bar_width/2, means1_txn, bar_width, yerr=stdevs1_txn, label=label1_short, capsize=5, alpha=0.8)
        r2 = ax_txn.bar(x_indices_txn + bar_width/2, means2_txn, bar_width, yerr=stdevs2_txn, label=label2_short, capsize=5, alpha=0.8)
        ax_txn.set_ylabel('Count'); ax_txn.set_title('Comparison of Transaction Counts')
        ax_txn.set_xticks(x_indices_txn); ax_txn.set_xticklabels(plot_txn_labels)
        ax_txn.legend(); ax_txn.grid(axis='y', linestyle='--', alpha=0.6)
        ax_txn.bar_label(r1, padding=3, fmt='%.0f'); ax_txn.bar_label(r2, padding=3, fmt='%.0f')
        ax_txn.set_ylim(bottom=0)
        if ax_txn.get_yscale() != 'log':
             ax_txn.set_ylim(top=max(means1_txn + means2_txn) * 1.15 if max(means1_txn + means2_txn) > 0 else 1)
        save_plot(fig_txn, "pgbench_comparison_transactions")
    else: print(f"   Skipping Transactions plot: No common transaction metrics with data found.")


    # --- Plot 5: Percentage Difference ---
    print(f" - Plotting Percentage Difference ({label2_short} vs {label1_short})...")
    percent_diffs = []
    plot_labels_p = []

    for metric in METRICS_FOR_PERCENT_DIFF_PLOT: # Use specific list for this plot
        res1 = agg_data1.get(metric)
        res2 = agg_data2.get(metric)

        if res1 and res2 and res1.get("mean") is not None and res2.get("mean") is not None:
            mean1 = res1["mean"]
            mean2 = res2["mean"]
            percent_diff = float('nan')
            if abs(mean1) > ZERO_THRESHOLD:
                diff = ((mean2 - mean1) / mean1) * 100.0
                if not math.isnan(diff) and not math.isinf(diff): percent_diff = diff
            elif abs(mean2) < ZERO_THRESHOLD: percent_diff = 0.0

            if not math.isnan(percent_diff):
                percent_diffs.append(percent_diff)
                label = metric.replace('_', ' ').replace(' ms', '').title()
                plot_labels_p.append(label)
        else:
             print(f"   Skipping metric '{metric}' for % diff plot: Missing data.")

    if plot_labels_p:
        x_indices_p = np.arange(len(plot_labels_p))
        fig_p, ax_p = plt.subplots(figsize=(max(6, len(plot_labels_p) * 2), 6))
        colors = ['red' if x < 0 else 'green' for x in percent_diffs]
        rects_p = ax_p.bar(x_indices_p, percent_diffs, color=colors, alpha=0.8)
        ax_p.set_ylabel('Percentage Difference (%)')
        ax_p.set_title(f'pgbench Performance % Difference ({label2_short} vs {label1_short})')
        ax_p.set_xticks(x_indices_p); ax_p.set_xticklabels(plot_labels_p, rotation=45, ha="right")
        ax_p.grid(axis='y', linestyle='--', alpha=0.6); ax_p.axhline(0, color='grey', linewidth=0.8)
        ax_p.bar_label(rects_p, padding=3, fmt='%.1f%%')
        max_abs_diff = max(abs(v) for v in percent_diffs) if percent_diffs else 10
        ax_p.set_ylim(bottom=min(0, -max_abs_diff * 1.15), top=max(0, max_abs_diff * 1.15))
        save_plot(fig_p, "pgbench_comparison_percent_diff")
    else:
         print(f"   Skipping Percentage Difference plot: No common metrics with data.")


# --- Directory Processing Function ---
# (process_directory remains the same as the previous version)
def process_directory(directory_path):
    """Finds, parses, and aggregates pgbench log files in a directory."""
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
        metrics = parse_pgbench_log(log_file)
        if metrics.get("error"):
            if "Could not extract key pgbench metrics" not in metrics["error"]:
                 print(f"\n   Parse Error in '{os.path.basename(log_file)}': {metrics['error']}", file=sys.stderr)
            parse_errors += 1
        elif metrics:
            all_results.append(metrics)
    valid_files_parsed = len(all_results)
    print(f"Successfully parsed data from {valid_files_parsed} file(s).")
    if parse_errors > 0:
        print(f"Skipped or encountered errors in {parse_errors} file(s).", file=sys.stderr)
    if valid_files_parsed > 0:
        aggregated_data = aggregate_pgbench_results(all_results)
        return aggregated_data, valid_files_parsed
    else:
        print(f"No valid data collected from directory '{directory_path}'.")
        return None, 0


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare pgbench logs from two directories, print summaries with % diff, save summary, and plot comparisons (ASCII only)."
    )
    parser.add_argument(
        "dir1", metavar="BASELINE_DIR", type=str,
        help="Path to the baseline directory containing pgbench benchmark *.log files.",
    )
    parser.add_argument(
        "dir2", metavar="COMPARISON_DIR", type=str,
        help="Path to the comparison directory containing pgbench benchmark *.log files.",
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
            if summary_dir and not os.path.exists(summary_dir): os.makedirs(summary_dir) # Create dir if needed
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
        plot_pgbench_comparison(agg_data1, agg_data2, args.dir1, args.dir2)
    elif PLOT_ENABLED:
        print("\nComparison plots cannot be generated as data from both directories is required.", file=sys.stderr)

    print("\nScript finished.")
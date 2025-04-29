# Ensure only ASCII characters are used in this script
"""
Processes pgbench benchmark log files (*.log) from two directories,
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
    print("Warning: matplotlib or numpy not found. Plotting will be disabled.", file=sys.stderr)
    print("Install using: pip install matplotlib numpy", file=sys.stderr)


# --- Configuration ---
# Log file pattern to search for within the directory
LOG_FILE_PATTERN = "*.log"


# --- pgbench Log Parsing Function ---

def parse_pgbench_log(filepath):
    """Parses a single pgbench log file to extract key metrics."""
    metrics = {
        "scaling_factor": None,
        "query_mode": None, # string
        "clients": None,
        "threads": None,
        "duration_s": None,
        "transactions_processed": None,
        "transactions_failed_count": None,
        "transactions_failed_pct": None,
        "latency_avg_ms": None,
        "initial_conn_time_ms": None,
        "tps": None, # Transactions Per Second (excluding initial connection time)
        "error": None,
        "filepath": filepath
    }

    if not os.path.exists(filepath):
        metrics["error"] = f"Error: File not found at {filepath}"
        return metrics
    if not os.path.isfile(filepath):
        metrics["error"] = f"Error: Path is not a file {filepath}"
        return metrics

    try:
        # Use utf-8 with replace for robustness, as determined previously
        with open(filepath, "r", encoding='utf-8', errors='replace') as f:
            content = f.read()

        # --- Extracting Metrics using Regular Expressions ---

        match = re.search(r"scaling factor:\s+(\d+)", content)
        if match: metrics["scaling_factor"] = int(match.group(1))

        match = re.search(r"query mode:\s+(\w+)", content)
        if match: metrics["query_mode"] = match.group(1) # String

        match = re.search(r"number of clients:\s+(\d+)", content)
        if match: metrics["clients"] = int(match.group(1))

        match = re.search(r"number of threads:\s+(\d+)", content)
        if match: metrics["threads"] = int(match.group(1))

        match = re.search(r"duration:\s+(\d+)\s+s", content)
        if match: metrics["duration_s"] = int(match.group(1))

        match = re.search(r"number of transactions actually processed:\s+(\d+)", content)
        if match: metrics["transactions_processed"] = int(match.group(1))

        # number of failed transactions: 0 (0.000%)
        match = re.search(r"number of failed transactions:\s+(\d+)\s+\(([\d.]+)%\)", content)
        if match:
            metrics["transactions_failed_count"] = int(match.group(1))
            metrics["transactions_failed_pct"] = float(match.group(2))

        match = re.search(r"latency average\s*=\s*([\d.]+)\s+ms", content)
        if match: metrics["latency_avg_ms"] = float(match.group(1))

        match = re.search(r"initial connection time\s*=\s*([\d.]+)\s+ms", content)
        if match: metrics["initial_conn_time_ms"] = float(match.group(1))

        # tps = 1550.182177 (without initial connection time)
        match = re.search(r"tps\s*=\s*([\d.]+)\s+\(w", content) # Match 'tps =' up to '(w'
        if match: metrics["tps"] = float(match.group(1))

    except Exception as e:
        # Ensure error message uses only ASCII characters if possible
        err_msg = str(e).encode('ascii', 'replace').decode('ascii')
        metrics["error"] = f"An error occurred during parsing {filepath}: {err_msg}"

    # --- Basic Validation ---
    if metrics["tps"] is None or metrics["latency_avg_ms"] is None:
         if not metrics["error"]:
              metrics["error"] = f"Could not extract key pgbench metrics (TPS/Latency) from '{os.path.basename(filepath)}'"

    return metrics


# --- Aggregation Function ---

def aggregate_pgbench_results(all_metrics_data):
    """Calculates mean and standard deviation for metrics across multiple runs."""
    aggregated_results = {}
    valid_run_count = len(all_metrics_data)

    if valid_run_count == 0:
        return None

    # --- Identify Consistent String/Config Values ---
    query_modes = set(run_data.get("query_mode") for run_data in all_metrics_data if run_data and run_data.get("query_mode"))
    if len(query_modes) == 1:
        aggregated_results["query_mode"] = {"value": query_modes.pop(), "consistent": True, "count": valid_run_count}
    elif len(query_modes) > 1:
        # Count how many runs contributed to the varied modes
        mode_count = sum(1 for run_data in all_metrics_data if run_data and run_data.get("query_mode"))
        aggregated_results["query_mode"] = {"value": list(query_modes), "consistent": False, "count": mode_count}
    else:
        aggregated_results["query_mode"] = {"value": None, "consistent": False, "count": 0}

    # --- Aggregate Numeric Values ---
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
            if value is not None:
                collected_data[key].append(value)

    for metric_name, values in collected_data.items():
        count = len(values)
        is_config = metric_name in ["scaling_factor", "clients", "threads", "duration_s"]
        if count > 0:
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if count > 1 else 0.0
            consistent = is_config and (stdev < 1e-9 or count == 1)
            aggregated_results[metric_name] = {"mean": mean, "stdev": stdev, "count": count, "consistent": consistent}
        # else: don't add entry if no data was found for this metric

    if not any(key in aggregated_results for key in ["tps", "latency_avg_ms"]): # Check if essential results exist
        return None

    aggregated_results["valid_run_count"] = valid_run_count
    return aggregated_results


# --- Summary Printing Function ---

def print_pgbench_summary(agg_results, directory_path):
    """Prints the aggregated pgbench summary (mean +/- stdev) for a directory."""
    if not agg_results:
        print(f"\n--- No aggregated results to display for directory: {directory_path} ---", file=sys.stderr)
        return

    valid_runs = agg_results.get("valid_run_count", 0)
    print("\n" + "-" * 65)
    print(f"pgbench Aggregated Benchmark Summary")
    print(f"Directory Processed : {directory_path}")
    print(f"Log Files Aggregated: {valid_runs}")
    print("-" * 65)
    print("Configuration:")

    # Helper to format and print a metric line (ASCII safe)
    def print_line(label, data, unit="", precision=2, check_consistency=False, is_int=False):
        label_padded = f"  {label:<28}" # Pad label for alignment
        if data and data.get("mean") is not None: # Check if data exists for this metric
            mean = data["mean"]
            stdev = data["stdev"]
            count = data["count"]
            consistent = data.get("consistent", False)

            if check_consistency and consistent:
                # Use is_int flag for formatting consistent integer values
                value = int(mean) if is_int else f"{mean:.{precision}f}"
                print(f"{label_padded}: {value}{unit} (n={count}, consistent)")
            else:
                mean_str = f"{int(mean)}" if is_int else f"{mean:.{precision}f}"
                stdev_str = f"{int(stdev)}" if is_int else f"{stdev:.{precision}f}"
                consistency_note = " (inconsistent)" if check_consistency and not consistent and count > 1 else ""
                # Use +/- instead of the plus-minus symbol
                print(f"{label_padded}: {mean_str} +/- {stdev_str}{unit} (n={count}){consistency_note}")
        else:
            print(f"{label_padded}: N/A (n=0)")

    # Print Configuration Parameters
    print_line("Scaling Factor", agg_results.get("scaling_factor"), check_consistency=True, is_int=True)

    qm_data = agg_results.get("query_mode")
    label_padded_qm = f"  {'Query Mode':<28}"
    if qm_data and qm_data["value"] is not None:
         if qm_data["consistent"]:
             print(f"{label_padded_qm}: {qm_data['value']} (n={qm_data['count']}, consistent)")
         else:
             print(f"{label_padded_qm}: Varied {qm_data['value']} (n={qm_data['count']})")
    else:
         print(f"{label_padded_qm}: N/A (n=0)")

    print_line("Number of Clients", agg_results.get("clients"), check_consistency=True, is_int=True)
    print_line("Number of Threads", agg_results.get("threads"), check_consistency=True, is_int=True)
    print_line("Duration", agg_results.get("duration_s"), unit=" s", check_consistency=True, is_int=True)

    print("-" * 65)
    print("Results:")

    # Print Performance Metrics
    print_line("Transactions Processed", agg_results.get("transactions_processed"), precision=0, is_int=True)
    print_line("Failed Transactions (Count)", agg_results.get("transactions_failed_count"), precision=0, is_int=True)
    print_line("Failed Transactions (%)", agg_results.get("transactions_failed_pct"), unit="%", precision=3)
    print_line("Latency Average", agg_results.get("latency_avg_ms"), unit=" ms", precision=3)
    print_line("Initial Connection Time", agg_results.get("initial_conn_time_ms"), unit=" ms", precision=3)
    print_line("Transactions Per Second (TPS)", agg_results.get("tps"), precision=2)

    print("-" * 65)


# --- Plotting Function ---

def plot_pgbench_comparison(agg_data1, agg_data2, label1, label2):
    """
    Creates separate comparison plots for pgbench TPS, Latency, Transactions,
    and Connection Time. Saves plots as PDF.
    """
    if not PLOT_ENABLED:
        print("\nPlotting disabled as matplotlib/numpy are not installed.")
        return

    print("\nGenerating pgbench comparison plots...")

    label1_short = os.path.basename(label1.rstrip('/\\')) or "Dir 1"
    label2_short = os.path.basename(label2.rstrip('/\\')) or "Dir 2"

    # Helper function for creating individual bar plots
    def create_single_metric_plot(metric_key, title, ylabel, filename_suffix, precision=2):
        print(f" - Plotting '{metric_key}'...")
        d1_data = agg_data1.get(metric_key)
        d2_data = agg_data2.get(metric_key)

        if d1_data and d2_data:
            metrics_plot = [metric_key] # Single metric category
            means1 = [d1_data["mean"]]
            stdevs1 = [d1_data["stdev"]]
            means2 = [d2_data["mean"]]
            stdevs2 = [d2_data["stdev"]]

            x_indices = np.arange(len(metrics_plot))
            bar_width = 0.35

            fig, ax = plt.subplots(figsize=(6, 5)) # Standard size for single metric
            rects1 = ax.bar(x_indices - bar_width/2, means1, bar_width, yerr=stdevs1, label=label1_short, capsize=5, alpha=0.8)
            rects2 = ax.bar(x_indices + bar_width/2, means2, bar_width, yerr=stdevs2, label=label2_short, capsize=5, alpha=0.8)

            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xticks(x_indices)
            # Use a cleaned up label or just the metric key
            ax.set_xticklabels([metric_key.replace('_', ' ').replace(' ms', '').title()])
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            # Use specified precision for bar labels
            fmt_str = f'%.{precision}f'
            ax.bar_label(rects1, padding=3, fmt=fmt_str)
            ax.bar_label(rects2, padding=3, fmt=fmt_str)
            ax.set_ylim(bottom=0, top=max(means1 + means2) * 1.15 if max(means1 + means2) > 0 else 1) # Avoid zero ylim top

            fig.tight_layout()

            plot_filename = f"pgbench_comparison_{filename_suffix}.pdf"
            try:
                plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
                print(f"   Plot saved to '{plot_filename}'")
            except Exception as e:
                err_msg = str(e).encode('ascii', 'replace').decode('ascii')
                print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
            plt.close(fig)
        else:
             print(f"   Skipping plot for '{metric_key}': Missing data in one or both directories.")

    # --- Create the individual plots ---
    create_single_metric_plot("tps", "Comparison of Transactions Per Second", "TPS", "tps", precision=1)
    create_single_metric_plot("latency_avg_ms", "Comparison of Average Latency", "Latency (ms)", "latency", precision=3)
    create_single_metric_plot("initial_conn_time_ms", "Comparison of Initial Connection Time", "Time (ms)", "conn_time", precision=3)

    # --- Plot 4: Transaction Counts (Grouped Bar) ---
    transaction_metrics = ["transactions_processed", "transactions_failed_count"]
    print(f" - Plotting Transactions ({', '.join(transaction_metrics)})...")
    means1_txn, stdevs1_txn = [], []
    means2_txn, stdevs2_txn = [], []
    plot_txn_labels = []

    for metric in transaction_metrics:
        d1_txn = agg_data1.get(metric)
        d2_txn = agg_data2.get(metric)
        if d1_txn and d2_txn:
            means1_txn.append(d1_txn["mean"])
            stdevs1_txn.append(d1_txn["stdev"])
            means2_txn.append(d2_txn["mean"])
            stdevs2_txn.append(d2_txn["stdev"])
            plot_txn_labels.append(metric.replace('transactions_', '').replace('_count','').replace('_',' ').title())
        else:
            print(f"   Skipping transaction metric '{metric}' for plot: Missing data.")

    if plot_txn_labels: # Check if we have any transaction data to plot
        x_indices_txn = np.arange(len(plot_txn_labels))
        bar_width = 0.35

        fig_txn, ax_txn = plt.subplots(figsize=(max(6, len(plot_txn_labels) * 2.5), 6)) # Adjust width

        rects1_txn = ax_txn.bar(x_indices_txn - bar_width/2, means1_txn, bar_width,
                                yerr=stdevs1_txn, label=label1_short, capsize=5, alpha=0.8)
        rects2_txn = ax_txn.bar(x_indices_txn + bar_width/2, means2_txn, bar_width,
                                yerr=stdevs2_txn, label=label2_short, capsize=5, alpha=0.8)

        ax_txn.set_ylabel('Count')
        ax_txn.set_title('Comparison of Transaction Counts')
        ax_txn.set_xticks(x_indices_txn)
        ax_txn.set_xticklabels(plot_txn_labels)
        ax_txn.legend()
        ax_txn.grid(axis='y', linestyle='--', alpha=0.6)

        # Consider log scale if processed vs failed have vastly different scales
        # if max(means1_txn + means2_txn) / max(1, min(filter(None, means1_txn + means2_txn))) > 1000: # Heuristic
        #     ax_txn.set_yscale('log')
        #     ax_txn.set_ylabel('Count (Log Scale)')
        #     print("   Note: Using log scale for Y-axis in Transaction Counts plot.")

        # Add bar labels (integer format)
        ax_txn.bar_label(rects1_txn, padding=3, fmt='%.0f')
        ax_txn.bar_label(rects2_txn, padding=3, fmt='%.0f')

        ax_txn.set_ylim(bottom=0) # Ensure y-axis starts at 0 unless log scale
        if ax_txn.get_yscale() != 'log':
             ax_txn.set_ylim(top=max(means1_txn + means2_txn) * 1.15 if max(means1_txn + means2_txn) > 0 else 1)


        fig_txn.tight_layout()

        plot_filename = f"pgbench_comparison_transactions.pdf"
        try:
            plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
            print(f"   Plot saved to '{plot_filename}'")
        except Exception as e:
            err_msg = str(e).encode('ascii', 'replace').decode('ascii')
            print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
        plt.close(fig_txn)
    else:
         print(f"   Skipping Transactions plot: No common transaction metrics with data found.")


# --- Directory Processing Function ---

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
        # Pass back the count of files *used* in aggregation
        return aggregated_data, valid_files_parsed
    else:
        print(f"No valid data collected from directory '{directory_path}'.")
        return None, 0


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare pgbench benchmark logs from two directories, print summaries, and plot comparisons (ASCII only)."
    )
    parser.add_argument(
        "dir1",
        metavar="DIRECTORY_1",
        type=str,
        help="Path to the first directory containing pgbench benchmark *.log files.",
    )
    parser.add_argument(
        "dir2",
        metavar="DIRECTORY_2",
        type=str,
        help="Path to the second directory containing pgbench benchmark *.log files.",
    )
    args = parser.parse_args()

    # Process first directory
    agg_data1, count1 = process_directory(args.dir1)

    # Process second directory
    agg_data2, count2 = process_directory(args.dir2)

    # Print summaries if data was aggregated
    if agg_data1:
        print_pgbench_summary(agg_data1, args.dir1)
    if agg_data2:
        print_pgbench_summary(agg_data2, args.dir2)

    # Generate comparison plot if both directories yielded data
    if agg_data1 and agg_data2:
        plot_pgbench_comparison(agg_data1, agg_data2, args.dir1, args.dir2)
    elif PLOT_ENABLED:
        print("\nComparison plots cannot be generated as data from both directories is required.", file=sys.stderr)

    print("\nScript finished.")
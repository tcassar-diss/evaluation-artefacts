# No non-ASCII characters below (removed utf-8 coding line)
"""
Processes Redis benchmark CSV files from two directories, prints an
aggregated summary for each directory, and generates comparison plots.
Uses only ASCII characters.
"""

import argparse
import collections
import csv
import glob
import os
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
# CSV file pattern to search for within the directory
CSV_FILE_PATTERN = "*.csv"
# Expected numeric columns (based on example) - used for type conversion
NUMERIC_COLUMNS = [
    "rps",
    "avg_latency_ms",
    "min_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "max_latency_ms",
]
# Column name containing the test identifier
TEST_NAME_COLUMN = "test"
# Metrics to create comparison plots for (only used if plotting enabled)
METRICS_TO_PLOT = ["rps", "avg_latency_ms", "p99_latency_ms"]


# --- CSV Parsing Function ---


def parse_redis_csv(filepath):
    """
    Parses a single Redis benchmark CSV file.

    Returns:
        A dictionary where keys are test names and values are
        dictionaries of metrics for that test from this file,
        or None if parsing fails badly.
        Also returns the header list if successful.
    """
    results = {}
    header = None
    try:
        # Specify utf-8 encoding but replace errors for robustness
        # This is generally safe even in ASCII-only mode, as ASCII is a subset of UTF-8
        # If strict ASCII is needed even for input, use 'ascii' and errors='ignore'/'replace'
        # However, handling potentially varied input robustly is often better.
        with open(
            filepath, "r", newline="", encoding="utf-8", errors="replace"
        ) as csvfile:
            reader = csv.DictReader(csvfile)
            header = reader.fieldnames

            if not header or TEST_NAME_COLUMN not in header:
                print(
                    f"\n   Warning: Skipping file '{os.path.basename(filepath)}'. Invalid header or missing required column '{TEST_NAME_COLUMN}'.",
                    file=sys.stderr,
                )
                return None, None

            for row in reader:
                test_name = row.get(TEST_NAME_COLUMN)
                if not test_name:
                    continue

                metrics = {}
                for col in header:
                    if col == TEST_NAME_COLUMN:
                        continue
                    if col in row:
                        try:
                            if col in NUMERIC_COLUMNS:
                                metrics[col] = float(row[col])
                        except (ValueError, TypeError):
                            print(
                                f"\n   Warning: Could not convert value '{row[col]}' to float for metric '{col}' in test '{test_name}' in file '{os.path.basename(filepath)}'. Skipping metric.",
                                file=sys.stderr,
                            )
                            metrics[col] = None

                if any(
                    v is not None for k, v in metrics.items() if k in NUMERIC_COLUMNS
                ):
                    results[test_name] = metrics

        if not results:
            return None, header

        return results, header

    except FileNotFoundError:
        print(f"\n   Error: File not found: {filepath}", file=sys.stderr)
        return None, None
    except Exception as e:
        # Ensure error message uses only ASCII characters if possible
        err_msg = str(e).encode("ascii", "replace").decode("ascii")
        print(
            f"\n   Error processing CSV file '{os.path.basename(filepath)}': {err_msg}",
            file=sys.stderr,
        )
        return None, None


# --- Aggregation Function ---


def aggregate_redis_results(all_file_results):
    """Aggregates results from multiple files."""
    collected_data = collections.defaultdict(lambda: collections.defaultdict(list))
    valid_file_count = len(all_file_results)

    if valid_file_count == 0:
        return None

    for file_data in all_file_results:
        if not file_data:
            continue

        for test_name, metrics in file_data.items():
            for metric_name, value in metrics.items():
                if value is not None and metric_name in NUMERIC_COLUMNS:
                    collected_data[test_name][metric_name].append(value)

    aggregated_stats = collections.defaultdict(dict)
    for test_name, metrics_data in collected_data.items():
        for metric_name, values in metrics_data.items():
            count = len(values)
            if count > 0:
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if count > 1 else 0.0
                aggregated_stats[test_name][metric_name] = {
                    "mean": mean,
                    "stdev": stdev,
                    "count": count,
                }

    if not aggregated_stats:
        return None

    return aggregated_stats


# --- Summary Printing Function ---


def print_aggregated_summary(agg_results, directory_path, file_count, header_order):
    """Prints the aggregated Redis summary (mean +/- stdev) for a directory."""
    if not agg_results:
        print(
            f"\n--- No aggregated results to display for directory: {directory_path} ---",
            file=sys.stderr,
        )
        return

    print("\n" + "-" * 70)
    print(f"Redis Aggregated Benchmark Summary")
    print(f"Directory Processed : {directory_path}")
    print(f"CSV Files Aggregated: {file_count}")
    print("-" * 70)

    metric_order = (
        [h for h in header_order if h != TEST_NAME_COLUMN and h in NUMERIC_COLUMNS]
        if header_order
        else sorted(NUMERIC_COLUMNS)
    )

    sorted_test_names = sorted(agg_results.keys())

    for test_name in sorted_test_names:
        print(f"Test: {test_name}")
        test_metrics = agg_results[test_name]

        for metric_name in metric_order:
            data = test_metrics.get(metric_name)
            label_padded = f"  - {metric_name:<18}"

            if data:
                mean = data["mean"]
                stdev = data["stdev"]
                count = data["count"]
                precision = 3 if "latency" in metric_name else 2
                mean_str = f"{mean:.{precision}f}"
                stdev_str = f"{stdev:.{precision}f}"
                # Use +/- instead of the plus-minus symbol
                print(f"{label_padded}: {mean_str} +/- {stdev_str} (n={count})")
            # else: Metric not found for this test, skip printing N/A for cleaner output

        print("-" * 30)

    print("-" * 70)


# --- Plotting Function ---


def plot_comparison(agg_data1, agg_data2, label1, label2, header_order):
    """
    Creates comparison plots for specified metrics between two aggregated datasets.
    """
    if not PLOT_ENABLED:
        print("\nPlotting disabled as matplotlib/numpy are not installed.")
        return

    print("\nGenerating comparison plots...")

    label1_short = os.path.basename(label1.rstrip("/\\")) or "Dir 1"
    label2_short = os.path.basename(label2.rstrip("/\\")) or "Dir 2"

    common_tests = sorted(list(set(agg_data1.keys()) & set(agg_data2.keys())))

    if not common_tests:
        print(
            "Error: No common test names found between the two directories. Cannot generate plots.",
            file=sys.stderr,
        )
        return

    print(f"Found {len(common_tests)} common tests for comparison.")

    bar_width = 0.35
    metric_order = (
        [h for h in header_order if h != TEST_NAME_COLUMN and h in NUMERIC_COLUMNS]
        if header_order
        else sorted(NUMERIC_COLUMNS)
    )

    for metric in METRICS_TO_PLOT:
        if metric not in metric_order:
            continue

        print(f" - Plotting '{metric}'...")

        means1, stdevs1 = [], []
        means2, stdevs2 = [], []
        plot_test_labels = []

        for test_name in common_tests:
            d1_metric_data = agg_data1.get(test_name, {}).get(metric)
            d2_metric_data = agg_data2.get(test_name, {}).get(metric)

            if d1_metric_data and d2_metric_data:
                means1.append(d1_metric_data["mean"])
                stdevs1.append(d1_metric_data["stdev"])
                means2.append(d2_metric_data["mean"])
                stdevs2.append(d2_metric_data["stdev"])
                plot_test_labels.append(test_name)

        if not plot_test_labels:
            print(
                f"   Skipping plot for '{metric}': No common tests found with aggregated data for this metric in both directories."
            )
            continue

        x_indices_plot = np.arange(len(plot_test_labels))

        fig, ax = plt.subplots(figsize=(max(10, len(plot_test_labels) * 0.8), 6))

        rects1 = ax.bar(
            x_indices_plot - bar_width / 2,
            means1,
            bar_width,
            yerr=stdevs1,
            label=label1_short,
            capsize=5,
            alpha=0.8,
        )
        rects2 = ax.bar(
            x_indices_plot + bar_width / 2,
            means2,
            bar_width,
            yerr=stdevs2,
            label=label2_short,
            capsize=5,
            alpha=0.8,
        )

        metric_title = metric.replace("_", " ").title()
        ax.set_ylabel(metric_title)
        ax.set_title(f"Comparison of {metric_title} by Test Type")
        ax.set_xticks(x_indices_plot)
        ax.set_xticklabels(plot_test_labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        fig.tight_layout()

        plot_filename = f"plots/comparison_{metric}.pdf"
        try:
            plt.savefig(plot_filename, bbox_inches="tight")
            print(f"   Plot saved to '{plot_filename}'")
        except Exception as e:
            # Ensure error message uses only ASCII characters if possible
            err_msg = str(e).encode("ascii", "replace").decode("ascii")
            print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
        plt.close(fig)


# --- Directory Processing Function ---


def process_directory(directory_path):
    """Finds, parses, and aggregates CSV files in a directory."""
    print(f"\nProcessing directory: {directory_path}")

    if not os.path.isdir(directory_path):
        print(
            f"Error: Provided path '{directory_path}' is not a valid directory.",
            file=sys.stderr,
        )
        return None, None, 0

    search_pattern = os.path.join(directory_path, CSV_FILE_PATTERN)
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(
            f"Warning: No files matching pattern '{CSV_FILE_PATTERN}' found in directory '{directory_path}'.",
            file=sys.stderr,
        )
        return None, None, 0

    all_results = []
    first_header = None
    parse_errors = 0
    files_found_count = len(csv_files)

    print(
        f"Found {files_found_count} file(s) matching '{CSV_FILE_PATTERN}'. Parsing..."
    )

    for csv_file in csv_files:
        parsed_data, header = parse_redis_csv(csv_file)

        if parsed_data:
            all_results.append(parsed_data)
            if header and not first_header:
                first_header = header
        elif header is None and parsed_data is None:
            parse_errors += 1

    valid_files_parsed = len(all_results)
    print(f"Successfully parsed data from {valid_files_parsed} file(s).")
    if parse_errors > 0:
        print(
            f"Skipped or encountered errors in {parse_errors} file(s).", file=sys.stderr
        )

    if valid_files_parsed > 0:
        aggregated_data = aggregate_redis_results(all_results)
        return aggregated_data, first_header, valid_files_parsed
    else:
        print(f"No valid data collected from directory '{directory_path}'.")
        return None, None, 0


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Redis benchmark CSVs from two directories, print summaries, and plot comparisons (ASCII only)."
    )
    parser.add_argument(
        "dir1",
        metavar="DIRECTORY_1",
        type=str,
        help="Path to the first directory containing Redis benchmark *.csv files.",
    )
    parser.add_argument(
        "dir2",
        metavar="DIRECTORY_2",
        type=str,
        help="Path to the second directory containing Redis benchmark *.csv files.",
    )
    args = parser.parse_args()

    agg_data1, header1, count1 = process_directory(args.dir1)
    agg_data2, header2, count2 = process_directory(args.dir2)

    final_header_order = header1 if header1 else header2

    if agg_data1:
        print_aggregated_summary(agg_data1, args.dir1, count1, final_header_order)
    if agg_data2:
        print_aggregated_summary(agg_data2, args.dir2, count2, final_header_order)

    if agg_data1 and agg_data2:
        plot_comparison(agg_data1, agg_data2, args.dir1, args.dir2, final_header_order)
    elif PLOT_ENABLED:
        print(
            "\nComparison plots cannot be generated as data from both directories is required.",
            file=sys.stderr,
        )

    print("\nScript finished.")

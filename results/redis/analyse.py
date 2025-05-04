# No non-ASCII characters below
"""
Processes Redis benchmark CSV files from two directories, prints a
comparison summary including percentage differences (Dir2 vs Dir1) to
stdout and ./summary.txt, and generates comparison plots saved as PDF files.
Uses only ASCII characters.
"""

import argparse
import collections
import csv
import glob
import math  # Needed for checking isnan/isinf
import os
import statistics
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns

font = {"size": 18}

plt.rc("font", **font)
plt.style.use("science")

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
# These will be plotted for absolute values AND percentage difference
METRICS_TO_PLOT = ["rps", "avg_latency_ms", "p99_latency_ms"]
# Output file for the text summary
SUMMARY_FILENAME = "summary.txt"
# Threshold for baseline mean close to zero to avoid division issues
ZERO_THRESHOLD = 1e-9


# --- CSV Parsing Function ---
# (parse_redis_csv remains the same as the provided script)
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
        err_msg = str(e).encode("ascii", "replace").decode("ascii")
        print(
            f"\n   Error processing CSV file '{os.path.basename(filepath)}': {err_msg}",
            file=sys.stderr,
        )
        return None, None


# --- Aggregation Function ---
# (aggregate_redis_results remains the same as the provided script)
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


# --- Comparison Summary Generation Function (NEW) ---


def generate_comparison_summary(results1, results2, label1, label2, header_order):
    """Generates a comparison summary string including percentage difference."""
    if not results1 or not results2:
        return None  # Cannot generate if data is missing

    label1_short = os.path.basename(label1.rstrip("/\\")) or "Baseline"
    label2_short = os.path.basename(label2.rstrip("/\\")) or "Comparison"

    all_tests = sorted(list(set(results1.keys()) | set(results2.keys())))
    metric_order = (
        [h for h in header_order if h != TEST_NAME_COLUMN and h in NUMERIC_COLUMNS]
        if header_order
        else sorted(NUMERIC_COLUMNS)
    )

    summary_lines = []
    separator = "-" * 95
    header_line1 = "=" * 95
    header_line2 = "Redis Benchmark Comparison Summary"
    header_line3 = f"Baseline (Dir1): {label1_short}"
    header_line4 = f"Comparison (Dir2): {label2_short}"
    header_line5 = "=" * 95
    col_header_format = "{:<30} | {:<25} | {:<25} | {:<10}"
    col_headers = col_header_format.format(
        "Test / Metric",
        f"{label1_short} (Mean+/-Stdev)",
        f"{label2_short} (Mean+/-Stdev)",
        "% Diff",
    )

    summary_lines.extend(
        [
            header_line1,
            header_line2,
            header_line3,
            header_line4,
            header_line5,
            col_headers,
            separator,
        ]
    )

    for test_name in all_tests:
        summary_lines.append(
            f"{test_name:<30} | {'':<25} | {'':<25} | "
        )  # Test name line
        res1_test = results1.get(test_name, {})
        res2_test = results2.get(test_name, {})

        for metric in metric_order:
            res1 = res1_test.get(metric)
            res2 = res2_test.get(metric)

            def format_res(res):
                if res:
                    precision = 3 if "latency" in metric else 2
                    mean_str = f"{res['mean']:.{precision}f}"
                    stdev_str = f"{res['stdev']:.{precision}f}"
                    count = res["count"]
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
                # else: leave as N/A if baseline is zero/tiny and comparison is not

            metric_label = f"  - {metric:<26}"  # Indent metric name
            summary_lines.append(
                col_header_format.format(
                    metric_label, res1_str, res2_str, percent_diff_str
                )
            )

        summary_lines.append(separator)  # Separator after each test

    return "\n".join(summary_lines)


# --- Plotting Function (Modified to add % diff plot) ---


def plot_comparison(agg_data1, agg_data2, label1, label2, header_order):
    """
    Creates comparison plots for specified metrics (absolute and % diff)
    between two aggregated datasets. Saves plots as PDF.
    """
    if not PLOT_ENABLED:
        print("\nPlotting disabled as matplotlib/numpy are not installed.")
        return
    if not agg_data1 or not agg_data2:
        print(
            "\nCannot generate plots due to missing aggregated data.", file=sys.stderr
        )
        return

    print("\nGenerating comparison plots...")

    label1_short = os.path.basename(label1.rstrip("/\\")) or "Baseline"
    label2_short = os.path.basename(label2.rstrip("/\\")) or "Comparison"

    common_tests = sorted(list(set(agg_data1.keys()) & set(agg_data2.keys())))

    if not common_tests:
        print(
            "Error: No common test names found between the two directories. Cannot generate plots.",
            file=sys.stderr,
        )
        return

    print(f"Found {len(common_tests)} common tests for comparison.")

    # Create output directory if it doesn't exist
    output_plot_dir = f"./{label1_short}-plots"
    if not os.path.exists(output_plot_dir):
        try:
            os.makedirs(output_plot_dir)
            print(f"Created plot output directory: {output_plot_dir}")
        except OSError as e:
            print(
                f"Error: Could not create plot directory '{output_plot_dir}': {e}",
                file=sys.stderr,
            )
            output_plot_dir = "."  # Save in current dir as fallback

    bar_width = 0.35
    metric_order = (
        [h for h in header_order if h != TEST_NAME_COLUMN and h in NUMERIC_COLUMNS]
        if header_order
        else sorted(NUMERIC_COLUMNS)
    )

    percent_diffs_by_metric = collections.defaultdict(list)
    plot_labels_p = []  # Labels (test names) for the % diff plot

    # --- Plots for Absolute Values ---
    for metric in METRICS_TO_PLOT:
        if metric not in metric_order:
            continue

        print(f" - Plotting Absolute '{metric}'...")

        means1, stdevs1 = [], []
        means2, stdevs2 = [], []
        plot_test_labels_abs = []  # Test names used in this specific plot

        for test_name in common_tests:
            d1_metric_data = agg_data1.get(test_name, {}).get(metric)
            d2_metric_data = agg_data2.get(test_name, {}).get(metric)

            if d1_metric_data and d2_metric_data:
                means1.append(d1_metric_data["mean"])
                stdevs1.append(d1_metric_data["stdev"])
                means2.append(d2_metric_data["mean"])
                stdevs2.append(d2_metric_data["stdev"])
                plot_test_labels_abs.append(test_name)

                # While iterating, calculate % diff for later plot
                mean1 = d1_metric_data["mean"]
                mean2 = d2_metric_data["mean"]
                percent_diff = float("nan")  # Default to NaN
                if abs(mean1) > ZERO_THRESHOLD:
                    diff = ((mean2 - mean1) / mean1) * 100.0
                    if not math.isnan(diff) and not math.isinf(diff):
                        percent_diff = diff
                elif abs(mean2) < ZERO_THRESHOLD:
                    percent_diff = 0.0
                percent_diffs_by_metric[metric].append(percent_diff)

        if not plot_test_labels_abs:
            print(
                f"   Skipping absolute plot for '{metric}': No common tests found with data."
            )
            continue

        # Rebuild percent diff data just for tests included in the absolute plot
        # (This ensures alignment if some tests were skipped)
        current_percent_diffs = []
        final_plot_labels_p = []  # Reset labels for %diff specific to plotted data
        for i, test_name in enumerate(plot_test_labels_abs):
            # Find index in original common_tests to get corresponding %diff
            original_index = common_tests.index(test_name)
            if metric in percent_diffs_by_metric and original_index < len(
                percent_diffs_by_metric[metric]
            ):
                diff_val = percent_diffs_by_metric[metric][original_index]
                current_percent_diffs.append(diff_val)
                final_plot_labels_p.append(test_name)
            else:  # Should not happen if logic is correct, but safety check
                current_percent_diffs.append(float("nan"))
                final_plot_labels_p.append(test_name)

        # Plot absolute values
        x_indices_plot = np.arange(len(plot_test_labels_abs))
        fig, ax = plt.subplots(figsize=(max(10, len(plot_test_labels_abs) * 0.8), 6))
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
        ax.set_xticklabels(plot_test_labels_abs, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        plot_filename = os.path.join(output_plot_dir, f"redis_comparison_{metric}.pdf")
        try:
            plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
            print(f"   Absolute plot saved to '{plot_filename}'")
        except Exception as e:
            err_msg = str(e).encode("ascii", "replace").decode("ascii")
            print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
        plt.close(fig)

        # Store calculated percent diffs for the final plot
        # Ensure alignment if tests were skipped for absolute plot
        if metric in METRICS_TO_PLOT:  # Only store if it's needed for final plot
            percent_diffs_by_metric[metric] = current_percent_diffs
            # Use the labels from the last successfully plotted metric for the %diff x-axis
            plot_labels_p = final_plot_labels_p

    # --- Plot 4: Percentage Difference ---
    print(f" - Plotting Percentage Difference ({', '.join(METRICS_TO_PLOT)})...")

    # Check if we have any valid labels and corresponding data for the %diff plot
    if not plot_labels_p or not any(
        metric in percent_diffs_by_metric for metric in METRICS_TO_PLOT
    ):
        print(
            "   Skipping Percentage Difference plot: No common data found for selected metrics."
        )
        return  # Exit plotting function if no data

    x_indices_p = np.arange(len(plot_labels_p))
    num_metrics_p = len(METRICS_TO_PLOT)
    total_width = 0.8  # Total width allocated for bars per group
    single_bar_width = total_width / num_metrics_p

    fig_p, ax_p = plt.subplots(
        figsize=(max(10, len(plot_labels_p) * 1.2), 6)
    )  # Make wider potentially

    # Calculate offsets for each metric's bar within a group
    offsets = np.linspace(
        -total_width / 2 + single_bar_width / 2,
        total_width / 2 - single_bar_width / 2,
        num_metrics_p,
    )

    plotted_metrics_legend = {}  # To store handles for legend

    for i, metric in enumerate(METRICS_TO_PLOT):

        if metric == "p99_latency_ms":
            continue

        if metric in percent_diffs_by_metric:
            # Filter out NaNs for this specific metric before plotting
            valid_indices = [
                idx
                for idx, val in enumerate(percent_diffs_by_metric[metric])
                if not math.isnan(val)
            ]
            valid_x = x_indices_p[valid_indices]
            valid_diffs = [
                percent_diffs_by_metric[metric][idx] for idx in valid_indices
            ]

            if valid_diffs:  # Only plot if there's valid data
                rects = ax_p.bar(
                    valid_x + offsets[i],
                    valid_diffs,
                    single_bar_width,
                    label=metric.replace("_ms", " (ms)"),
                    alpha=0.8,
                )
                plotted_metrics_legend[metric.replace("_ms", " (ms)")] = rects

    ax_p.set_ylabel(r"Percentage Difference (\%)")
    ax_p.set_title(f"Redis Performance % Difference ({label2_short} vs {label1_short})")
    ax_p.set_xticks(x_indices_p)
    ax_p.set_xticklabels(plot_labels_p, rotation=45, ha="right")
    # Use handles collected for legend

    ax_p.legend(
        handles=plotted_metrics_legend.values(),
        labels=plotted_metrics_legend.keys(),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        # You might also want ncol=1 if you have many metrics
        # to force a single vertical column:
        # ncol=1
    )

    ax_p.grid(axis="y", linestyle="--", alpha=0.6)
    ax_p.axhline(0, color="grey", linewidth=0.8)  # Line at 0%

    fig_p.tight_layout()
    plot_filename = os.path.join(output_plot_dir, "redis_comparison_percent_diff.pdf")
    try:
        plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
        print(f"   Percentage Difference plot saved to '{plot_filename}'")
    except Exception as e:
        err_msg = str(e).encode("ascii", "replace").decode("ascii")
        print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
    plt.close(fig_p)


# --- Directory Processing Function ---
# (process_directory remains the same as the provided script)
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
        description="Compare Redis benchmark CSVs from two directories, print summaries with % diff, save summary, and plot comparisons (ASCII only)."
    )
    parser.add_argument(
        "dir1",
        metavar="BASELINE_DIR",  # Changed label
        type=str,
        help="Path to the baseline directory containing Redis benchmark *.csv files.",
    )
    parser.add_argument(
        "dir2",
        metavar="COMPARISON_DIR",  # Changed label
        type=str,
        help="Path to the comparison directory containing Redis benchmark *.csv files.",
    )
    args = parser.parse_args()

    # Process directories
    agg_data1, header1, count1 = process_directory(args.dir1)
    agg_data2, header2, count2 = process_directory(args.dir2)

    final_header_order = header1 if header1 else header2

    # Generate and handle comparison summary
    summary_text = None
    if agg_data1 and agg_data2:
        summary_text = generate_comparison_summary(
            agg_data1, agg_data2, args.dir1, args.dir2, final_header_order
        )

    if summary_text:
        SUMMARY_FILENAME = f"./{args.dir1}-{SUMMARY_FILENAME}"
        print("\n" + summary_text)  # Print summary to stdout
        try:
            with open(SUMMARY_FILENAME, "w", encoding="ascii") as f_summary:
                f_summary.write(summary_text)
            print(f"\nComparison summary saved to '{SUMMARY_FILENAME}'")
        except IOError as e:
            print(
                f"\nError: Could not write summary to file '{SUMMARY_FILENAME}': {e}",
                file=sys.stderr,
            )
    else:
        print(
            "\nComparison summary cannot be generated due to missing data.",
            file=sys.stderr,
        )

    # Generate comparison plots if both directories yielded data
    if agg_data1 and agg_data2:
        plot_comparison(agg_data1, agg_data2, args.dir1, args.dir2, final_header_order)
    elif PLOT_ENABLED:
        print(
            "\nComparison plots cannot be generated as data from both directories is required.",
            file=sys.stderr,
        )

    print("\nScript finished.")

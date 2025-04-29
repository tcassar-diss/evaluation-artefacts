# Ensure only ASCII characters are used in this script
"""
Processes two NASA NPB benchmark CSV files, calculates statistics
(mean +/- stdev) for each benchmark within each file, prints a
comparison summary, and generates a comparison bar plot saved as a PDF file.
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


# --- CSV Processing Function for one file ---


def process_npb_csv(filepath):
    """
    Reads one NPB CSV file and calculates mean/stdev for each benchmark
    across the runs within that file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        A dictionary where keys are microbenchmark names and values are
        dictionaries containing {"mean": float, "stdev": float, "count": int}.
        Returns None on critical error.
    """
    # results: benchmark_name -> {"mean": ..., "stdev": ..., "count": ...}
    results = {}
    print(f"--> Processing file: {filepath}")

    try:
        # Use utf-8 with replace for robustness
        with open(
            filepath, "r", newline="", encoding="utf-8", errors="replace"
        ) as csvfile:
            reader = csv.reader(csvfile)
            try:
                header = next(reader)  # Read the header row
            except StopIteration:
                print(
                    f"   Error: Skipping empty file '{os.path.basename(filepath)}'.",
                    file=sys.stderr,
                )
                return None  # Indicate failure for this file

            if not header or len(header) < 2:
                print(
                    f"   Error: Skipping file '{os.path.basename(filepath)}'. Invalid header or no run columns.",
                    file=sys.stderr,
                )
                return None

            benchmark_col_index = 0
            run_col_indices = list(range(1, len(header)))

            file_row_count = 0
            benchmarks_processed = set()
            for row_num, row in enumerate(reader, start=2):
                if not row:
                    continue

                if len(row) <= benchmark_col_index:
                    print(
                        f"   Warning: Skipping row {row_num} in '{os.path.basename(filepath)}': Not enough columns.",
                        file=sys.stderr,
                    )
                    continue

                benchmark_name = row[benchmark_col_index].strip()
                if not benchmark_name:
                    print(
                        f"   Warning: Skipping row {row_num} in '{os.path.basename(filepath)}': Empty benchmark name.",
                        file=sys.stderr,
                    )
                    continue

                # Collect run values for this benchmark from this row/file
                run_values = []
                for col_index in run_col_indices:
                    if col_index < len(row):
                        value_str = row[col_index].strip()
                        try:
                            value_float = float(value_str)
                            run_values.append(value_float)
                        except (ValueError, TypeError):
                            print(
                                f"   Warning: Invalid numeric value '{value_str}' for benchmark '{benchmark_name}' in file '{os.path.basename(filepath)}', row {row_num}, column {col_index + 1}. Skipping value.",
                                file=sys.stderr,
                            )
                    # else: handles rows shorter than header

                # Calculate stats if we have values
                count = len(run_values)
                if count > 0:
                    mean = statistics.mean(run_values)
                    stdev = statistics.stdev(run_values) if count > 1 else 0.0
                    results[benchmark_name] = {
                        "mean": mean,
                        "stdev": stdev,
                        "count": count,
                    }
                    benchmarks_processed.add(benchmark_name)
                file_row_count += 1

            if file_row_count == 0 or not benchmarks_processed:
                print(
                    f"   Warning: No valid benchmark data processed from '{os.path.basename(filepath)}'."
                )
                return None  # No usable data

            print(
                f"   Processed {len(benchmarks_processed)} benchmarks from this file."
            )
            return results

    except FileNotFoundError:
        print(f"   Error: File not found: {filepath}", file=sys.stderr)
        return None
    except Exception as e:
        err_msg = str(e).encode("ascii", "replace").decode("ascii")
        print(
            f"   Error processing CSV file '{os.path.basename(filepath)}': {err_msg}",
            file=sys.stderr,
        )
        return None


# --- Summary Printing Function ---


def print_comparison_summary(results1, results2, label1, label2):
    """Prints a side-by-side comparison summary."""
    if not results1 or not results2:
        print("\nCannot print comparison summary due to missing data.")
        return

    label1_short = os.path.basename(label1.rstrip("/\\")) or "File 1"
    label2_short = os.path.basename(label2.rstrip("/\\")) or "File 2"

    common_benchmarks = sorted(list(set(results1.keys()) & set(results2.keys())))
    all_benchmarks = sorted(list(set(results1.keys()) | set(results2.keys())))

    print("\n" + "=" * 78)
    print("NASA NPB Benchmark Comparison Summary")
    print(f"File 1: {label1_short}")
    print(f"File 2: {label2_short}")
    print("=" * 78)
    # Header line with dynamic padding based on label lengths
    header_format = "{:<15} | {:<25} | {:<25}"
    print(
        header_format.format(
            "Microbenchmark",
            f"{label1_short} (Mean+/-Stdev)",
            f"{label2_short} (Mean+/-Stdev)",
        )
    )
    print("-" * 78)

    for benchmark in all_benchmarks:
        res1 = results1.get(benchmark)
        res2 = results2.get(benchmark)

        def format_res(res):
            if res:
                mean_str = f"{res['mean']:.2f}"
                stdev_str = f"{res['stdev']:.2f}"
                count = res["count"]
                return f"{mean_str} +/- {stdev_str} (n={count})"
            else:
                return "N/A"

        res1_str = format_res(res1)
        res2_str = format_res(res2)

        print(header_format.format(benchmark, res1_str, res2_str))

    print("-" * 78)


# --- Plotting Function ---


def plot_comparison(results1, results2, label1, label2):
    """
    Creates a grouped bar plot comparing the mean performance of each benchmark
    between the two files. Saves plot as PDF.
    """
    if not PLOT_ENABLED:
        print("\nPlotting disabled as matplotlib/numpy are not installed.")
        return
    if not results1 or not results2:
        print("\nCannot generate plot due to missing results data.", file=sys.stderr)
        return

    print("\nGenerating NPB comparison plot...")

    label1_short = os.path.basename(label1.rstrip("/\\")) or "File 1"
    label2_short = os.path.basename(label2.rstrip("/\\")) or "File 2"

    # Find common benchmarks present in both datasets
    common_benchmarks = sorted(list(set(results1.keys()) & set(results2.keys())))

    if not common_benchmarks:
        print(
            "Error: No common benchmarks found between the two files. Cannot generate plot.",
            file=sys.stderr,
        )
        return

    print(f"Found {len(common_benchmarks)} common benchmarks for comparison.")

    # Prepare data for plotting
    means1 = [results1[b]["mean"] for b in common_benchmarks]
    stdevs1 = [results1[b]["stdev"] for b in common_benchmarks]
    means2 = [results2[b]["mean"] for b in common_benchmarks]
    stdevs2 = [results2[b]["stdev"] for b in common_benchmarks]
    plot_labels = common_benchmarks  # Use benchmark names as labels

    x_indices = np.arange(len(plot_labels))  # the label locations
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(plot_labels) * 0.8), 6))  # Adjust width

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

    # Add some text for labels, title and axes ticks
    ax.set_ylabel("Million Operations per Second")  # Generic label
    ax.set_title("NASA NPB Microbenchmark Comparison")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(plot_labels, rotation=45, ha="right")  # Rotate labels
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Add bar labels (optional)
    # ax.bar_label(rects1, padding=3, fmt='%.1f')
    # ax.bar_label(rects2, padding=3, fmt='%.1f')

    # Adjust y-axis limit
    all_means = means1 + means2
    ax.set_ylim(bottom=0, top=max(all_means) * 1.15 if all_means else 1)

    fig.tight_layout()  # Adjust layout

    # Save the plot as PDF with tight bounding box
    plot_filename = "npb_comparison.pdf"
    try:
        plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
        print(f"   Plot saved to '{plot_filename}'")
    except Exception as e:
        # Ensure error message uses only ASCII characters if possible
        err_msg = str(e).encode("ascii", "replace").decode("ascii")
        print(f"   Error saving plot '{plot_filename}': {err_msg}", file=sys.stderr)
    plt.close(fig)  # Close the figure to free memory


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare NASA NPB benchmark statistics from two CSV files, print summary, and plot results (ASCII only)."
    )
    parser.add_argument(
        "csv_files",
        metavar="FILE",
        type=str,
        nargs=2,  # Expect exactly TWO file paths
        help="Paths to the two NPB benchmark *.csv files to compare.",
    )
    args = parser.parse_args()

    file1_path = args.csv_files[0]
    file2_path = args.csv_files[1]

    # Validate file existence (basic check)
    if not os.path.isfile(file1_path):
        print(
            f"Error: Input file not found or is not a file: {file1_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.path.isfile(file2_path):
        print(
            f"Error: Input file not found or is not a file: {file2_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Process each file individually
    results1 = process_npb_csv(file1_path)
    results2 = process_npb_csv(file2_path)

    # Proceed only if both files were processed successfully
    if results1 and results2:
        # Print the comparison summary table
        print_comparison_summary(results1, results2, file1_path, file2_path)

        # Generate the comparison plot
        plot_comparison(results1, results2, file1_path, file2_path)
    else:
        print(
            "\nAborting comparison due to errors during file processing.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nScript finished.")

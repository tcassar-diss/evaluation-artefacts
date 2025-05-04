import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # Ensure this library is installed (`pip install scienceplots`)
import seaborn as sns
import os # For path manipulation
import glob # To find files matching a pattern
import matplotlib.ticker as ticker # Import ticker for locator control

# --- User Configuration ---
# Automatically find all CSV files in the current directory
csv_files = glob.glob('*.csv')
SAMPLE_SIZE = 1000 # Number of samples for the time breakdown plot
RANDOM_SEED = 42 # Seed for reproducible random sampling (optional, set to None for different samples each run)
Y_AXIS_TICK_THRESHOLD = 100 # Hide y-axis ticks if number of bars exceeds this
IQR_MULTIPLIER = 1.5 # Multiplier for IQR calculation to define outlier bounds
X_AXIS_PADDING = 1.05 # Padding factor for the shared x-axis limit
# --- End User Configuration ---


# Apply requested styling
font = {"size": 18}
plt.rc("font", **font)
plt.style.use("science") # Ensure the 'science' style is available

# Store mean data for the final plot
all_means_data = {}
global_max_time_clipped = 0 # Initialize global max time after clipping

# Check if any CSV files were found
if not csv_files:
    print("Error: No CSV files found in the current directory.")
else:
    print("--- First Pass: Calculating Global Max Time (Post-Clipping) ---")
    # --- First Pass: Determine the maximum x-axis limit needed after clipping ---
    for i, file_path in enumerate(csv_files):
        print(f"Analyzing file {i+1}/{len(csv_files)} for max time: {file_path}")
        try:
            df_pass1 = pd.read_csv(file_path)
            if df_pass1.empty:
                print(f"  Warning: File {file_path} is empty. Skipping.")
                continue

            # Convert to numeric
            try:
                numeric_df_pass1 = df_pass1.apply(pd.to_numeric)
            except ValueError:
                print(f"  Warning: File {file_path} contains non-numeric data. Skipping max time calculation.")
                continue

            numeric_df_pass1 = numeric_df_pass1.select_dtypes(include=np.number).copy()
            if numeric_df_pass1.empty:
                 print(f"  Warning: No numeric data found in {file_path}. Skipping max time calculation.")
                 continue

            # --- Sampling (Consistent with second pass) ---
            if len(numeric_df_pass1) > SAMPLE_SIZE:
                numeric_df_pass1 = numeric_df_pass1.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)

            # --- Outlier Clipping (Consistent with second pass) ---
            total_times_pass1 = numeric_df_pass1.sum(axis=1)
            if total_times_pass1.empty: # Check if sampling resulted in empty df
                 print(f"  Warning: Sampling resulted in empty data for {file_path}. Skipping max time calculation.")
                 continue

            q1_pass1 = total_times_pass1.quantile(0.25)
            q3_pass1 = total_times_pass1.quantile(0.75)
            iqr_pass1 = q3_pass1 - q1_pass1
            lower_bound_pass1 = q1_pass1 - IQR_MULTIPLIER * iqr_pass1
            upper_bound_pass1 = q3_pass1 + IQR_MULTIPLIER * iqr_pass1

            # Filter based on bounds
            clipped_times_pass1 = total_times_pass1[(total_times_pass1 >= lower_bound_pass1) & (total_times_pass1 <= upper_bound_pass1)]

            if not clipped_times_pass1.empty:
                max_time_in_file = clipped_times_pass1.max()
                global_max_time_clipped = max(global_max_time_clipped, max_time_in_file)
                print(f"  Max clipped time in this file: {max_time_in_file:.2f} ns")
            else:
                print(f"  Warning: No data left after clipping in {file_path}. Skipping max time update.")

        except Exception as e:
            print(f"  An error occurred during first pass for {file_path}: {e}")

    print(f"\n--- Global Maximum Clipped Syscall Time: {global_max_time_clipped:.2f} ns ---")
    if global_max_time_clipped == 0:
        print("Warning: Global max time is 0. Plots might have unexpected x-axis limits.")

    print("\n--- Second Pass: Generating Plots ---")
    # --- Second Pass: Process files again for plotting ---
    # Store original header order for consistent plotting later
    header_order = None
    # Sort csv_files to ensure consistent processing order for the final plot
    csv_files.sort()
    for i, file_path in enumerate(csv_files):
        base_filename = os.path.splitext(os.path.basename(file_path))[0] # Get filename without extension
        print(f"\n--- Processing File {i+1}/{len(csv_files)}: {file_path} ---")

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            # Store header order from the first valid file
            if header_order is None and not df.empty:
                 header_order = df.columns.tolist()

            # --- Validate Data ---
            if df.empty:
                print(f"Warning: File {file_path} is empty or contains only headers. Skipping.")
                continue
            try:
                numeric_df_full = df.apply(pd.to_numeric)
            except ValueError as e:
                print(f"Error: File {file_path} contains non-numeric data that cannot be analyzed numerically. Skipping.")
                print(f"Details: {e}")
                continue

            # --- Calculate and Print Statistics (on FULL dataset) ---
            print("\nColumn Statistics (min, max, diff) on FULL dataset:")
            stats_df = numeric_df_full.select_dtypes(include=np.number)
            if stats_df.empty:
                print("  No numeric columns found for statistics.")
            else:
                for col in stats_df.columns:
                    col_min = stats_df[col].min()
                    col_max = stats_df[col].max()
                    col_diff = col_max - col_min
                    print(f"  {col}:")
                    print(f"    Min:  {col_min} ns")
                    print(f"    Max:  {col_max} ns")
                    print(f"    Diff: {col_diff} ns")

            # --- Calculate and Store Means (on FULL dataset) ---
            print("\nCalculating mean times for each stage...")
            means_df = numeric_df_full.select_dtypes(include=np.number)
            if not means_df.empty:
                file_means = means_df.mean()
                # Ensure means are stored in the original header order if possible
                if header_order:
                    try:
                        # Use fill_value=0 for stages potentially missing in a file
                        file_means = file_means.reindex(header_order, fill_value=0)
                    except Exception as reindex_e:
                         print(f"  Warning: Could not reindex means for {file_path} to header order. Error: {reindex_e}")
                all_means_data[os.path.basename(file_path)] = file_means
                print("  Means calculated.")
            else:
                print("  No numeric columns found for mean calculation.")


            # --- Prepare Data for Time Breakdown Plot (Sampling, Clipping) ---
            numeric_df_plot = numeric_df_full.select_dtypes(include=np.number).copy() # Work on a copy
            num_rows_original = len(numeric_df_plot)

            if numeric_df_plot.empty:
                print(f"\nWarning: No numeric columns found in {file_path} to plot. Skipping time breakdown plot.")
                continue # Skip to next file if no numeric data

            # --- Sampling ---
            is_sampled = False
            if num_rows_original > SAMPLE_SIZE:
                print(f"\nSampling {SAMPLE_SIZE} rows for time breakdown plot...")
                numeric_df_plot = numeric_df_plot.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
                is_sampled = True
            else:
                print(f"\nUsing all {num_rows_original} rows for time breakdown plot (less than sample size).")

            num_rows_after_sampling = len(numeric_df_plot)

            # --- Outlier Clipping (based on total time per row) ---
            print("Clipping outliers based on total time per syscall (IQR method)...")
            total_times = numeric_df_plot.sum(axis=1)
            if total_times.empty: # Check if sampling resulted in empty df
                 print(f"  Warning: Sampling resulted in empty data for {file_path}. Skipping plot generation.")
                 continue

            q1 = total_times.quantile(0.25)
            q3 = total_times.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - IQR_MULTIPLIER * iqr
            upper_bound = q3 + IQR_MULTIPLIER * iqr

            # Filter rows based on bounds
            original_count = len(numeric_df_plot)
            numeric_df_plot = numeric_df_plot[(total_times >= lower_bound) & (total_times <= upper_bound)]
            clipped_count = original_count - len(numeric_df_plot)
            # Avoid division by zero if original_count was 0 (though unlikely here)
            percentage_clipped = (clipped_count / original_count * 100) if original_count > 0 else 0
            print(f"  Removed {clipped_count} outliers ({percentage_clipped:.1f}% of plotted data).")

            num_rows_to_plot = len(numeric_df_plot) # Final count for plotting

            if numeric_df_plot.empty:
                 print(f"Warning: No data left after outlier clipping. Skipping time breakdown plot for {file_path}.")
                 continue # Skip plot if clipping removed everything

            # --- NO Normalization (per row) ---
            # We are now plotting absolute times

            # --- Generate and Save Stacked Bar Chart (Sampled, Clipped, Absolute Time) ---
            print(f"Generating time breakdown plot for {file_path}...")
            fig1, ax1 = plt.subplots(figsize=(12, 8)) # Adjust size as needed

            numeric_df_plot.plot(kind='barh', stacked=True, ax=ax1, width=0.8) # Use width for bar thickness

            # Customize the plot
            ax1.set_xlabel("Time (ns)") # Updated label
            ax1.set_ylabel("Sampled Syscall Index (Post-Clipping)") # Updated label
            plot_title = f"Syscall Time Breakdown - {os.path.basename(file_path)}"
            if is_sampled:
                plot_title = f"Sampled Syscall Time Breakdown - {os.path.basename(file_path)}"
            ax1.set_title(plot_title + " (Clipped)") # Add clipped indicator

            # Place legend outside the plot area
            ax1.legend(title="Filtering Stage", bbox_to_anchor=(1.05, 1), loc='upper left')

            # Set x-axis limits using the pre-calculated global max
            ax1.set_xlim(0, global_max_time_clipped * X_AXIS_PADDING if global_max_time_clipped > 0 else None) # Avoid setting limit if max is 0

            # --- Add logic to hide y-axis ticks if too many bars ---
            if num_rows_to_plot > Y_AXIS_TICK_THRESHOLD:
                print(f"Hiding y-axis tick labels for {num_rows_to_plot} bars to improve performance.")
                ax1.yaxis.set_major_locator(ticker.NullLocator()) # Hide major ticks
            # --- End of y-axis tick logic ---

            # Define filename for the time breakdown plot
            time_plot_filename = f"{base_filename}_time_breakdown_clipped.pdf"
            if is_sampled:
                 time_plot_filename = f"{base_filename}_time_breakdown_sampled_clipped.pdf"

            # Save the plot as PDF with tight bounding box
            plt.savefig(time_plot_filename, bbox_inches='tight')
            print(f"Saved time breakdown plot to: {time_plot_filename}")
            plt.close(fig1) # Close the figure to free memory


        except FileNotFoundError:
            print(f"Error: File not found at {file_path}. Skipping.")
        except pd.errors.EmptyDataError:
             print(f"Warning: File {file_path} is empty or contains only headers. Skipping.")
        except Exception as e:
            # Catch other potential errors during processing/plotting
            print(f"An unexpected error occurred while processing {file_path}: {e}. Skipping.")
            # Attempt to close any potentially open plot window in case of error
            plt.close('all')

    # --- Generate and Save Mean Comparison Plot (Horizontal Stacked Bar Chart) ---
    print("\n--- Generating Mean Comparison Plot (Horizontal Stacked) ---")
    if not all_means_data:
        print("No mean data collected. Skipping mean comparison plot.")
    elif header_order is None:
        print("Could not determine header order. Skipping mean comparison plot.")
    else:
        try:
            # Convert collected means into a DataFrame (Files as keys/columns, Stages as index)
            mean_df = pd.DataFrame(all_means_data)

            # Ensure the DataFrame index (stages) matches the desired header order
            try:
                 # Use fill_value=0 for stages potentially missing in a file when reindexing columns
                mean_df = mean_df.reindex(header_order, axis='index', fill_value=0)
            except Exception as reindex_e:
                print(f"  Warning: Could not reindex DataFrame stages to header order. Plotting with potentially incorrect stage order. Error: {reindex_e}")

            # Transpose the DataFrame: Files become the index, Stages become columns
            mean_df_transposed = mean_df.T # Index=Files, Columns=Stages

            # Create the horizontal stacked bar chart
            fig_means, ax_means = plt.subplots(figsize=(12, 8)) # Adjust size as needed
            # Plot the transposed DataFrame as a horizontal stacked bar chart
            mean_df_transposed.plot(kind='barh', stacked=True, ax=ax_means, width=0.8)

            # Customize the plot
            ax_means.set_title("Mean Syscall Time Breakdown Across Files")
            ax_means.set_ylabel("File") # Files are now on the y-axis
            ax_means.set_xlabel("Total Mean Time (ns)") # x-axis is the stacked total time
            # No rotation needed for y-axis labels
            # Legend now represents the stages
            ax_means.legend(title="Filtering Stage", bbox_to_anchor=(1.05, 1), loc='upper left')

            # Save the plot
            mean_plot_filename = "mean_stacked_horizontal_comparison.pdf" # Updated filename
            plt.savefig(mean_plot_filename, bbox_inches='tight')
            print(f"Saved mean comparison plot to: {mean_plot_filename}")
            plt.close(fig_means) # Close the figure

        except Exception as e:
            print(f"An error occurred while generating the mean comparison plot: {e}")


    print("\n--- Processing Complete ---")
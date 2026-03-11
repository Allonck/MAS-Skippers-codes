#!/usr/bin/env python3

import argparse
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import os
import re # For regular expressions to parse TRIMSEC
from astropy.time import Time
from astropy import units as u # For Time calculations (if dummy files were used, but now removed)

# --- Helper Function to obtain FITS files ---
def obtain_fits_files(path, filtering="", ends_with=False, NOT=False):
    """
    Scans a directory for FITS files, optionally filtering by filename.

    Args:
        path (str): The directory to scan.
        filtering (str): Substring to filter filenames (e.g., "darksDGM2_NOAB").
                         If empty, no filename filtering is applied.
        ends_with (bool): If True, filter files ending with the filter string.
        NOT (bool): If True, exclude files matching the filter string.

    Returns:
        list: A list of full paths to the filtered FITS files.
    """
    print(f"Scanning directory: {path}")
    all_files = []
    if os.path.exists(path):
        all_files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.fit', '.fits'))]
    else:
        print(f"Error: Directory '{path}' does not exist.")
        return [] # Return empty list if directory doesn't exist

    filtered_files = []
    for f in all_files:
        basename = os.path.basename(f)
        if filtering: # Apply filter only if a filter string is provided
            matches_filter = (filtering in basename) if not ends_with else basename.endswith(filtering)
            if (matches_filter and not NOT) or (not matches_filter and NOT):
                filtered_files.append(f)
        else: # If no filter string, include all FITS files
            filtered_files.append(f)
            
    return filtered_files


def analyze_cte_trail(file_path, extension_number, n_previous_cols, n_trailing_cols, 
                      y_roi_range=None, last_active_column_override=None): # Added override argument
    """
    Analyzes the mean pixel value across columns around the last active zone to evaluate CTE.

    Args:
        file_path (str): Path to the FITS file.
        extension_number (int): The FITS extension containing the image data and relevant headers.
        n_previous_cols (int): Number of columns BEFORE the last active column to include in analysis.
        n_trailing_cols (int): Number of columns AFTER the last active column (overscan) to analyze.
        y_roi_range (list, optional): [y_start, y_end] for the vertical ROI. If None, uses full image height.
        last_active_column_override (int, optional): Manually specify the 0-indexed last active column.
                                                     If provided, overrides TRIMSEC parsing.

    Returns:
        tuple: (column_indices, mean_values, last_active_col_0_based) or (None, None, None) if an error occurs.
    """
    try:
        with fits.open(file_path) as hdul:
            hdr_ext = hdul[extension_number].header
            image_data = hdul[extension_number].data

            if image_data is None:
                print(f"Error: No data in extension {extension_number} of {file_path}.")
                return None, None, None

            last_active_col_0_based = None
            if last_active_column_override is not None:
                last_active_col_0_based = last_active_column_override
                print(f"Using provided last active column: {last_active_col_0_based} (0-indexed)")
            else:
                # Parse TRIMSEC to find the end of the active area (fallback if no override)
                trimsec_str = hdr_ext.get('TRIMSEC')
                if trimsec_str is None:
                    print(f"Warning: 'TRIMSEC' not found in header of extension {extension_number} for {file_path}. Cannot determine active column end. Assuming full image width as active.")
                    active_x_end_1_based = image_data.shape[1] 
                else:
                    match = re.match(r'\[(\d+):(\d+),(\d+):(\d+)\]', trimsec_str)
                    if match:
                        active_x_end_1_based = int(match.group(2))
                    else:
                        print(f"Warning: Could not parse 'TRIMSEC' format ('{trimsec_str}') in {file_path}. Assuming full image width as active.")
                        active_x_end_1_based = image_data.shape[1]
                last_active_col_0_based = active_x_end_1_based - 1


            # Define the range of columns to analyze
            start_col_to_analyze_0_based = max(0, last_active_col_0_based - n_previous_cols)
            end_col_to_analyze_0_based = min(image_data.shape[1] - 1, last_active_col_0_based + n_trailing_cols)
            
            if start_col_to_analyze_0_based >= image_data.shape[1] or \
               end_col_to_analyze_0_based < start_col_to_analyze_0_based:
                print(f"Warning: Calculated column range ({start_col_to_analyze_0_based}:{end_col_to_analyze_0_based}) is invalid for image shape {image_data.shape} in {file_path}. Skipping.")
                return None, None, None

            # Define the vertical ROI for mean calculation
            if y_roi_range:
                y_start, y_end = y_roi_range
                y_start = max(0, y_start)
                y_end = min(image_data.shape[0], y_end)
                if y_end <= y_start:
                    print(f"Warning: Invalid vertical ROI ({y_roi_range}) for {file_path}. Using full height.")
                    y_start, y_end = 0, image_data.shape[0]
            else:
                y_start, y_end = 0, image_data.shape[0]

            cols_data = image_data[y_start:y_end, start_col_to_analyze_0_based : end_col_to_analyze_0_based + 1]
            mean_values_per_column = np.mean(cols_data, axis=0)
            column_indices = np.arange(start_col_to_analyze_0_based, end_col_to_analyze_0_based + 1)

            print(f"Analyzed {len(column_indices)} columns from {start_col_to_analyze_0_based} to {end_col_to_analyze_0_based} for {os.path.basename(file_path)}")
            return column_indices, mean_values_per_column, last_active_col_0_based

    except Exception as e:
        print(f"An error occurred while analyzing CTE trail for {file_path}: {e}")
        return None, None, None


def plot_fits_characteristics(output_plot_filename, path, extension_number, 
                              n_previous_cols_cte=10, n_trailing_cols_cte=15, y_roi_cte=None,
                              file_filter="", last_active_column=None):
    """
    Processes FITS files to analyze CTE by plotting mean pixel value vs column position.

    Adds the CTE estimate (value of first overscan pixel divided by 512) next to each filename in the legend.
    """
    cte_data_to_plot = []

    path_files = obtain_fits_files(path, filtering=file_filter, NOT=False)

    if not path_files:
        print(f"No FITS files found in '{path}' with the specified filter '{file_filter}'.")
        return

    print(f"Processing {len(path_files)} FITS files from '{path}' for CTE analysis...")

    for file_path in path_files:
        col_indices, mean_vals, last_active_col = analyze_cte_trail(
            file_path, extension_number,
            n_previous_cols_cte, n_trailing_cols_cte,
            y_roi_range=y_roi_cte,
            last_active_column_override=last_active_column
        )

        if col_indices is not None and mean_vals is not None:
            # Estimar CTE: valor del primer píxel posterior al área activa
            relative_index = last_active_col + 1 - col_indices[0]
            if 0 <= relative_index < len(mean_vals):
                cte_pixel = mean_vals[relative_index]
                cte = cte_pixel / 512
                label = f"{os.path.basename(file_path)} (CTE={cte:.5f})"
            else:
                cte = None
                label = f"{os.path.basename(file_path)} (CTE=NaN)"

            cte_data_to_plot.append((label, col_indices, mean_vals, last_active_col))
            print(f"Collected CTE trail data for '{label}'")

    # --- Plotting ---
    if len(cte_data_to_plot) > 0:
        fig, ax_cte = plt.subplots(1, 1, figsize=(10, 6))

        ax_cte.set_title('Mean Pixel Value vs Column Position (CTE Trail)')
        ax_cte.set_xlabel('Pixel Column Position (0-indexed)')
        ax_cte.set_ylabel('Mean Pixel Value (in ROI)')
        ax_cte.grid(True, linestyle='--', alpha=0.7)

        if cte_data_to_plot:
            first_last_active_col = cte_data_to_plot[0][3]
            if first_last_active_col is not None:
                ax_cte.axvline(x=first_last_active_col, color='black', linestyle=':',
                               label='Last Active Column (0-indexed)')
                ax_cte.legend()

        for file_label, col_indices, mean_vals, _ in cte_data_to_plot:
            ax_cte.plot(col_indices, mean_vals, marker='o', linestyle='-', label=file_label)

        if len(cte_data_to_plot) <= 20:
            ax_cte.legend(title="FITS File")
        else:
            print("Too many files for CTE plot legend; legend omitted.")

        plt.tight_layout()
        plt.savefig(output_plot_filename)
        print(f"Plot saved as {output_plot_filename}")
        plt.show()
    else:
        print("No valid CTE trail data collected to generate plots.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots of CTE trail from FITS files.")
    parser.add_argument("--output_filename", type=str, default="cte_trail_plot.png",
                        help="Name of the output plot file.")
    parser.add_argument("--path", type=str, default=".",
                        help="Path where the .fits files are located. (Defaults to current directory)")
    parser.add_argument("--extension_number", type=int, default=1,
                        help="FITS extension number containing the image data and headers (0 for primary).")
    parser.add_argument("--n_previous_cols_cte", type=int, default=10,
                        help="Number of columns BEFORE the last active column to include in CTE analysis.")
    parser.add_argument("--n_trailing_cols_cte", type=int, default=15,
                        help="Number of columns AFTER the last active column (overscan) to include in CTE analysis.")
    parser.add_argument("--y_roi_cte_start", type=int, default=None,
                        help="Start Y-index for vertical ROI for CTE analysis (0-indexed). If None, uses full image height.")
    parser.add_argument("--y_roi_cte_end", type=int, default=None,
                        help="End Y-index for vertical ROI for CTE analysis (0-indexed). If None, uses full image height.")
    parser.add_argument("--filter", type=str, default="",
                        help="Substring to filter FITS filenames (e.g., 'darksDGM2_NOAB').")
    parser.add_argument("--last_active_column", type=int, default=539,
                        help="Manually specify the 0-indexed last active column. Overrides TRIMSEC parsing if provided.")
    
    args = parser.parse_args()

    # Assemble y_roi_cte_range if provided
    y_roi_cte_range = None
    if args.y_roi_cte_start is not None and args.y_roi_cte_end is not None:
        y_roi_cte_range = [args.y_roi_cte_start, args.y_roi_cte_end]

    plot_fits_characteristics(
        output_plot_filename=args.output_filename,
        path=args.path,
        extension_number=args.extension_number,
        n_previous_cols_cte=args.n_previous_cols_cte,
        n_trailing_cols_cte=args.n_trailing_cols_cte,
        y_roi_cte=y_roi_cte_range,
        file_filter=args.filter,
        last_active_column=args.last_active_column
    )


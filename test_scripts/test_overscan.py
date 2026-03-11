#!/usr/bin/env python3

import argparse
from astropy.io import fits
import numpy as np
import os

def roi_shifting(roi):
    extensions = [1, 14, 16, 15, 13, 11, 12, 10, 5, 2, 4, 3, 9, 6, 8, 7]
    shifted_roi = []
    for idx, ext in enumerate(extensions):
        x_start = roi[0] + 15 * (ext - 1)
        x_end = roi[1] + 15 * (ext - 1)
        y_start, y_end = roi[2], roi[3]
        roi_shifted = [x_start, x_end, y_start, y_end]
        shifted_roi.append(roi_shifted)
    return shifted_roi

def overscan_correction_combined(file, output_file, roi_vector, method='mean'):
    """
    Applies overscan correction to a multi-extension FITS file using individual ROIs per extension.

    Args:
        file (str): Path to input FITS file.
        output_file (str): Path to save corrected FITS file.
        roi_vector (list): List of ROIs, one per extension, each as [col_start, col_end, row_start, row_end].
        method (str): 'mean' (default) or 'poly' for polynomial fit.

    Returns:
        None. Writes a corrected FITS file.
    """

    if len(roi_vector) < 1:
        raise ValueError("ROI vector must contain at least one region.")

    with fits.open(file, mode='readonly') as hdul:
        corrected_hdul = fits.HDUList([fits.PrimaryHDU(header=hdul[0].header.copy())])

        for ext in range(1, len(hdul)):
            if hdul[ext].data is None:
                print(f"⚠️ Warning: Extension {ext} has no data. Skipping.")
                continue

            if ext - 1 >= len(roi_vector):
                print(f"⚠️ Warning: No ROI defined for extension {ext}. Skipping.")
                continue

            data = hdul[ext].data.astype(float)
            col_start, col_end, row_start, row_end = roi_vector[ext - 1]

            if (row_end > data.shape[0]) or (col_end > data.shape[1]):
                print(f"⚠️ ROI out of bounds for extension {ext}. Skipping.")
                corrected_hdul.append(fits.ImageHDU(data=data, header=hdul[ext].header))
                continue

            overscan_region = data[row_start:row_end, col_start:col_end]

            if overscan_region.size == 0:
                print(f"⚠️ Overscan region empty for extension {ext}. Skipping correction.")
                corrected_hdul.append(fits.ImageHDU(data=data, header=hdul[ext].header))
                continue

            if method == 'mean':
                overscan_value = np.mean(overscan_region)
            elif method == 'poly':
                x = np.arange(col_end - col_start)
                y = np.mean(overscan_region, axis=0)
                poly_coeffs = np.polyfit(x, y, deg=2)
                poly_fit = np.polyval(poly_coeffs, x)
                overscan_value = np.mean(poly_fit)
            else:
                raise ValueError("Method must be 'mean' or 'poly'.")

            corrected_data = data - overscan_value
            corrected_hdul.append(fits.ImageHDU(data=corrected_data, header=hdul[ext].header))

            print(f"✅ Ext {ext}: Overscan = {overscan_value:.3f}")

    corrected_hdul.writeto(output_file, overwrite=True)
    print(f"✅ Saved corrected file: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply overscan correction to multi-extension FITS file using shifted ROIs.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input FITS file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save corrected FITS file.")
    parser.add_argument("--roi", type=int, nargs=4, required=True, metavar=("X_START", "X_END", "Y_START", "Y_END"),
                        help="Base ROI to shift across extensions: col_start col_end row_start row_end")
    parser.add_argument("--method", type=str, choices=["mean", "poly"], default="mean", help="Correction method: mean or poly fit.")

    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        raise FileNotFoundError(f"Input FITS file '{args.input_file}' not found.")

    base_roi = args.roi
    roi_vector = roi_shifting(base_roi)
    overscan_correction_combined(args.input_file, args.output_file, roi_vector, method=args.method)


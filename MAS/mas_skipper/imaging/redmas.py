#redmas = reduce mas
import numpy as np
from astropy.io import fits
#from ..core import core

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

            # Verify the limits
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
                poly_coef = np.polyfit(x, y, deg=2)
                poly_fit = np.polyval(poly_coef, x)
                overscan_value = np.mean(poly_fit)
            else:
                raise ValueError("Method must be 'mean' or 'poly'.")

            corrected_data = data - overscan_value
            corrected_hdul.append(fits.ImageHDU(data=corrected_data, header=hdul[ext].header))

            print(f"✅ Ext {ext}: Overscan = {overscan_value:.3f}")

    corrected_hdul.writeto(output_file, overwrite=True)
    print(f"✅ Saved corrected file: {output_file}")

def bias_subtraction(path_files, master_bias_file, output_file):
    """
    Applies bias subtraction to all 16 extensions in a FITS file.

    This function reads a science FITS file and subtracts the corresponding extensions
    of a master bias file to remove the electronic bias pattern from each image extension.

    Args:
        path_files (str): Path to the input FITS file (science image).
        master_bias_file (str): Path to the master bias FITS file.
        output_file (str): Path to save the bias-corrected FITS file.

    Returns:
        None

    Raises:
        ValueError: If the input files have mismatched extensions or data is missing.

    Notes:
        - Assumes that both the science and master bias files contain 16 image extensions.
        - Skips any extension that is missing or contains no data in either file.
        - Preserves original headers in each extension.
    """
    with fits.open(path_files, mode='readonly') as sci_hdul, fits.open(master_bias_file, mode='readonly') as bias_hdul:
        # Copy the original primary header
        corrected_hdul = fits.HDUList([fits.PrimaryHDU(header=sci_hdul[0].header.copy())])

        for ext in range(1, 17):  # Iterate through the 16 extensions
            if ext >= len(sci_hdul) or sci_hdul[ext].data is None:
                print(f"Warning: Extension {ext} does not exist in {path_files}. Passing...")
                continue

            if ext >= len(bias_hdul) or bias_hdul[ext].data is None:
                print(f"Warning: Extension {ext} does not exist in {master_bias_file}. Passing...")
                continue

            # Read science and bias data
            science_data = sci_hdul[ext].data.astype(float)
            bias_data = bias_hdul[ext].data.astype(float)

            # Perform bias subtraction
            corrected_data = science_data - bias_data

            # Create a new FITS extension
            corrected_hdu = fits.ImageHDU(data=corrected_data, header=sci_hdul[ext].header)
            corrected_hdul.append(corrected_hdu)

            print(f"Bias subtracted for extension {ext}")

        # Save the bias-corrected FITS file
        corrected_hdul.writeto(output_file, overwrite=True)
        print(f"Bias-corrected FITS saved as: {output_file}")
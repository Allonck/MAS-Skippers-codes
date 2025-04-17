import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from matplotlib.patches import Rectangle #Importación corregida
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from itertools import combinations

def obtain_path_files(path, ends_with=True, filtering=".fits", NOT=False):
    """
    Displays a single extension of a FITS file as an image.

    Args:
        path (str): The path to the FITS files.
        ends_with (bool): Criteria to apply filter. Starts with or Ends with.
        filtering (str, optional): String to search in the directory. Defaults to ".fits".
        NOT (bool, optional): If True, exclude files matching the filter. Defaults to False.

    Returns:
        path_files (list)

    Example:
        >>> obtain_path_files(path="/data/ccd", ends_with=True, filtering="o_", NOT=False)
        >>> obtain_path_files(path="/data/ccd", ends_with=True, filtering="b.fits", NOT=True)

    Notes:
        - It will filter all the files in the directory, selecting only the ones that match or do not match the filter.
        - If NOT is True, it will select files that DO NOT match the filter.
    """
    if ends_with is True:
        if NOT is False:
            file_names = sorted([f for f in os.listdir(path) if f.endswith(filtering)])
        else:
            file_names = sorted([f for f in os.listdir(path) if not f.endswith(filtering)])
    elif ends_with is False:
        if NOT is False:
            file_names = sorted([f for f in os.listdir(path) if f.startswith(filtering)])
        else:
            file_names = sorted([f for f in os.listdir(path) if not f.startswith(filtering)])
    else:
        print(f"Bool variable {ends_with} is wrong.")
        return []

    # Create a list of full file paths
    path_files = [os.path.join(path, file_name) for file_name in file_names]
    print(f"Found {len(path_files)} files")
    return path_files
    
def generate_output_paths(input_paths, suffix="_corrected"):
    """
    Generates output file paths by adding a suffix to the input file names.

    Args:
        input_paths (list): List of input file paths.
        suffix (str, optional): Suffix to add to the file names. Defaults to "_corrected".

    Returns:
        output_paths (list): List of output file paths.
    """
    output_paths = []
    for input_path in input_paths:
        directory, filename = os.path.split(input_path)
        name, extension = os.path.splitext(filename)
        output_filename = f"{name}{suffix}{extension}"
        output_path = os.path.join(directory, output_filename)
        output_paths.append(output_path)
    return output_paths
    
def show_fits_image(filename, index = 1, cmap="gray"):
    """
    Displays a single extension of a FITS file as an image.

    Args:
        filename (str): The path to the FITS file.
        index (int, optional): The extension index to display. Defaults to 1.
        cmap (str, optional): The colormap to use for the image. Defaults to 'hot'.

    Returns:
        None

    Example:
        >>> show_fits_image("example.fits", index=2, cmap="viridis")

    Notes:
        The image intensity is scaled using `astropy.visualization.ZScaleInterval` 
        for optimal visualization. A colorbar is added to indicate the intensity range.
    """
    with fits.open(filename) as hdulist:
        image_data = hdulist[index].data

    zscale = ZScaleInterval()
    zlow, zhigh = zscale.get_limits(image_data)

    fig, ax = plt.subplots(figsize=(16, 9))

    im = ax.imshow(image_data, cmap=cmap, clim=(zlow, zhigh))

    # Create colorbar with same height as the y-axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1.5%", pad=0.05)
    fig = ax.figure
    fig.colorbar(im, cax=cax)
    
def new_gain_dynamic_ROI_single_extension(path_files, roi, extension_number, min_points=5):
    """
    Grafica V(M1' - M1) vs M1 + M1' para una extensión específica del MAS-CCD Skipper y muestra el ROI.

    NOTA: ROI DINÁMICO
    """
    gains = {}
    exposure_times = defaultdict(list)
    all_rois = []  # List to store all ROIs

    for file_path in path_files:
        with fits.open(file_path) as hdulist:
            exptime = hdulist[0].header.get('EXPTIME', None)
            if exptime is None:
                continue
            exposure_times[exptime].append(file_path)

    extension_variances = []
    extension_sum_of_means = []
    exposure_time_keys = sorted(exposure_times.keys())

    for exptime in exposure_time_keys:
        files = exposure_times[exptime]
        if len(files) >= 2:
            file1, file2 = files[:2]  # Take the first two files
            print(f"file1: {file1}, file2: {file2}")
            with fits.open(file1) as hdul1, fits.open(file2) as hdul2:
                if extension_number >= len(hdul1) or extension_number >= len(hdul2):
                    continue
                data1 = hdul1[extension_number].data
                data2 = hdul2[extension_number].data
                if data1 is None or data2 is None:
                    continue
                if roi:
                    x_start, x_end, y_start, y_end = roi
                    data1_roi = data1[y_start:y_end, x_start:x_end]
                    data2_roi = data2[y_start:y_end, x_start:x_end]
                    diff_data = data2_roi - data1_roi
                    variance = np.var(diff_data)
                    sum_of_mean = np.mean(data2_roi) + np.mean(data1_roi)

                    if np.isnan(variance) or np.isnan(sum_of_mean):
                        continue
                    extension_variances.append(variance)
                    extension_sum_of_means.append(sum_of_mean)
                    all_rois.append((data1_roi, data2_roi, file1, file2))

    # Create a single figure for all ROIs
    num_rois = len(all_rois)
    if num_rois > 0:
        rows = 6
        cols = 4
        fig_rois, axes = plt.subplots(rows, cols, figsize=(20, 16))
        axes = axes.flatten()

        for i, (data1_roi, data2_roi, file1, file2) in enumerate(all_rois):
            zscale1 = ZScaleInterval()
            zlow1, zhigh1 = zscale1.get_limits(data1_roi)
            axes[2*i].imshow(data1_roi, cmap="gray", clim=(zlow1, zhigh1), extent=[x_start, x_end, y_end, y_start]) #Set extent
            axes[2*i].invert_yaxis() #Invert Y axis
            contid1 = fits.open(file1)[extension_number].header.get('CONTID', 'N/A')
            exptime1 = fits.open(file1)[0].header.get('EXPTIME', 'N/A')
            axes[2*i].set_title(f"ROI {os.path.basename(file1)}, EXPTIME={exptime1}", fontsize=8)
            axes[2*i].text(0.95, 0.05, f"{contid1}", transform=axes[2*i].transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right', alpha=0.8, color ="red")

            zscale2 = ZScaleInterval()
            zlow2, zhigh2 = zscale2.get_limits(data2_roi)
            axes[2*i+1].imshow(data2_roi, cmap="gray", clim=(zlow2, zhigh2), extent=[x_start, x_end, y_end, y_start]) #Set extent
            axes[2*i+1].invert_yaxis() #Invert Y axis
            contid2 = fits.open(file2)[extension_number].header.get('CONTID', 'N/A')
            exptime2 = fits.open(file2)[0].header.get('EXPTIME', 'N/A')
            axes[2*i+1].set_title(f"ROI {os.path.basename(file2)}, EXPTIME={exptime2}", fontsize=8)
            axes[2*i+1].text(0.95, 0.05, f"{contid2}", transform=axes[2*i+1].transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right', alpha=0.8, color="red")

        fig_rois.tight_layout()
        fig_rois.show()

    if len(extension_variances) < 4:
        print(f"No suficientes puntos para la extensión {extension_number}")
        plt.close()
        return None

    extension_variances = np.array(extension_variances)
    extension_sum_of_means = np.array(extension_sum_of_means)

    slope, intercept, x_best, y_best, best_indices = best_gain_fit(extension_variances, extension_sum_of_means, min_points)
    if slope is None:
        print(f"No se pudo realizar el ajuste para la extensión {extension_number}")
        plt.close()
        return None

    gains[extension_number] = 1 / slope

    fig_gain, ax_gain = plt.subplots(figsize=(8, 6))
    ax_gain.plot(extension_variances, extension_sum_of_means, 'o', label=f'Ext {extension_number}', color="blue", markersize=5)
    fit_line = np.polyval([slope, intercept], extension_variances)
    ax_gain.plot(extension_variances, fit_line, 'r--', label=f'Fit (Gain={slope:.3f})')
    gain_text = f"Gain: {1 / slope:.3f} e-/ADU"
    ax_gain.text(0.05, 0.95, gain_text, transform=ax_gain.transAxes, fontsize=10, color='red', verticalalignment='top')
    ax_gain.set_xlabel("Varianza de la diferencia")
    ax_gain.set_ylabel("Suma de Medias")
    ax_gain.grid()
    fig_gain.tight_layout()
    fig_gain.show()

    return list(gains.values())

def new_full_well_dynamic_ROI_single_extension(path_files, roi, extension_number, min_points=5, threshold=0.01):
    """
    Grafica Tiempo de exposición vs Cuentas promedio para una extensión específica del MAS-CCD Skipper,
    muestra el ROI, realiza un ajuste lineal y encuentra el full well.

    NOTA: ROI DINÁMICO
    """
    exposure_times = defaultdict(list)
    all_rois = []  # List to store all ROIs
    extension_exptimes = []
    extension_means = []

    for file_path in path_files:
        with fits.open(file_path) as hdulist:
            exptime = hdulist[0].header.get('EXPTIME', None)
            if exptime is None:
                continue
            exposure_times[exptime].append(file_path)

    exposure_time_keys = sorted(exposure_times.keys())

    for exptime in exposure_time_keys:
        files = exposure_times[exptime]
        if len(files) >= 2:
            file1, file2 = files[:2]  # Take the first two files
            with fits.open(file1) as hdul1, fits.open(file2) as hdul2:
                if extension_number >= len(hdul1) or extension_number >= len(hdul2):
                    continue
                data1 = hdul1[extension_number].data
                data2 = hdul2[extension_number].data
                if data1 is None or data2 is None:
                    continue
                if roi:
                    x_start, x_end, y_start, y_end = roi
                    data1_roi = data1[y_start:y_end, x_start:x_end]
                    data2_roi = data2[y_start:y_end, x_start:x_end]
                    mean_data = (np.mean(data1_roi) + np.mean(data2_roi)) / 2
                    extension_exptimes.append(exptime)
                    extension_means.append(mean_data)
                    all_rois.append((data1_roi, data2_roi, file1, file2))

    # Create a single figure for all ROIs
    num_rois = len(all_rois)
    if num_rois > 0:
        rows = 6
        cols = 4
        fig_rois, axes = plt.subplots(rows, cols, figsize=(20, 16))
        axes = axes.flatten()

        for i, (data1_roi, data2_roi, file1, file2) in enumerate(all_rois):
            zscale1 = ZScaleInterval()
            zlow1, zhigh1 = zscale1.get_limits(data1_roi)
            axes[2*i].imshow(data1_roi, cmap="gray", clim=(zlow1, zhigh1), extent=[x_start, x_end, y_end, y_start]) #Set extent
            axes[2*i].invert_yaxis() #Invert Y axis
            contid1 = fits.open(file1)[extension_number].header.get('CONTID', 'N/A')
            exptime1 = fits.open(file1)[0].header.get('EXPTIME', 'N/A')
            axes[2*i].set_title(f"ROI {os.path.basename(file1)}, EXPTIME={exptime1}", fontsize=8)
            axes[2*i].text(0.95, 0.05, f"{contid1}", transform=axes[2*i].transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right', alpha=0.8, color ="red")

            zscale2 = ZScaleInterval()
            zlow2, zhigh2 = zscale2.get_limits(data2_roi)
            axes[2*i+1].imshow(data2_roi, cmap="gray", clim=(zlow2, zhigh2), extent=[x_start, x_end, y_end, y_start]) #Set extent
            axes[2*i+1].invert_yaxis() #Invert Y axis
            contid2 = fits.open(file2)[extension_number].header.get('CONTID', 'N/A')
            exptime2 = fits.open(file2)[0].header.get('EXPTIME', 'N/A')
            axes[2*i+1].set_title(f"ROI {os.path.basename(file2)}, EXPTIME={exptime2}", fontsize=8)
            axes[2*i+1].text(0.95, 0.05, f"{contid2}", transform=axes[2*i+1].transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right', alpha=0.8, color="red")

        fig_rois.tight_layout()
        fig_rois.show()

    if len(extension_exptimes) < 4:
        print(f"No suficientes puntos para la extensión {extension_number}")
        plt.close()
        return None

    # Fit Lineal y Full Well
    slope, intercept, good_exptimes, good_means, _ = best_gain_fit(np.array(extension_exptimes), np.array(extension_means), min_points)

    if slope is None:
        return None

    fit_line = np.polyval([slope, intercept], good_exptimes)
    residuals = np.abs(good_means - fit_line) / good_means

    full_well_index = np.where(residuals > threshold)[0][0] if np.any(residuals > threshold) else len(good_exptimes) - 1
    full_well = good_means[full_well_index]

    fig_counts, ax_counts = plt.subplots(figsize=(8, 6))
    ax_counts.plot(extension_exptimes, extension_means, 'o', label=f'Ext {extension_number}', color="red", markersize=5)
    ax_counts.plot(good_exptimes, fit_line, 'b--', label='Ajuste Lineal')
    ax_counts.axvline(x=good_exptimes[full_well_index], color='g', linestyle='--', label='Full Well')

    ax_counts.set_title(f"Extensión {extension_number}")
    ax_counts.set_xlabel("Tiempo de Exposición (EXPTIME)")
    ax_counts.set_ylabel("Media de cuentas")
    ax_counts.grid()
    ax_counts.legend()
    fig_counts.tight_layout()
    fig_counts.show()

    return full_well, extension_means

def visualize_roi__mean_variance(file_path, roi, extension_number):
    """
    Visualiza un ROI en una imagen FITS y muestra la media de las cuentas dentro del ROI.

    Args:
        file_path (str): Ruta al archivo FITS.
        roi (tuple): Región de interés (ROI) como (x_start, x_end, y_start, y_end).
        extension_number (int): Número de extensión del archivo FITS a utilizar.
    """
    try:
        with fits.open(file_path) as hdul:
            data = hdul[extension_number].data
            if data is None:
                print("Error: No se encontraron datos en la extensión especificada.")
                return

            x_start, x_end, y_start, y_end = roi
            roi_data = data[y_start:y_end, x_start:x_end]
            mean_counts = np.mean(roi_data)
            std_counts = np.std(roi_data)
            
            zscale = ZScaleInterval()
            zlow, zhigh = zscale.get_limits(roi_data)

            plt.figure(figsize=(8, 6))
            plt.imshow(roi_data, cmap='gray', clim=(zlow, zhigh), extent=[x_start, x_end, y_end, y_start])
            plt.colorbar(label='Cuentas')
            plt.gca().invert_yaxis() # Invertir el eje Y
            plt.title(f'ROI y Media de Cuentas: {mean_counts:.2f}, STD: {std_counts}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {file_path}")
    except Exception as e:
        print(f"Ocurrió un error: {e}")
    return mean_counts, std_counts


def show_multi_fits_image(filename):
    """
    Displays all data-containing extensions of a FITS file.

    Args:
        filename (str): The path to the FITS file.

    Returns:
        None

    Example:
        >>> show_multi_fits_image("example.fits")

    Notes:
        - This function iterates over all extensions in the FITS file and uses 
          `show_fits_image` to display the extensions that contain valid data.
        - If an extension does not contain data, it is skipped with a message logged 
          to the console.
    """
    with fits.open(filename) as hdulist:
        print(f"Displaying file: {filename}")
        # Iterate over all extensions
        for i, hdu in enumerate(hdulist):
            if hdu.data is not None:
                print(f"  Extension {i}: Data shape = {hdu.data.shape}")
                # Example: Show the image for this extension
                show_fits_image(filename, index=i, cmap='gray')

def combine_fits_extensions(path_files, extensions, output_file, method='mean'):
    """
    Combine mltiples extensiones dentro de un solo archivo FITS.

    Args:
        path_files (str): Ruta del archivo FITS de entrada.
        extensions (list): Lista de índices de extensiones a combinar.
        output_file (str): Nombre del archivo FITS de salida.
        method (str): Método de combinación ('mean', 'median', 'sum').

    Returns:
        None
    Note: This cannot handle lists. A for loop is needed to use it.
    """
    with fits.open(path_files) as hdul:
        if max(extensions) >= len(hdul):
            raise ValueError("Una de las extensiones seleccionadas no existe en el archivo FITS.")

        images = []

        for ext in extensions:
            if hdul[ext].data is not None:
                images.append(hdul[ext].data)
            else:
                print(f"Advertencia: La extensión {ext} no contiene datos y será omitida.")

        if not images:
            raise ValueError("No se encontraron datos en las extensiones seleccionadas.")

        if method == 'mean':
            combined_data = np.mean(images, axis=0)
        elif method == 'median':
            combined_data = np.median(images, axis=0)
        elif method == 'sum':
            combined_data = np.sum(images, axis=0)
        else:
            raise ValueError("Método no válido. Usa 'mean', 'median' o 'sum'.")

        combined_hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=combined_data)])

        combined_hdul.writeto(output_file, overwrite=True)
        print(f"Imagen combinada guardada en: {output_file}")
        # Visualizar la imagen combinada (usa tu función para mostrar FITS)
        show_fits_image(output_file, index=1, cmap='gray')

def combine_fits_extensions_shifted(path_files, output_file):
    """
    Loads, processes, and combines the specified extensions from a FITS file,
    removes specified extensions, averages the remaining images, and saves and displays the result.

    Args:
        path_files (str): Path to the input FITS file.
        output_file (str): Name of the output FITS file to save.

    Returns:
        hdu
    Note: This cannot handle lists. A for loop is needed to use it.
    """
    # Define the order of extensions
    extensions = [1, 14, 16, 15, 13, 11, 12, 10, 5, 2, 4, 3, 9, 6, 8, 7]
    cut_size = 512
    gap = 15

    # Get prescan, overscan (biassec) and NCOL from the FITS header
    hdul = fits.open(path_files)
    hdr = hdul[1].header  # Assuming the trimsec and biassec are in the first extension header
    trimsec = hdr['TRIMSEC']
    biassec = hdr['BIASSEC']

    trimsec_list = list(map(int, trimsec.strip("[]").replace(",", ":").split(":")))
    prescan = trimsec_list[0]

    biassec_list = list(map(int, biassec.strip("[]").replace(",", ":").split(":")))
    overscan = biassec_list[1] - biassec_list[0] #Calculate the width of the overscan.

    ncol = int(hdul[2].header['NCOL'])
    pad_right = ncol - cut_size

    hdul.close()

    # Collect the processed images
    image_collector = []

    # Load and process the images
    for i, ext in enumerate(extensions):
        image_data = fits.getdata(path_files, i + 1)
        image_data = image_data[:, (gap * (ext - 1)):(gap * (ext - 1)) + cut_size + prescan + overscan]
        image_data = np.pad(image_data, ((0, 0), (0, pad_right - prescan - overscan)), mode='constant', constant_values=0)
        image_collector.append(image_data)

    print("Pre-removal:", len(image_collector))
    print("Shape of image_collector:", np.shape(image_collector))

    # Indices of the extensions to remove
    indices_to_remove = [7, 10, 14]  # Modify these indices as needed

    # Remove the specified extensions
    image_collector = np.delete(image_collector, indices_to_remove, axis=0)

    print("Post-removal:", len(image_collector))

    # Compute the average of the remaining images
    average_image = np.mean(image_collector, axis=0)

    # Get original headers
    hdul_original = fits.open(path_files)
    primary_hdr = hdul_original[0].header.copy() # copy the primary header
    image_hdr = hdul_original[1].header.copy() # copy the image header
    hdul_original.close()

    # Save the average as a new FITS file, keeping the headers
    primary_hdu = fits.PrimaryHDU(header=primary_hdr)
    image_hdu = fits.ImageHDU(data=average_image, header=image_hdr)
    hdu = fits.HDUList([primary_hdu, image_hdu])

    hdu.writeto(output_file, overwrite=True)
    print(f"Combined image saved in : {output_file}")
    # Display the average image using show_fits_image
    show_fits_image(output_file, index=1, cmap="gray")
    return hdu
    
def overscan_correction(path_files, output_file, method='mean'):
    """
    Applies overscan correction to all 16 extensions in a FITS file.

    Args:
        path_files (str): Path to the input FITS file.
        output_file (str): Path to save the corrected FITS file.
        method (str): 'mean' for simple subtraction, 'poly' for polynomial fitting.

    Returns:
        None
    """

    # Get overscan (biassec) from the FITS header
    hdul = fits.open(path_files)
    hdr = hdul[1].header  # Assuming the trimsec and biassec are in the first extension header
    biassec = hdr['BIASSEC']
    biassec_list = list(map(int, biassec.strip("[]").replace(",", ":").split(":")))
    overscan_width = biassec_list[1] - biassec_list[0]  # Calculate the width of the overscan.
    print(f"overscan_width: {overscan_width}")
    hdul.close()

    with fits.open(path_files, mode='readonly') as hdul:
        # Copy the original primary header
        corrected_hdul = fits.HDUList([fits.PrimaryHDU(header=hdul[0].header.copy())])

        for ext in range(1, 17):  # Iterate through the 16 extensions
            if ext >= len(hdul) or hdul[ext].data is None:
                print(f"⚠️ Warning: Extension {ext} does not exist or has no data.")
                continue

            data = hdul[ext].data.astype(float)  # Convert to float for precision
            overscan_region = data[:, -overscan_width:]  # Last `overscan_width` pixels

            if method == 'mean':
                overscan_value = np.mean(overscan_region)  # Calculate mean overscan level
            elif method == 'poly':
                # Fit a polynomial across the overscan region (1D polynomial)
                x = np.arange(overscan_region.shape[1])
                y = np.mean(overscan_region, axis=0)
                poly_coeffs = np.polyfit(x, y, deg=2)  # 2nd-degree polynomial
                poly_fit = np.polyval(poly_coeffs, x)
                overscan_value = np.mean(poly_fit)
            else:
                raise ValueError("❌ Invalid method. Use 'mean' or 'poly'.")

            print(f"Overscan mean pre correction: {overscan_value}")
            corrected_data = data - overscan_value  # Subtract overscan
            area_post = corrected_data[:, -overscan_width:]
            print(f"Overscan mean post correction: {np.mean(area_post)}")

            # Create a new FITS extension
            corrected_hdu = fits.ImageHDU(data=corrected_data, header=hdul[ext].header)
            corrected_hdul.append(corrected_hdu)

        # Save the corrected FITS file
        corrected_hdul.writeto(output_file, overwrite=True)
        print(f"Overscan-corrected FITS saved as: {output_file}")

def bias_subtraction(path_files, master_bias_file, output_file):
    """
    Applies bias subtraction to all 16 extensions in a FITS file.

    Args:
        path_files (str): Path to the input FITS file (science image).
        master_bias_file (str): Path to the master bias FITS file.
        output_file (str): Path to save the bias-corrected FITS file.

    Returns:
        None
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
        
def calculate_background_noise(file_path, extension=1, roi=None):
    """
    Calculates the background noise of a FITS image by standard deviation.

    Args:
        file_path (str): path of the FITS file.
        extension (int): Index of the extension to analyze.
        roi (tuple): Region of interest in format (xmin, xmax, ymin, ymax). If None, uses the whole image.

    Returns:
        float: estimated image noise (standard deviation in ADU).
    """
    with fits.open(file_path) as hdulist:
        data = hdulist[extension].data
        if data is None:
            raise ValueError(f"There is no data in the extension {extension}.")

        if roi:
            xmin, xmax, ymin, ymax = roi
            data = data[ymin:ymax, xmin:xmax]
        noise = np.std(data)
        return noise

def calculate_overscan_noise(path_files, extension=1, method='std'):
    """
    Calcula el ruido del sobreescaneo en una extensión de un archivo FITS.

    Args:
        path_files (str): Ruta al archivo FITS.
        extension (int, optional): Número de la extensión a procesar. Por defecto es 1.
        method (str, optional): Método para calcular el ruido. 'std' para la desviación estándar,
                                'var' para la varianza. Por defecto es 'std'.

    Returns:
        float: El ruido del sobreescaneo.
    """
    with fits.open(path_files, mode='readonly') as hdul:
        if extension >= len(hdul) or hdul[extension].data is None:
            raise ValueError(f"Extensión {extension} no existe o no tiene datos.")

        # Obtiene el BIASSEC del encabezado
        hdr = hdul[extension].header #access the header of the extension.
        biassec = hdr['BIASSEC']
        psamp = int(hdr["PSAMP"])
        biassec_list = list(map(int, biassec.strip("[]").replace(",", ":").split(":")))
        overscan_start = biassec_list[0]
        overscan_end = biassec_list[1]

        # Extrae la región de sobreescaneo
        data = hdul[extension].data.astype(float)
        overscan_region = data[:, overscan_start:overscan_end]

        # Calcula el ruido
        if method == 'std':
            noise = np.std(overscan_region)
        elif method == 'var':
            noise = np.var(overscan_region)
        else:
            raise ValueError("Método inválido. Usa 'std' o 'var'.")

        return noise / psamp

def plot_variance_vs_sum(path_files, roi):
    """
    Grafica V(M1' - M1) vs M1 + M1' para cada una de las 16 extensiones del MAS-CCD Skipper.

    Args:
        path_files (list of str): Lista de archivos FITS en pares con el mismo tiempo de exposición.
    """
    # Organizar imágenes por tiempo de exposición
    exposure_times = defaultdict(list)
    for file_path in path_files:
        with fits.open(file_path) as hdulist:
            exptime = hdulist[0].header.get('EXPTIME', None)
            if exptime is None:
                print(f" EXPTIME no encontrado en {file_path}")
                continue
            exposure_times[exptime].append(file_path)

    # Obtener el número de extensiones del primer archivo FITS
    with fits.open(path_files[0]) as hdulist:
        num_extensions = len(hdulist) - 1  # Restar 1 para excluir la extensión principal
    ext_side = int(np.sqrt(num_extensions))
    # Crear figura con subplots (4x4) para las 16 extensiones
    fig, axes = plt.subplots(ext_side, ext_side, figsize=(12, 12))
    fig.suptitle("V(M1' - M1) vs M1 + M1' para el MAS-CCD Skipper", fontsize=14)

    # Iterar sobre cada extensión (1 a 16)
    for ext in range(1, num_extensions + 1):
        row, col = (ext - 1) // ext_side, (ext - 1) % ext_side
        ax = axes[row, col]

        # Listas para almacenar varianza y suma de medias
        extension_variances = []
        extension_sum_of_means = []

        # Ordenar los tiempos de exposición
        exposure_time_keys = sorted(exposure_times.keys())

        # Procesar pares de imágenes con el mismo tiempo de exposición
        for exptime in exposure_time_keys:
            if len(exposure_times[exptime]) == 2:
                file1, file2 = exposure_times[exptime]

                with fits.open(file1) as hdul1, fits.open(file2) as hdul2:
                    data1 = hdul1[ext].data
                    data2 = hdul2[ext].data

                    if data1 is None or data2 is None:
                        continue

                    # Apply ROI selection
                    if roi:
                        x_start, x_end, y_start, y_end = roi
                        data1 = data1[y_start:y_end, x_start:x_end]
                        data2 = data2[y_start:y_end, x_start:x_end]

                    # Calcular varianza de la diferencia y suma de medias
                    diff_data = data2 - data1  # M1' - M1
                    variance = np.var(diff_data) #/ 2  # Factor 2 por la diferencia
                    sum_of_mean = np.mean(data2) + np.mean(data1)  # M1 + M1'
                    
                    extension_variances.append(variance)
                    extension_sum_of_means.append(sum_of_mean)

        # Graficar
        ax.plot(extension_variances, extension_sum_of_means, 'o', label=f'Ext {ext}', color="blue", markersize = 5)

        ax.set_title(f"Extensión {ext}")
        ax.set_xlabel("Varianza de la diferencia")
        ax.set_ylabel("Suma de Medias")
        #ax.legend()
        ax.grid()

    # Ajustar diseño y mostrar
    plt.tight_layout()
    plt.show()


    
def plot_exposure_time_vs_mean(path_files, roi=None):
    """
    Grafica el tiempo de exposición (EXPTIME) vs la media para cada una de las 16 extensiones del MAS-CCD Skipper.

    Args:
        path_files (list of str): Lista de archivos FITS en pares con el mismo tiempo de exposición.
        roi (tuple, optional): Región de interés (ROI) como (x_start, x_end, y_start, y_end).
    Return:
        extension_means (list): Lista de listas de cuentas promedio en el ROI seleccionado, una lista por extensión.
    """
    # Organizar imágenes por tiempo de exposición
    exposure_times = defaultdict(list)
    for file_path in path_files:
        with fits.open(file_path) as hdulist:
            exptime = hdulist[0].header.get('EXPTIME', None)
            if exptime is None:
                print(f"EXPTIME no encontrado en {file_path}")
                continue
            exposure_times[exptime].append(file_path)

    # Obtener el número de extensiones del primer archivo FITS
    with fits.open(path_files[0]) as hdulist:
        num_extensions = len(hdulist) - 1  # Restar 1 para excluir la extensión principal
    ext_side = int(np.sqrt(num_extensions))

    # Crear figura con subplots (4x4) para las 16 extensiones
    fig, axes = plt.subplots(ext_side, ext_side, figsize=(12, 12))
    fig.suptitle("Tiempo de Exposición vs Media de cuentas para el MAS-CCD Skipper", fontsize=14)

    # Lista para almacenar las medias de cada extensión
    all_extension_means = []

    # Iterar sobre cada extensión (1 a 16)
    for ext in range(1, num_extensions + 1):
        row, col = (ext - 1) // ext_side, (ext - 1) % ext_side
        ax = axes[row, col]

        # Listas para almacenar tiempos de exposición y medias
        extension_exptimes = []
        extension_means = []

        # Ordenar los tiempos de exposición
        exposure_time_keys = sorted(exposure_times.keys())

        # Procesar pares de imágenes con el mismo tiempo de exposición
        for exptime in exposure_time_keys:
            if len(exposure_times[exptime]) == 2:
                file1, file2 = exposure_times[exptime]

                with fits.open(file1) as hdul1, fits.open(file2) as hdul2:
                    data1 = hdul1[ext].data
                    data2 = hdul2[ext].data

                    if data1 is None or data2 is None:
                        continue

                    # Aplicar ROI selection
                    if roi:
                        x_start, x_end, y_start, y_end = roi
                        data1 = data1[y_start:y_end, x_start:x_end]
                        data2 = data2[y_start:y_end, x_start:x_end]

                    # Calcular la media de los datos
                    mean_data = (np.mean(data1) + np.mean(data2)) / 2

                    extension_exptimes.append(exptime)
                    extension_means.append(mean_data)

        # Graficar
        ax.plot(extension_exptimes, extension_means, 'o', label=f'Ext {ext}', color="red", markersize=5)

        ax.set_title(f"Extensión {ext}")
        ax.set_xlabel("Tiempo de Exposición (EXPTIME)")
        ax.set_ylabel("Media de cuentas")
        #ax.legend()
        ax.grid()

        # Agregar las medias de esta extensión a la lista principal
        all_extension_means.append(extension_means)

    # Ajustar diseño y mostrar
    plt.tight_layout()
    plt.show()
    return all_extension_means
    
# def linear_reg_param(x, y):
#     """
#     Fits a linear regression model and calculates residuals, slope, intercept, and RMSE.

#     Args:
#         x (array-like): Independent variable values (1D array).
#         y (array-like): Dependent variable values (1D array).

#     Returns:
#         - 'residuals' (numpy.ndarray): The residuals of the fit (y - y_pred).
#         - 'slope' (float): The slope of the fitted line.
#         - 'intercept' (float): The intercept of the fitted line.
#         - 'rmse' (float): The root mean squared error of the fit.
#     """
#     x = np.array(x).reshape(-1, 1)
#     y = np.array(y).reshape(-1, 1)

#     model = LinearRegression()
#     model.fit(x, y)

#     y_pred = model.predict(x)
#     residuals = y - y_pred
#     # RMSE
#     rmse = np.sqrt(mean_squared_error(y, y_pred))

#     # Retrieve slope and intercept
#     slope = model.coef_[0][0]
#     intercept = model.intercept_[0]
#     return residuals, slope, intercept, rmse

def best_gain_fit(x, y, min_points=5):
    """
    Encuentra el mejor ajuste lineal eliminando outliers usando el criterio de menor MSE.
    Se asegura de usar al menos `min_points` para evitar ajustes sobre pocos datos.
    """
    n = len(x)
    if n < min_points:
        return None, None, None, None, None  # No es posible ajustar

    best_mse = np.inf
    best_fit = None
    best_indices = None

    # Probar todas las combinaciones de puntos desde min_points hasta n
    for r in range(n, min_points - 1, -1):  # De n a min_points datos
        for indices in combinations(range(n), r):
            x_fit = x[list(indices)]
            y_fit = y[list(indices)]
            
            if len(x_fit) < min_points:
                continue

            model = LinearRegression()
            model.fit(x_fit.reshape(-1, 1), y_fit)
            y_pred = model.predict(x_fit.reshape(-1, 1))
            mse = mean_squared_error(y_fit, y_pred)

            if mse < best_mse:
                best_mse = mse
                best_fit = (model.coef_[0], model.intercept_)
                best_indices = indices

    if best_fit is None:
        return None, None, None, None, None

    slope, intercept = best_fit
    x_best = x[list(best_indices)]
    y_best = y[list(best_indices)]

    return slope, intercept, x_best, y_best, best_indices

def new_gain(path_files, roi, min_points=5): 
    """
    Grafica V(M1' - M1) vs M1 + M1' para cada una de las 16 extensiones del MAS-CCD Skipper.

    NOTA: EL ROI ES FIJO. ESTOY USANDO UN ROI COMÜN PARA LAS 16 EXTENSIONES QUE ES DENTRO DE 260,440 EN X*
    """
    gains = {}
    exposure_times = defaultdict(list)
    for file_path in path_files:
        with fits.open(file_path) as hdulist:
            exptime = hdulist[0].header.get('EXPTIME', None)
            if exptime is None:
                continue
            exposure_times[exptime].append(file_path)

    # Obtener el número de extensiones del primer archivo FITS
    with fits.open(path_files[0]) as hdulist:
        num_extensions = len(hdulist) - 1  # Restar 1 para excluir la extensión principal

    ext_side = int(np.sqrt(num_extensions))
    fig, axes = plt.subplots(ext_side, ext_side, figsize=(12, 12))
    fig.suptitle("V(M1' - M1) vs M1 + M1' para el MAS-CCD Skipper", fontsize=14)

    # Encontrar el valor máximo de sum_of_means para todas las extensiones
    max_y_value = 0
    all_extension_sum_of_means = []
    
    for ext in range(1, num_extensions + 1):
        extension_variances = []
        extension_sum_of_means = []
        exposure_time_keys = sorted(exposure_times.keys())

        for exptime in exposure_time_keys:
            if len(exposure_times[exptime]) == 2:
                file1, file2 = exposure_times[exptime]
                with fits.open(file1) as hdul1, fits.open(file2) as hdul2:
                    if ext >= len(hdul1) or ext >= len(hdul2):
                        continue
                    data1 = hdul1[ext].data
                    data2 = hdul2[ext].data
                    if data1 is None or data2 is None:
                        continue
                    if roi:
                        x_start, x_end, y_start, y_end = roi
                        data1 = data1[y_start:y_end, x_start:x_end]
                        data2 = data2[y_start:y_end, x_start:x_end]
                    diff_data = data2 - data1
                    variance = np.var(diff_data)
                    sum_of_mean = np.mean(data2) + np.mean(data1)
                    if np.isnan(variance) or np.isnan(sum_of_mean):
                        continue
                    extension_variances.append(variance)
                    extension_sum_of_means.append(sum_of_mean)

        if len(extension_variances) < 4:
            continue

        extension_variances = np.array(extension_variances)
        extension_sum_of_means = np.array(extension_sum_of_means)
        all_extension_sum_of_means.extend(extension_sum_of_means)

        slope, intercept, x_best, y_best, best_indices = best_gain_fit(extension_variances, extension_sum_of_means, min_points)
        if slope is None:
            continue

        gains[ext] = 1 / slope

    max_y_value = max(all_extension_sum_of_means)
    
    for ext in range(1, num_extensions + 1):
        row, col = (ext - 1) // ext_side, (ext - 1) % ext_side
        ax = axes[row, col]
        extension_variances = []
        extension_sum_of_means = []
        exposure_time_keys = sorted(exposure_times.keys())

        for exptime in exposure_time_keys:
            if len(exposure_times[exptime]) == 2:
                file1, file2 = exposure_times[exptime]
                with fits.open(file1) as hdul1, fits.open(file2) as hdul2:
                    if ext >= len(hdul1) or ext >= len(hdul2):
                        continue
                    data1 = hdul1[ext].data
                    data2 = hdul2[ext].data
                    if data1 is None or data2 is None:
                        continue
                    if roi:
                        x_start, x_end, y_start, y_end = roi
                        data1 = data1[y_start:y_end, x_start:x_end]
                        data2 = data2[y_start:y_end, x_start:x_end]
                    diff_data = data2 - data1
                    variance = np.var(diff_data)
                    sum_of_mean = np.mean(data2) + np.mean(data1)
                    if np.isnan(variance) or np.isnan(sum_of_mean):
                        continue
                    extension_variances.append(variance)
                    extension_sum_of_means.append(sum_of_mean)

        if len(extension_variances) < 4:
            continue

        extension_variances = np.array(extension_variances)
        extension_sum_of_means = np.array(extension_sum_of_means)

        slope, intercept, x_best, y_best, best_indices = best_gain_fit(extension_variances, extension_sum_of_means, min_points)
        if slope is None:
            continue

        gains[ext] = 1 / slope
        ax.plot(extension_variances, extension_sum_of_means, 'o', label=f'Ext {ext}', color="blue", markersize=5)
        fit_line = np.polyval([slope, intercept], extension_variances)
        ax.plot(extension_variances, fit_line, 'r--', label=f'Fit (Gain={slope:.3f})')
        gain_text = f"Gain: {1 / slope:.3f} e-/ADU"
        ax.text(0.05, 0.95, gain_text, transform=ax.transAxes, fontsize=10, color='red', verticalalignment='top')
        ax.set_title(f"Extensión {ext}")
        ax.set_xlabel("Varianza de la diferencia")
        ax.set_ylabel("Suma de Medias")
        ax.grid()
        ax.set_ylim(0, max_y_value*1.1)

    plt.tight_layout()
    plt.show()
    gains = list(gains.values())
    return gains
    
def calculate_gain_simple(path_files, roi, max_slope_diff=0.5):
    """
    Calcula la ganancia para cada extensión del MAS-CCD Skipper.
    Calcula las pendientes entre puntos consecutivos y luego hace la media.
    Descarta puntos donde la pendiente difiere demasiado.

    Args:
        path_files (list of str): Lista de archivos FITS en pares con el mismo tiempo de exposición.
        roi (tuple): Región de interés (x_start, x_end, y_start, y_end).
        max_slope_diff (float): Máxima diferencia permitida entre pendientes consecutivas.

    Returns:
        dict: Diccionario con la ganancia estimada para cada extensión.
    """
    gains = {}

    # Organizar imágenes por tiempo de exposición
    exposure_times = defaultdict(list)
    for file_path in path_files:
        with fits.open(file_path) as hdulist:
            exptime = hdulist[0].header.get('EXPTIME', None)
            if exptime is None:
                print(f"⚠️ EXPTIME no encontrado en {file_path}")
                continue
            exposure_times[exptime].append(file_path)

    # Obtener el número de extensiones del primer archivo FITS
    with fits.open(path_files[0]) as hdulist:
        num_extensions = len(hdulist) - 1  # Restar 1 para excluir la extensión principal

    # Crear figura con subplots (4x4) para visualizar cada extensión
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle("Cálculo de Ganancia para el MAS-CCD Skipper", fontsize=16)

    # Iterar sobre cada extensión (1 a num_extensions)
    for ext in range(1, num_extensions + 1):
        row, col = (ext - 1) // 4, (ext - 1) % 4
        ax = axes[row, col]

        # Listas para almacenar varianza y suma de medias
        extension_variances = []
        extension_sum_of_means = []

        # Ordenar los tiempos de exposición
        exposure_time_keys = sorted(exposure_times.keys())

        # Procesar pares de imágenes con el mismo tiempo de exposición
        for exptime in exposure_time_keys:
            if len(exposure_times[exptime]) == 2:
                file1, file2 = exposure_times[exptime]

                with fits.open(file1) as hdul1, fits.open(file2) as hdul2:
                    if ext >= len(hdul1) or ext >= len(hdul2):
                        print(f" Extensión {ext} no disponible en {file1} o {file2}")
                        continue

                    data1 = hdul1[ext].data
                    data2 = hdul2[ext].data

                    # Apply ROI selection
                    if roi:
                        x_start, x_end, y_start, y_end = roi
                        data1 = data1[y_start:y_end, x_start:x_end]
                        data2 = data2[y_start:y_end, x_start:x_end]

                    if data1 is None or data2 is None:
                        continue

                    # Calcular varianza de la diferencia y suma de medias
                    diff_data = data1 - data2
                    variance = np.var(diff_data)
                    sum_of_mean = np.mean(data1) + np.mean(data2)

                    # Evitar valores atípicos o NaN
                    if np.isnan(variance) or np.isnan(sum_of_mean):
                        continue

                    extension_variances.append(variance)
                    extension_sum_of_means.append(sum_of_mean)

        # Asegurar que hay suficientes puntos para calcular pendientes
        if len(extension_variances) < 2:
            print(f" No hay suficientes datos para calcular la ganancia en la extensión {ext}.")
            continue

        # Calcular pendientes entre puntos consecutivos
        slopes = []
        for i in range(len(extension_variances) - 1):
            slope = (extension_sum_of_means[i+1] - extension_sum_of_means[i]) / (extension_variances[i+1] - extension_variances[i])
            slopes.append(slope)

        # Filtrar pendientes atípicas
        filtered_slopes = []
        if slopes:
            filtered_slopes.append(slopes[0])  # Añadir la primera pendiente siempre
            for i in range(1, len(slopes)):
                if abs(slopes[i] - filtered_slopes[-1]) <= max_slope_diff:
                    filtered_slopes.append(slopes[i])

        # Calcular ganancia promedio
        if filtered_slopes:
            avg_slope = np.mean(filtered_slopes)
            gains[ext] = 1 / avg_slope
        else:
            print(f" No se pudieron calcular pendientes válidas para la extensión {ext}.")
            gains[ext] = np.nan

        # Graficar los datos
        ax.plot(extension_variances, extension_sum_of_means, 'o', label=f'Ext {ext} Data', color="blue")

        # Graficar la línea de ganancia promedio
        if filtered_slopes:
            fit_line = np.polyval([avg_slope, 0], extension_variances)
            ax.plot(extension_variances, fit_line, 'r--', label=f'Fit (Gain={1/avg_slope:.3f})')

            # Anotar la ganancia en el gráfico
            gain_text = f"Gain: {1 / avg_slope:.3f} e-/ADU"
            ax.text(0.05, 0.95, gain_text, transform=ax.transAxes, fontsize=10, color='red', verticalalignment='top')
        else:
            ax.text(0.05, 0.95, "Gain: N/A", transform=ax.transAxes, fontsize=10, color='red', verticalalignment='top')

        ax.set_title(f"Extensión {ext}")
        ax.set_xlabel("Varianza (ADU²)")
        ax.set_ylabel("Suma de Medias (ADU)")
        ax.grid()

    # Ajustar diseño y mostrar
    plt.tight_layout()
    plt.show()

    return gains

def plot_exposure_time_vs_mean_linear_fit(path_files, roi=None, threshold=0.01, min_points=5):
    """
    Grafica el tiempo de exposición (EXPTIME) vs la media, realiza un ajuste lineal,
    descarta puntos con residuales < threshold y encuentra el full well.

    Args:
        path_files (list of str): Lista de archivos FITS en pares con el mismo tiempo de exposición.
        roi (tuple, optional): Región de interés (ROI) como (x_start, x_end, y_start, y_end).
        threshold (float, optional): Umbral para descartar puntos (porcentaje).
    Return:
        all_extension_full_wells (list): Lista de valores de full well para cada extensión.
    """
    exposure_times = defaultdict(list)
    for file_path in path_files:
        with fits.open(file_path) as hdulist:
            exptime = hdulist[0].header.get('EXPTIME', None)
            if exptime is None:
                print(f"EXPTIME no encontrado en {file_path}")
                continue
            exposure_times[exptime].append(file_path)

    with fits.open(path_files[0]) as hdulist:
        num_extensions = len(hdulist) - 1
    ext_side = int(np.sqrt(num_extensions))

    fig, axes = plt.subplots(ext_side, ext_side, figsize=(12, 12))
    fig.suptitle("Tiempo de Exposición vs Media con Ajuste Lineal", fontsize=14)

    all_extension_full_wells = []

    for ext in range(1, num_extensions + 1):
        row, col = (ext - 1) // ext_side, (ext - 1) % ext_side
        ax = axes[row, col]

        extension_exptimes = []
        extension_means = []
        exposure_time_keys = sorted(exposure_times.keys())

        for exptime in exposure_time_keys:
            if len(exposure_times[exptime]) == 2:
                file1, file2 = exposure_times[exptime]
                with fits.open(file1) as hdul1, fits.open(file2) as hdul2:
                    data1 = hdul1[ext].data
                    data2 = hdul2[ext].data
                    if data1 is None or data2 is None:
                        continue
                    if roi:
                        x_start, x_end, y_start, y_end = roi
                        data1 = data1[y_start:y_end, x_start:x_end]
                        data2 = data2[y_start:y_end, x_start:x_end]
                    mean_data = (np.mean(data1) + np.mean(data2)) / 2
                    extension_exptimes.append(exptime)
                    extension_means.append(mean_data)

        if len(extension_exptimes) < 4:
            all_extension_full_wells.append(None)
            continue

        slope, intercept, good_exptimes, good_means, _ = best_gain_fit(np.array(extension_exptimes), np.array(extension_means),min_points)

        if slope is None:
            all_extension_full_wells.append(None)
            continue

        fit_line = np.polyval([slope, intercept], good_exptimes)
        residuals = np.abs(good_means - fit_line) / good_means

        full_well_index = np.where(residuals > threshold)[0][0] if np.any(residuals > threshold) else len(good_exptimes) - 1
        full_well = good_means[full_well_index]

        all_extension_full_wells.append(full_well)

        ax.plot(extension_exptimes, extension_means, 'o', label=f'Ext {ext}', color="red", markersize=5)
        ax.plot(good_exptimes, fit_line, 'b--', label='Ajuste Lineal')
        ax.axvline(x=good_exptimes[full_well_index], color='g', linestyle='--', label='Full Well')

        ax.set_title(f"Extensión {ext}")
        ax.set_xlabel("Tiempo de Exposición (EXPTIME)")
        ax.set_ylabel("Media de cuentas")
        ax.grid()

    plt.tight_layout()
    plt.show()
    return all_extension_full_wells
    
def calculate_cte(archivo_fits, numero_linea, roi=None, ext=1):
    """
    Calcula la CTE de un CCD a partir de un archivo FITS, graficando la posición del píxel 
    vs las cuentas en una línea específica, con opción de ROI.

    Args:
        archivo_fits (str): La ruta al archivo FITS.
        numero_linea (int): El número de línea del CCD a analizar (ej., 50).
        roi (list, optional): Lista [inicio_pixel, fin_pixel] para seleccionar un ROI en la línea. 
                               Defaults to None (analiza toda la línea).
        ext (int, optional): El número de extensión en el archivo fits. Defaults to 1.

    Returns:
        float: El valor de CTE calculado.
        None: Si hay un error al procesar el archivo o realizar el ajuste lineal.
    """
    try:
        with fits.open(archivo_fits) as hdulist:
            datos_ccd = hdulist[ext].data
            linea_ccd = datos_ccd[numero_linea, :]

            if roi:
                inicio_pixel, fin_pixel = roi[0], roi[1]  # Extrae los valores de la lista
                linea_ccd = linea_ccd[inicio_pixel:fin_pixel]
                posiciones_pixel = np.arange(inicio_pixel, fin_pixel)
            else:
                posiciones_pixel = np.arange(len(linea_ccd))

            modelo_regresion = LinearRegression()
            modelo_regresion.fit(posiciones_pixel.reshape(-1, 1), linea_ccd)
            pendiente = modelo_regresion.coef_[0]
            cte = 1 - abs(pendiente / np.mean(linea_ccd))

            plt.figure(figsize=(10, 6))
            plt.scatter(posiciones_pixel, linea_ccd, label=f'Data')
            plt.plot(posiciones_pixel, modelo_regresion.predict(posiciones_pixel.reshape(-1, 1)), 'r-', label='Ajuste Lineal')
            plt.xlabel(f"Posición del Píxel, Ext:{ext}, Line:{numero_linea}")
            plt.ylabel('Cuentas')
            plt.title(f'CTE = {cte:.6f}')
            plt.legend()
            plt.grid(True)
            plt.show()

            return cte

    except Exception as e:
        print(f"Error al procesar el archivo {archivo_fits}: {e}")
        return None
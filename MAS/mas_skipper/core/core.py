import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict

from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider

def get_roi_info(file_path, sci_file=None):
    """
    Lee la información de ROI desde el encabezado de una imagen FITS.

    Args:
        file_path (str): Ruta al archivo FITS (bias, flat o ciencia).
        sci_file (str, optional): Ruta a una imagen científica para alinear el ROI.

    Returns:
        tuple: (skiprow, nrows, ncols, is_roi)
            - skiprow (int): Fila inicial del corte (0 si no es ROI).
            - nrows (int): Número de filas.
            - ncols (int): Número de columnas.
            - is_roi (bool): True si la imagen es ROI.
    """
    NAXIS2_MAX = 1200
    NAXIS1_MAX = 900

    try:
        skiprow, nrows, ncols, is_roi = 0, NAXIS2_MAX, NAXIS1_MAX, False
        with fits.open(file_path, mode='readonly') as hdul:
            # Buscar la primera extensión válida
            for hdu in hdul[1:]:
                if hdu.data is not None:
                    skiprow = int(hdu.header.get('SKIPROW', 690))
                    nrows = int(hdu.header.get('NAXIS2', NAXIS2_MAX))
                    ncols = int(hdu.header.get('NAXIS1', NAXIS1_MAX))
                    is_roi = skiprow > 0 or nrows < NAXIS2_MAX or hdu.header.get('ROI', False) or hdu.header.get('WINDOW', False)
                    break

        # Si es un archivo corregido, alinear con imagen científica
        if sci_file and os.path.basename(file_path).startswith('o_'):
            with fits.open(sci_file, mode='readonly') as sci_hdul:
                for hdu in sci_hdul[1:]:
                    if hdu.data is not None:
                        skiprow = int(hdu.header.get('SKIPROW', 0))
                        nrows = int(hdu.header.get('NAXIS2', NAXIS2_MAX))
                        ncols = int(hdu.header.get('NAXIS1', NAXIS1_MAX))
                        is_roi = skiprow > 0 or nrows < NAXIS2_MAX or hdu.header.get('ROI', False) or hdu.header.get('WINDOW', False)
                        break

        # Validaciones
        if skiprow < 0 or skiprow >= NAXIS2_MAX:
            print(f"⚠️ SKIPROW inválido ({skiprow}) en {file_path}. Asumiendo 0.")
            skiprow = 0
        if nrows <= 0 or nrows > NAXIS2_MAX:
            print(f"⚠️ NAXIS2 inválido ({nrows}) en {file_path}. Ajustando a {NAXIS2_MAX}.")
            nrows = NAXIS2_MAX
        if ncols <= 0 or ncols > NAXIS1_MAX:
            print(f"⚠️ NAXIS1 inválido ({ncols}) en {file_path}. Ajustando a {NAXIS1_MAX}.")
            ncols = NAXIS1_MAX
        if skiprow + nrows > NAXIS2_MAX:
            print(f"⚠️ ROI fuera de rango ({skiprow}+{nrows}) en {file_path}. Ajustando nrows.")
            nrows = NAXIS2_MAX - skiprow

        print(f"🔍 ROI info para {file_path}: skiprow={skiprow}, nrows={nrows}, ncols={ncols}, is_roi={is_roi}, rango=[{skiprow}:{skiprow + nrows}]")
        return skiprow, nrows, ncols, is_roi

    except Exception as e:
        print(f"⚠️ Error al leer ROI en {file_path}: {e}. Asumiendo imagen completa.")
        return 0, NAXIS2_MAX, NAXIS1_MAX, False

def roi_shifting(roi, return_extensions_order=False):
    """
    Shift an ROI vector to align the prescans of the MAS sensor channels.

    Args:
        roi (list): The ROI to align, format [x1, x2, y1, y2].
        return_extensions_order (bool): If True, also return the extension order used.
    
    Returns:
        shifted_roi (list): List of shifted ROIs aligned with amplifier layout.
        extensions (optional): Extension order used (if return_extensions_order=True).

    Notes:
        - For this case the amplifiers are 15 px apart (gap).
    """
    extensions = [1, 14, 16, 15, 13, 11, 12, 10, 5, 2, 4, 3, 9, 6, 8, 7]
    gap = 15
    shifted_roi = []

    for ext in extensions:
        x_start = roi[0] + (gap * (ext - 1))
        x_end = roi[1] + (gap * (ext - 1))
        y_start = roi[2]
        y_end = roi[3]
        shifted_roi.append([x_start, x_end, y_start, y_end])

    if return_extensions_order:
        return shifted_roi, extensions
    else:
        return shifted_roi

def roi_shifting_from_header(fits_path):
    """
    Define ROIs de overscan por extensión usando los headers.

    Lee NCOL, NROW y SKIPROW desde el header de cada extensión (del 1 al 16)
    y devuelve un vector de ROIs: [col1, col2, row1, row2].

    Returns:
        roi_vector (list of list of int): Lista de ROIs para cada extensión.
    """
    roi_vector = []
    with fits.open(fits_path, memmap=False) as hdul:
        for ext in range(1, len(hdul)):
            hdr = hdul[ext].header
            try:
                ncol = int(hdr.get('NCOL'))
                nrow = int(hdr.get('NROW'))
                skiprow = int(hdr.get('SKIPROW', 0))  # puede no estar presente

                col_start = 540
                col_end = 550
                row_start = skiprow
                row_end = skiprow + nrow

                # Validación mínima
                if col_end > ncol or row_end > (skiprow + nrow):
                    print(f"⚠️ ROI out of bounds in EXT {ext}. Skipping.")
                    roi_vector.append(None)
                else:
                    roi_vector.append([col_start, col_end, row_start, row_end])

            except Exception as e:
                print(f"❌ Error in EXT {ext}: {e}")
                roi_vector.append(None)
    return roi_vector

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
        - obtain_path_files(path="/data/CCD", ends_with=True, filtering="o_", NOT=False)
        - obtain_path_files(path="/data/CCD", ends_with=True, filtering="b.fits", NOT=True)

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
    
def obtain_output_paths(input_paths, suffix="_corrected"):
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

def show_fits_image(data, cmap="gray", figsize=(8, 8)):
    """
    Displays a 2D FITS image array with ZScale and colorbar.

    """
    zscale = ZScaleInterval()
    zlow, zhigh = zscale.get_limits(data)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, clim=(zlow, zhigh))
    # Create colorbar with same height as the y-axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1.5%", pad=0.05)
    fig = ax.figure
    fig.colorbar(im, cax=cax)
    ax.invert_yaxis()  # added this line
    return fig, ax, im

def combine_science_images(corrected_files, output_file, exclude_extensions=None, comb_mode='simple', weights=None):
    """
    Combina imágenes científicas corregidas promediando las extensiones válidas.

    Args:
        corrected_files (list): Lista de archivos FITS corregidos.
        output_file (str): Ruta para el archivo FITS combinado.
        roi_base (list): Ignorado (mantenido por compatibilidad). Por defecto None.
        exclude_extensions (list): Lista de extensiones (1-16) a excluir del promedio.
        comb_mode (str): Modo de combinación: 'simple' o 'weighted'.
        weights (list, optional): Pesos para las extensiones (longitud 16).
                                  Si None y modo 'weighted', calcula SNR internamente.
    """
    if not corrected_files:
        print("⚠️ No hay archivos corregidos para combinar.")
        return

    # Orden de extensiones según el CCD MAS-Skipper
    _, extorder = roi_shifting([0, 0, 0, 0], return_extensions_order=True)
    exclude_extensions = [int(ext) for ext in (exclude_extensions or []) if 1 <= int(ext) <= 16]
    print(f"🔍 Extensiones excluidas recibidas: {exclude_extensions}")
    active_extensions = [ext for ext in extorder if ext not in exclude_extensions]
    print(f"🔍 Extensiones activas tras exclusión: {active_extensions}")

    if not active_extensions:
        print("❌ Error: Todas las extensiones fueron excluidas.")
        return

    combined_images = []
    for file in corrected_files:
        print(f"\n📂 Procesando archivo: {file}")
        with fits.open(file) as hdul:
            nrows = int(hdul[1].header.get('NAXIS2', 1200))
            ncols = int(hdul[1].header.get('NAXIS1', 900))
            primary_hdr = hdul[0].header.copy()
            image_hdr = hdul[1].header.copy()

            valid_data = []
            valid_exts = []
            for ext in active_extensions:
                if ext >= len(hdul):
                    print(f"❌ Ext {ext} no existe en {file} (len={len(hdul)})")
                    continue
                if hdul[ext].data is None:
                    print(f"❌ Ext {ext} está vacía en {file}")
                    continue

                image_data = hdul[ext].data.astype(float)
                if image_data.shape != (nrows, ncols):
                    print(f"⚠️ Dimensiones incorrectas en ext {ext} de {file}. "
                          f"Esperado=({nrows},{ncols}), obtenido={image_data.shape}")
                    continue

                print(f"✅ Ext {ext} válida con shape {image_data.shape}")
                valid_data.append(image_data)
                valid_exts.append(ext)

            if not valid_data:
                print(f"⚠️ No hay extensiones válidas en {file}.")
                continue

            # --- Combinación de extensiones ---
            if comb_mode == 'simple':
                combined_image = np.mean(valid_data, axis=0)
                print(f"ℹ️ Combinación simple de {len(valid_exts)} extensiones")

            elif comb_mode == 'weighted':
                if weights is not None and len(weights) == 16:
                    # Dos interpretaciones: ext-1 o extorder
                    try:
                        cand1 = [float(weights[ext - 1]) for ext in valid_exts]
                    except Exception:
                        cand1 = None
                    try:
                        cand2 = [float(weights[extorder.index(ext)]) for ext in valid_exts]
                    except Exception:
                        cand2 = None

                    if cand1 and sum(cand1) > 0:
                        valid_weights = np.array(cand1)
                        used_mapping = "ext->index (weights[ext-1])"
                    elif cand2 and sum(cand2) > 0:
                        valid_weights = np.array(cand2)
                        used_mapping = "extorder mapping"
                    else:
                        valid_weights = None
                        used_mapping = "none"

                    if valid_weights is None or valid_weights.sum() == 0:
                        print(f"⚠️ Pesos no válidos ({used_mapping}). Usando simple.")
                        combined_image = np.mean(valid_data, axis=0)
                    else:
                        valid_weights /= valid_weights.sum()
                        print(f"ℹ️ Usando pesos con mapeo: {used_mapping}")
                        combined_image = np.average(valid_data, axis=0, weights=valid_weights)

                else:
                    # Calcular SNR internamente
                    sig_box = (460, 257, 510, 280)
                    r1_s, c1_s, r2_s, c2_s = sig_box
                    if r2_s > nrows or c2_s > ncols:
                        print(f"⚠️ sig_box fuera de rango {sig_box}, usando simple")
                        combined_image = np.mean(valid_data, axis=0)
                    else:
                        D = [img[r1_s:r2_s, c1_s:c2_s].ravel() for img in valid_data]
                        snr = [np.nanmean(d) / np.nanstd(d) if np.nanstd(d) > 0 else 0 for d in D]
                        weights_snr = np.array(snr) / np.sum(snr) if np.sum(snr) > 0 else np.ones(len(D)) / len(D)
                        print(f"ℹ️ Pesos calculados por SNR: {np.round(weights_snr,3)}")
                        combined_image = np.average(valid_data, axis=0, weights=weights_snr)
            else:
                print(f"⚠️ Modo {comb_mode} no reconocido. Usando simple.")
                combined_image = np.mean(valid_data, axis=0)

            combined_images.append(combined_image)

    if not combined_images:
        print("❌ No se pudieron combinar imágenes: no hay datos válidos en NINGÚN archivo.")
        return  # 🚫 No guardamos FITS vacío

    final_image = np.mean(combined_images, axis=0)
    primary_hdu = fits.PrimaryHDU(header=primary_hdr)
    image_hdu = fits.ImageHDU(data=final_image, header=image_hdr)
    image_hdu.header['HISTORY'] = f"Averaged {len(valid_exts)} extensions from {len(combined_images)} sci images, mode: {comb_mode}, excluded: {exclude_extensions}"
    hdu = fits.HDUList([primary_hdu, image_hdu])
    hdu.writeto(output_file, overwrite=True)
    print(f"\n✅ Imagen combinada guardada: {output_file}")

def optimize_weights_from_raw(raw_file, sig_box_base=None, exclude_extensions=[], visualize_rois=True):
    """
    Calcula pesos optimizados por SNR para las 16 extensiones de un FITS raw.
    Las ROIs se definen en el formato esperado por roi_shifting: col1, col2, row1, row2.
    """
    exclude_extensions = [int(ext) for ext in exclude_extensions if 1 <= int(ext) <= 16]
    if exclude_extensions:
        print(f"Excluyendo extensiones: {exclude_extensions}")

    weights = [0.0] * 16

    with fits.open(raw_file) as hdul:
        nrows = int(hdul[1].header.get('NAXIS2', 1100))
        ncols = int(hdul[1].header.get('NAXIS1', 895))

        # ROIs en orden natural (1 -> 16)
        if sig_box_base is None:
            sig_box_base = [375, 410, 470, 510] #modify
        ov_box_base = [575, 600, 10, nrows - 10] #[315, 355, 350, 390]

        sig_boxes, _ = roi_shifting(sig_box_base, return_extensions_order=True)
        ov_boxes, _ = roi_shifting(ov_box_base, return_extensions_order=True)

        data = []
        D, X = [], []

        for i in range(16):  # extensiones 1→16
            ext = i + 1
            if ext in exclude_extensions or ext >= len(hdul) or hdul[ext].data is None:
                continue

            img = hdul[ext].data.astype(float)
            c1_s, c2_s, r1_s, r2_s = sig_boxes[i]
            c1_x, c2_x, r1_x, r2_x = ov_boxes[i]

            # Ajustar límites
            c2_s, r2_s = min(c2_s, ncols), min(r2_s, nrows)
            c2_x, r2_x = min(c2_x, ncols), min(r2_x, nrows)
            c1_s, r1_s = max(c1_s, 0), max(r1_s, 0)
            c1_x, r1_x = max(c1_x, 0), max(r1_x, 0)

            if r2_s <= r1_s or c2_s <= c1_s or r2_x <= r1_x or c2_x <= c1_x:
                continue

            data.append(img)
            D.append(img[r1_s:r2_s, c1_s:c2_s].ravel())
            X.append(img[r1_x:r2_x, c1_x:c2_x].ravel())

        if not data:
            print("⚠️ No hay datos válidos. Pesos uniformes.")
            n_active = 16 - len(exclude_extensions)
            for i in range(16):
                weights[i] = 1.0 / n_active if (i + 1) not in exclude_extensions else 0.0
            return weights

        # Visualización opcional
        if visualize_rois:
            fig, ax, im_disp = show_fits_image(data[0])
            rect_sig = Rectangle((0, 0), 1, 1, edgecolor='red', facecolor='none', lw=2, label='Señal')
            rect_ov = Rectangle((0, 0), 1, 1, edgecolor='blue', facecolor='none', lw=2, label='Overscan')
            ax.add_patch(rect_sig)
            ax.add_patch(rect_ov)
            ax.legend()
            ax.set_title("Extensión 1")

            slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
            slider = Slider(slider_ax, 'Extensión', 0, len(data) - 1, valinit=0, valstep=1)

            def update(val):
                idx = int(slider.val)
                im_disp.set_data(data[idx])
                c1_s, c2_s, r1_s, r2_s = sig_boxes[idx]
                c1_x, c2_x, r1_x, r2_x = ov_boxes[idx]
                rect_sig.set_bounds(c1_s, r1_s, c2_s - c1_s, r2_s - r1_s)
                rect_ov.set_bounds(c1_x, r1_x, c2_x - c1_x, r2_x - r1_x)
                ax.set_title(f"Extensión {idx+1}")
                fig.canvas.draw_idle()

            slider.on_changed(update)
            plt.show()

        # Optimización
        def neg_snr(k):
            Dtot = sum(k[i] * D[i] for i in range(len(D)))
            Xtot = sum(k[i] * X[i] for i in range(len(X)))
            std_x = np.std(Xtot, ddof=1)
            if std_x <= 0: return 0
            return -((np.mean(Dtot) - np.mean(Xtot)) / std_x)

        cons = ({'type': 'eq', 'fun': lambda k: np.sum(k) - 1},)
        bounds = [(0.0, 1.0)] * len(D)
        k0 = np.ones(len(D)) / len(D)

        res = minimize(neg_snr, k0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'ftol': 1e-9, 'disp': True, 'maxiter': 500})

        if not res.success:
            print(f"⚠️ Optimización fallida: {res.message}. Pesos uniformes.")
            n_active = len(D)
            for i in range(16):
                weights[i] = 1.0 / n_active if (i + 1) not in exclude_extensions else 0.0
        else:
            print(f"Optimal SNR: {-res.fun:.4f}, Weights: {res.x}")
            j = 0
            for i in range(16):
                if (i + 1) not in exclude_extensions and j < len(res.x):
                    weights[i] = res.x[j]
                    j += 1

    return weights

def calculate_extension_gain_roi(path_files, roi, extension_number, n_points=5, savefigs=False,
                                               rowcols_roi=None):
    """
    Computes the gain of a specific extension of a MAS-CCD Skipper image using dynamic ROI sampling
    and variance-mean analysis. Returns gain value, number of points used, and optionally the gain and ROI figures.

    Args:
        path_files (list of str): List of FITS file paths to be processed. Each exposure time must have at least two files.
        roi (list): Region of interest defined as [x_start, x_end, y_start, y_end].
        extension_number (int): Index of the FITS extension (amplifier) to be analyzed.
        n_points (int, optional): Minimum number of valid data points required to perform the linear fit.
        savefigs (bool or str): If True, save gain figure. If "return", return figure objects. If False, do nothing.

    Returns:
        tuple: (gain [float], number of fit points [int], gain_figure or None, roi_figure or None)
    """
    if rowcols_roi is None:
        rowcols_roi = [9, 8]
    exposure_times = defaultdict(list)
    all_rois = []

    for file_path in path_files:
        with fits.open(file_path) as hdulist:
            exptime = hdulist[0].header.get('EXPTIME')
            if exptime is not None:
                exposure_times[exptime].append(file_path)

    extension_variances = []
    extension_sum_of_means = []

    for exptime in sorted(exposure_times.keys()):
        files = exposure_times[exptime]
        if len(files) >= 2:
            file1, file2 = files[:2]
            with fits.open(file1) as hdul1, fits.open(file2) as hdul2:
                if extension_number >= len(hdul1) or extension_number >= len(hdul2):
                    continue
                data1 = hdul1[extension_number].data
                data2 = hdul2[extension_number].data
                if data1 is None or data2 is None:
                    continue
                x0, x1, y0, y1 = roi
                roi1 = data1[y0:y1, x0:x1]
                roi2 = data2[y0:y1, x0:x1]
                var = np.var(roi2 - roi1)
                mean_sum = np.mean(roi2) + np.mean(roi1)
                if not np.isnan(var) and not np.isnan(mean_sum):
                    extension_variances.append(var)
                    extension_sum_of_means.append(mean_sum)
                    all_rois.append((roi1, roi2, file1, file2))

    if len(extension_variances) < n_points:
        print(f"Not enough data points ({len(extension_variances)}) for extension {extension_number}.")
        return None, 0, None, None

    x = np.array(extension_variances)
    y = np.array(extension_sum_of_means)

    slope, intercept, x_best, y_best, best_indices = best_gain_fit_fast(x, y, n_points=n_points)
    if slope is None:
        print(f"Not enough valid points after robust fit for extension {extension_number}.")
        return None, 0, None, None

    gain = 1 / slope
    num_fit_points = len(best_indices)

    # Plot gain fit
    fig_gain, ax_gain = plt.subplots(figsize=(8, 6))
    ax_gain.plot(x, y, 'o', label=f'All Points ext {extension_number}', color='blue')
    ax_gain.plot(x[best_indices], y[best_indices], 'go', label='Fit Points')
    fit_line = np.polyval([slope, intercept], x[best_indices])
    ax_gain.plot(x_best, fit_line, 'r--', label='Robust Fit')
    ax_gain.text(0.05, 0.95, f"Gain: {gain:.3f} ADU/e-", transform=ax_gain.transAxes, fontsize=10,
                 color='red', verticalalignment='top')
    ax_gain.set_xlabel("Variance of Pixel Difference (ADU²)")
    ax_gain.set_ylabel("Sum of Means (ADU)")
    ax_gain.grid()
    ax_gain.legend()
    fig_gain.tight_layout()

    # Plot ROIs
    fig_rois = None
    if all_rois:
        rows, cols = rowcols_roi[0], rowcols_roi[1]  # It was 6,4
        fig_rois, axes = plt.subplots(rows, cols, figsize=(16, 10)) #It was 20,16
        axes = axes.flatten()

        for i, (roi1_data, roi2_data, file1, file2) in enumerate(all_rois):
            zlow1, zhigh1 = ZScaleInterval().get_limits(roi1_data)
            axes[2 * i].imshow(roi1_data, cmap="gray", clim=(zlow1, zhigh1), extent=[roi[0], roi[1], roi[3], roi[2]])
            axes[2 * i].invert_yaxis()
            contid1 = fits.open(file1)[extension_number].header.get('CONTID', 'N/A')
            exptime1 = fits.open(file1)[0].header.get('EXPTIME', 'N/A')
            axes[2 * i].set_title(f"ROI {os.path.basename(file1)}, EXPTIME={exptime1}", fontsize=8)
            axes[2 * i].text(0.95, 0.05, f"{contid1}", transform=axes[2 * i].transAxes, fontsize=8,
                             verticalalignment='bottom', horizontalalignment='right', alpha=0.8, color="red")

            zlow2, zhigh2 = ZScaleInterval().get_limits(roi2_data)
            axes[2 * i + 1].imshow(roi2_data, cmap="gray", clim=(zlow2, zhigh2), extent=[roi[0], roi[1], roi[3], roi[2]])
            axes[2 * i + 1].invert_yaxis()
            contid2 = fits.open(file2)[extension_number].header.get('CONTID', 'N/A')
            exptime2 = fits.open(file2)[0].header.get('EXPTIME', 'N/A')
            axes[2 * i + 1].set_title(f"ROI {os.path.basename(file2)}, EXPTIME={exptime2}", fontsize=8)
            axes[2 * i + 1].text(0.95, 0.05, f"{contid2}", transform=axes[2 * i + 1].transAxes, fontsize=8,
                             verticalalignment='bottom', horizontalalignment='right', alpha=0.8, color="red")

        fig_rois.tight_layout()

    if savefigs is True:
        gain_base = os.path.splitext(os.path.basename(path_files[0]))[0]
        gain_dir = os.path.dirname(path_files[0])
        fig_gain_path = os.path.join(gain_dir, f"gain_{gain_base}_ext{extension_number}_gainplot.png")
        fig_rois_path = os.path.join(gain_dir, f"gain_{gain_base}_ext{extension_number}_rois.png")
        fig_gain.savefig(fig_gain_path)
        if fig_rois:
            fig_rois.savefig(fig_rois_path)
        print(f"Saved gain plot to {fig_gain_path}")
        if fig_rois:
            print(f"Saved ROI plot to {fig_rois_path}")
        plt.close(fig_gain)
        if fig_rois:
            plt.close(fig_rois)
        return [gain], num_fit_points, None, None
    elif savefigs == "return":
        return [gain], num_fit_points, fig_gain, fig_rois
    else:
        print("Selected not to save or return figures.")
        plt.close(fig_gain)
        if fig_rois:
            plt.close(fig_rois)
        return [gain], num_fit_points, None, None

def visualize_roi_mean_variance(file_path, roi, extension_number):
    """
    Extracts ROI statistics and returns a Matplotlib figure of the ROI.

    Args:
        file_path (str): Path to the input FITS file.
        roi (list): Region of interest defined as [x_start, x_end, y_start, y_end].
        extension_number (int): Index of the FITS extension to access.

    Returns:
        tuple:
            float: Mean pixel value within the ROI.
            float: Standard deviation of pixel values within the ROI.
            matplotlib.figure.Figure: The Matplotlib figure object of the ROI.
    """
    try:
        with fits.open(file_path) as hdul:
            data = hdul[extension_number].data
            if data is None:
                print("Error: No data found in the specified extension.")
                return np.nan, np.nan, None

            x_start, x_end, y_start, y_end = roi
            roi_data = data[y_start:y_end, x_start:x_end]
            mean_counts = np.mean(roi_data)
            std_counts = np.std(roi_data)

            zscale = ZScaleInterval()
            zlow, zhigh = zscale.get_limits(roi_data)

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(roi_data, cmap='gray', clim=(zlow, zhigh), extent=[x_start, x_end, y_end, y_start])
            plt.colorbar(im, label='Counts', ax=ax)
            ax.invert_yaxis() # Invert Y axis
            ax.set_title(f'ROI y Media de Counts: {mean_counts:.2f}, STD: {std_counts}, EXT: {extension_number}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

            return mean_counts, std_counts, fig

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return np.nan, np.nan, None
    except Exception as e:
        print(f"An error happened: {e}")
        return np.nan, np.nan, None

def best_gain_fit_fast(x, y, n_points=5, clip_sigma=0.01, max_iter=10):
    if len(x) < n_points:
        return None, None, None

    indices = np.arange(len(x))
    for _ in range(max_iter):
        model = LinearRegression()
        x_best = x[indices]
        y_best = y[indices]
        model.fit(x_best.reshape(-1, 1), y_best)
        y_pred = model.predict(x[indices].reshape(-1, 1))
        residuals = np.abs(y[indices] - y_pred) / y[indices]

        good = residuals <= clip_sigma

        if np.all(good):
            break
        if np.sum(good) < n_points:
            break
        indices = indices[good]

    if len(indices) >= n_points:
        model = LinearRegression()
        model.fit(x[indices].reshape(-1, 1), y[indices])
        slope = model.coef_[0]
        intercept = model.intercept_
        return slope, intercept, x_best, y_best, indices
    else:
        return None, None, None

def save_readout_noise(path_files_m, filename="rdn_test.txt", gain=None, roi_base=None,
                       return_mean=False, save_txt=False, saveplots=False):
    """
    Computes readout noise and optionally returns/saves plots.

    Args:
        path_files_m (list): List of FITS files.
        filename (str): Output file name to save the readout noise dot txt.
        gain (array): Gains per extensión.
        roi_base (list): ROI base in format [x1, x2, y1, y2].
        return_mean (bool): If means are returned.
        save_txt (bool): If the results are saved in the .txt.
        saveplots (bool or "return"): If it is saved or returned the ROI figures.

    Returns:
        all_noise_data: lista con ruido por extensión para cada imagen.
        all_mean_data: (optional) lista con medias por extensión.
        all_roi_figs: (optional) lista de figures.
    """
    if roi_base is None:
        roi_base = [545, 635, 540, 640]

    extensions = [1, 14, 16, 15, 13, 11, 12, 10, 5, 2, 4, 3, 9, 6, 8, 7]

    if gain is not None:
        try:
            gain = np.array(gain, dtype=float).flatten()
        except Exception as e:
            print(f"Error converting gain to numpy array: {e}.")
            return None, None, None, None

        if len(gain) != len(extensions):
            print(f"Error: gain must be an array of {len(extensions)} elements after flattening.")
            return None, None, None, None

    print("Note: using 895 as NROW. X axis min = 545, max = 635.")

    all_noise_data = []
    all_mean_data = [] if return_mean else None
    all_roi_figs = [] if saveplots == "return" else None

    for path_file in path_files_m:
        rd_noise_all = []
        rd_mean_all = []

        for idx, ext in enumerate(extensions):
            print(f"File: {path_file}, EXT index: {idx + 1}, MAS EXT: {ext}")
            roi = [
                roi_base[0] + (15 * (ext - 1)),
                roi_base[1] + (15 * (ext - 1)),
                roi_base[2],
                roi_base[3],
            ]
            try:
                rd_mean, rd_noise, roi_fig = visualize_roi_mean_variance(
                    path_file, extension_number=idx + 1, roi=roi
                )
            except Exception as e:
                print(f"Error processing file {path_file}, extension {ext}: {e}")
                rd_noise = np.nan
                rd_mean = np.nan
                roi_fig = None

            # Gain correction
            if gain is not None:
                if gain[idx] is not None:
                    rd_noise = rd_noise / gain[idx]
                    rd_mean = rd_mean / gain[idx]
                else:
                    print(f"Warning: Skipping gain correction for EXT {ext} (index {idx})")
                    rd_noise = np.nan
                    rd_mean = np.nan

            rd_noise_all.append(rd_noise)
            if return_mean:
                rd_mean_all.append(rd_mean)
            if saveplots == "return":
                all_roi_figs.append(roi_fig)

        # Now its added ONE TIME per image
        all_noise_data.append(rd_noise_all)
        if return_mean:
            all_mean_data.append(rd_mean_all)

        # Optional save
        if save_txt:
            try:
                string_line_noise = " ".join(f"{x:.5f}" for x in np.array(rd_noise_all))
                with open(filename, "a") as file_noise:
                    file_noise.write(string_line_noise + "\n")
                print(f"Saved noise: {path_file} -> '{filename}'")

                if return_mean:
                    filename_mean = filename.replace(".txt", "_mean.txt")
                    string_line_mean = " ".join(f"{x:.2f}" for x in np.array(rd_mean_all))
                    with open(filename_mean, "a") as file_mean:
                        file_mean.write(string_line_mean + "\n")
                    print(f"Saved mean: {path_file} -> '{filename_mean}'")

                if saveplots is True:
                    print("Saving readout noise ROI plots (not yet implemented)")

            except Exception as e:
                print(f"Error writing to file: {e}")

    if return_mean and saveplots == "return":
        return all_noise_data, all_mean_data, all_roi_figs, None
    elif saveplots == "return":
        return all_noise_data, None, all_roi_figs, None
    elif return_mean:
        return all_noise_data, all_mean_data, None, None
    else:
        return all_noise_data, None, None, None

def find_linear_subset(x, y, window_size_initial, error_threshold=0.01, expansion_step=1, max_iter_refine=5):
    """
    Detects the most linear subset (with an error <= error_threshold) in the
    middle section of the data using a sliding window and expansion.

    Args:
     x (np.ndarray): Array of x values.
     y (np.ndarray): Array of y values.
     window_size_initial (int): Initial size of the centered window.
     error_threshold (float): Maximum acceptable relative error threshold.
     expansion_step (int): Number of points to expand the window at each step.
     max_iter_refine (int): Maximum number of iterations to refine the fit at each window.

    Returns:
     tuple: A tuple containing the indices of the most linear subset found
     and the corresponding relative errors. Returns (None, None) if no subset meeting the threshold is found
    .

    Note: I am using this script to calculate FW, because previous iterations had O(n!),
    which is not feasible at n = 35.
    """
    
    n_total = len(x)
    if n_total < window_size_initial:
        return None, None

    best_indices = None

    # Iterate over the possible initial centers of the window
    for center in range(window_size_initial // 2, n_total - (window_size_initial // 2)):
        start = max(0, center - window_size_initial // 2)
        end = min(n_total, start + window_size_initial)
        current_indices = np.arange(start, end)

        for _ in range(max_iter_refine):
            if len(current_indices) < 2:
                break

            model = LinearRegression()
            model.fit(x[current_indices].reshape(-1, 1), y[current_indices])
            y_pred = model.predict(x[current_indices].reshape(-1, 1))
            residuals = np.abs(y[current_indices] - y_pred) / np.abs(y[current_indices])
            max_error = np.max(residuals) if residuals.size > 0 else np.inf

            if max_error <= error_threshold:
                if len(current_indices) > (len(best_indices) if best_indices is not None else 0):
                    best_indices = current_indices
                break  # The current windows is linearly enough

            # Try to refine the indexes (similar to previous function)
            good_indices_local = current_indices[residuals <= error_threshold]
            if len(good_indices_local) < 2:
                break
            current_indices = good_indices_local

    # Expansion phase of the best windows found
    if best_indices is not None:
        current_start = best_indices[0]
        current_end = best_indices[-1]

        while True:
            can_expand_start = current_start > 0
            can_expand_end = current_end < n_total - 1
            expanded = False

            if can_expand_start:
                new_start = current_start - expansion_step
                test_indices = np.arange(new_start, current_end + 1)
                if len(test_indices) >= 2:
                    model = LinearRegression()
                    model.fit(x[test_indices].reshape(-1, 1), y[test_indices])
                    y_pred = model.predict(x[test_indices].reshape(-1, 1))
                    residuals = np.abs(y[test_indices] - y_pred) / np.abs(y[test_indices])
                    if np.max(residuals) <= error_threshold:
                        current_start = new_start
                        best_indices = np.arange(current_start, current_end + 1)
                        expanded = True

            if can_expand_end:
                new_end = current_end + expansion_step
                test_indices = np.arange(current_start, new_end + 1)
                if len(test_indices) >= 2:
                    model = LinearRegression()
                    model.fit(x[test_indices].reshape(-1, 1), y[test_indices])
                    y_pred = model.predict(x[test_indices].reshape(-1, 1))
                    residuals = np.abs(y[test_indices] - y_pred) / np.abs(y[test_indices])
                    if np.max(residuals) <= error_threshold:
                        current_end = new_end
                        best_indices = np.arange(current_start, current_end + 1)
                        expanded = True

            if not expanded:
                break

        if best_indices is not None:
            model = LinearRegression()
            model.fit(x[best_indices].reshape(-1, 1), y[best_indices])
            y_pred = model.predict(x[best_indices].reshape(-1, 1))
            final_residuals = np.abs(y[best_indices] - y_pred) / np.abs(y[best_indices])
            return best_indices, final_residuals
        else:
            return None, None

    return best_indices, np.abs(y[best_indices] - model.predict(x[best_indices].reshape(-1, 1))) / np.abs(y[best_indices]) if best_indices is not None and len(best_indices) >= 2 else (None, None)

def obtain_fw_data(path_files, roi, extension_number):
    exposure_times = defaultdict(list)
    all_rois = []
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
            file1, file2 = files[:2]
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
    return extension_means, extension_exptimes
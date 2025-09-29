import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip
from astropy.wcs import WCS
from astroscrappy import detect_cosmics
from ..core.core import roi_shifting, get_roi_info

def overscan_correction_combined(input_file, output_file, roi_vector, sci_file=None, method='mean'):
    """
    Aplica corrección de overscan a todas las extensiones de un archivo FITS y recorta al ROI activo.

    Si is_roi=False (por defecto), se usa el ROI estándar [28,539,0,1024].
    Si is_roi=True, se ajusta dinámicamente según la imagen científica.

    Args:
        input_file (str): Ruta al archivo FITS de entrada.
        output_file (str): Ruta al archivo FITS de salida.
        roi_vector (list): Lista de ROIs de overscan por extensión (x_start, x_end, y_start, y_end).
        sci_file (str, optional): Imagen científica para alinear el ROI.
        method (str): Metodo de corrección ('mean' o 'poly').
    """
    # Obtener información de ROI y tamaño según imagen científica si se provee
    skiprow, nrows, ncols, is_roi = get_roi_info(input_file if sci_file is None else sci_file, sci_file=sci_file)

    # Definir ROI activo según tipo de imagen
    if is_roi:
        active_roi_base = [28, 539, 0, nrows]
    else:
        active_roi_base = [28, 539, 0, 1024]

    active_rois = roi_shifting(active_roi_base)

    with fits.open(input_file, mode='readonly') as hdul:
        n_ext = len(hdul) - 1
        master_hdul = fits.HDUList([fits.PrimaryHDU(header=hdul[0].header.copy())])

        for ext in range(1, n_ext + 1):
            if ext >= len(hdul) or hdul[ext].data is None:
                print(f"⚠️ Ext {ext} ausente en {input_file}. Saltando.")
                continue

            data = hdul[ext].data.astype(float)
            roi = roi_vector[ext - 1]
            x_start, x_end, y_start, y_end = roi
            active_roi = active_rois[ext - 1]

            # --- Detectar si input está guardado como ROI ---
            nrows_input = data.shape[0]
            is_roi_input = nrows_input < 1024  # asumiendo detector de 1024 filas

            # Calcular overscan
            overscan_region = data[y_start:y_end, x_start:x_end]
            if method == 'mean':
                overscan_value = np.mean(overscan_region)
            elif method == 'poly':
                coeffs = np.polyfit(np.arange(overscan_region.shape[0]),
                                    np.mean(overscan_region, axis=1), 1)
                overscan_value = np.polyval(coeffs, np.arange(data.shape[0]))[:, np.newaxis]
            else:
                raise ValueError("Método de overscan inválido. Usa 'mean' o 'poly'.")

            # Corregir overscan
            corrected_data = data - overscan_value

            # --- Aplicar SKIPROW solo si input es full-frame y la ciencia es ROI ---
            y0, y1, x0, x1 = active_roi[2], active_roi[3], active_roi[0], active_roi[1]
            if not is_roi_input and is_roi:
                # input full-frame → cortar con SKIPROW para alinear
                y0 = max(0, y0 + skiprow)
                y1 = min(corrected_data.shape[0], y1 + skiprow)
                trimmed_data = corrected_data[y0:y1, x0:x1]
            else:
                # input ya ROI o ciencia full-frame → corte directo
                trimmed_data = corrected_data[y0:y1, x0:x1]

            # Guardar HDU corregida
            header = hdul[ext].header.copy()
            header['HISTORY'] = f'Overscan corregido con metodo {method}'
            header['NAXIS2'] = trimmed_data.shape[0]
            header['NAXIS1'] = trimmed_data.shape[1]
            header['SKIPROW'] = skiprow
            header['ROI'] = is_roi
            header['EXTNAME'] = f'EXT{ext}'

            master_hdul.append(fits.ImageHDU(data=trimmed_data, header=header))
            print(f"✅ Ext {ext}: Input ROI={is_roi_input}, Ciencia ROI={is_roi}, "
                  f"Recortado a {trimmed_data.shape}")
    # 💾 Guardar al final
    master_hdul.writeto(output_file, overwrite=True)
    print(f"💾 Overscan corregido y recortado guardado: {output_file}")

def create_master_bias(bias_files, output_file, combine_type='median', sigma_clip_enabled=True, sigma=3.0, maxiters=5):
    """
    Crea un master bias combinando múltiples bias corregidos por overscan, por extensión.
    Copia el encabezado del primer archivo bias para cada extensión.

    Args:
        bias_files (list): Lista de archivos bias corregidos.
        output_file (str): Ruta para guardar el master bias.
        combine_type (str): 'median' o 'mean'.
        sigma_clip_enabled (bool): Si se aplica sigma clipping antes de combinar.
        sigma (float): Umbral sigma.
        maxiters (int): Iteraciones para clipping.

    Returns:
        None
    """
    if not bias_files:
        raise ValueError("No se entregaron archivos bias.")

    print(f"📥 Cargando {len(bias_files)} archivos bias...")

    # Asumimos que todos tienen la misma cantidad de extensiones.
    with fits.open(bias_files[0]) as hdul_ref:
        n_ext = len(hdul_ref) - 1

    master_hdul = fits.HDUList([fits.PrimaryHDU(header=hdul_ref[0].header.copy())])

    for ext in range(1, n_ext + 1):
        stack = []
        ref_header = None

        for fname in bias_files:
            with fits.open(fname) as hdul:
                if ext >= len(hdul) or hdul[ext].data is None:
                    print(f"⚠️ Ext {ext} ausente en {fname}. Saltando.")
                    continue

                data = hdul[ext].data.astype(float)
                stack.append(data)

                if ref_header is None:
                    ref_header = hdul[ext].header.copy()

        if len(stack) == 0:
            print(f"❌ No se pudo crear master bias para ext {ext}. No hay datos.")
            continue

        stack = np.array(stack)

        if sigma_clip_enabled:
            clipped = sigma_clip(stack, sigma=sigma, axis=0, maxiters=maxiters)
            stack = clipped.data
            print(f"✂️ Sigma clip aplicado en ext {ext} con σ={sigma}")

        if combine_type == 'median':
            combined = np.median(stack, axis=0)
        elif combine_type == 'mean':
            combined = np.mean(stack, axis=0)
        else:
            raise ValueError("Tipo de combinación inválido. Usa 'mean' o 'median'.")

        # Añadir EXTNAME si no está ya en el header
        if ref_header is None:
            ref_header = fits.Header()
        ref_header['EXTNAME'] = f'BIAS{ext}'
        ref_header['HISTORY'] = f"Master bias de {len(stack)} archivos"

        master_hdul.append(fits.ImageHDU(data=combined, header=ref_header))
        print(f"✅ Master bias creado para ext {ext}")

    master_hdul.writeto(output_file, overwrite=True)
    print(f"💾 Master bias guardado: {output_file}")

def bias_subtraction(input_file, master_bias, output_file):
    """
    Resta el master bias de un archivo FITS.

    Args:
        input_file (str): Ruta al archivo FITS de entrada (e.g., 'o_input.fits').
        master_bias (str): Ruta al archivo master bias.
        output_file (str): Ruta al archivo FITS de salida (e.g., 'ob_input.fits').

    Returns:
        None. Guarda el archivo corregido.
    """
    with fits.open(input_file, mode='readonly') as input_hdu, fits.open(master_bias, mode='readonly') as bias_hdu:
        hdu_list = fits.HDUList()
        hdu_list.append(input_hdu[0].copy())
        for ext in range(1, len(input_hdu)):
            if ext >= len(bias_hdu) or input_hdu[ext].data is None or bias_hdu[ext].data is None:
                continue
            corrected_data = input_hdu[ext].data - bias_hdu[ext].data
            hdu = fits.ImageHDU(data=corrected_data, header=input_hdu[ext].header)
            hdu.header['HISTORY'] = 'Bias subtraction applied'
            hdu_list.append(hdu)
        hdu_list.writeto(output_file, overwrite=True)
    print(f"✅ Bias restado: {output_file}")

def create_master_dark(dark_files, output_file, combine_type='median',
                       sigma_clip_enabled=True, sigma=3.0, maxiters=5):
    """
    Crea un master dark combinando múltiples darks corregidos por overscan y bias,
    para cada extensión de un CCD.

    Args:
        dark_files (list): Lista de archivos dark corregidos.
        output_file (str): Ruta para guardar el master dark.
        combine_type (str): 'median' o 'mean'.
        sigma_clip_enabled (bool): Si se aplica sigma clipping antes de combinar.
        sigma (float): Umbral sigma.
        maxiters (int): Iteraciones para clipping.

    Returns:
        None
    """
    if not dark_files:
        raise ValueError("No se entregaron archivos dark.")

    print(f"📥 Cargando {len(dark_files)} archivos dark...")

    # Asumimos que todos tienen la misma cantidad de extensiones
    with fits.open(dark_files[0]) as hdul_ref:
        n_ext = len(hdul_ref) - 1

    master_hdul = fits.HDUList([fits.PrimaryHDU(header=hdul_ref[0].header.copy())])

    for ext in range(1, n_ext + 1):
        stack = []
        ref_header = None

        for fname in dark_files:
            with fits.open(fname) as hdul:
                if ext >= len(hdul) or hdul[ext].data is None:
                    print(f"⚠️ Ext {ext} ausente en {fname}. Saltando.")
                    continue

                data = hdul[ext].data.astype(float)
                stack.append(data)

                if ref_header is None:
                    ref_header = hdul[ext].header.copy()

        if len(stack) == 0:
            print(f"❌ No se pudo crear master dark para ext {ext}. No hay datos.")
            continue

        stack = np.array(stack)

        if sigma_clip_enabled:
            clipped = sigma_clip(stack, sigma=sigma, axis=0, maxiters=maxiters)
            stack = clipped.data
            print(f"✂️ Sigma clip aplicado en ext {ext} con σ={sigma}")

        if combine_type == 'median':
            combined = np.median(stack, axis=0)
        elif combine_type == 'mean':
            combined = np.mean(stack, axis=0)
        else:
            raise ValueError("Tipo de combinación inválido. Usa 'mean' o 'median'.")

        # Añadir EXTNAME si no está ya en el header
        if ref_header is None:
            ref_header = fits.Header()
        ref_header['EXTNAME'] = f'DARK{ext}'
        ref_header['HISTORY'] = f"Master dark de {len(stack)} archivos"

        master_hdul.append(fits.ImageHDU(data=combined, header=ref_header))
        print(f"✅ Master dark creado para ext {ext}")

    master_hdul.writeto(output_file, overwrite=True)
    print(f"💾 Master dark guardado: {output_file}")

def dark_subtraction(input_file, master_dark, output_file):
    """
    Resta el master dark de un archivo FITS.

    Args:
        input_file (str): Ruta al archivo FITS de entrada (e.g., 'ob_input.fits').
        master_dark (str): Ruta al archivo master dark.
        output_file (str): Ruta al archivo FITS de salida (e.g., 'obd_input.fits').

    Returns:
        None. Guarda el archivo corregido.
    """
    with fits.open(input_file, mode='readonly') as input_hdu, fits.open(master_dark, mode='readonly') as dark_hdu:
        hdu_list = fits.HDUList()
        hdu_list.append(input_hdu[0].copy())
        for ext in range(1, len(input_hdu)):
            if ext >= len(dark_hdu) or input_hdu[ext].data is None or dark_hdu[ext].data is None:
                continue
            corrected_data = input_hdu[ext].data - dark_hdu[ext].data
            hdu = fits.ImageHDU(data=corrected_data, header=input_hdu[ext].header)
            hdu.header['HISTORY'] = 'Dark subtraction applied'
            hdu_list.append(hdu)
        hdu_list.writeto(output_file, overwrite=True)
    print(f"✅ Dark restado: {output_file}")


# def create_master_flat(flat_files, master_bias_path, output_file, combine_type='median', sigma_clip_enabled=True, sigma=3.0, maxiters=5):
#     """
#     Crea un master flat a partir de archivos flat corregidos por bias.
#
#     Args:
#         flat_files (list): Lista de archivos flat.
#         master_bias_path (str): Ruta al master bias (se usará para corregir flats).
#         output_file (str): Ruta para guardar el master flat.
#         combine_type (str): A-
#         sigma_clip_enabled (bool): A+
#         sigma (float): B
#         maxiters (int): B-
#     """
#     if not flat_files:
#         raise ValueError("No se entregaron archivos flat.")
#
#     print(f"📥 Procesando {len(flat_files)} archivos flat...")
#
#     with fits.open(flat_files[0]) as hdul:
#         n_ext = len(hdul) - 1
#
#     master_hdul = fits.HDUList([fits.PrimaryHDU()])
#
#     for ext in range(1, n_ext + 1):
#         stack = []
#
#         for fname in flat_files:
#             with fits.open(fname) as flat_hdul, fits.open(master_bias_path) as bias_hdul:
#                 if ext >= len(flat_hdul) or flat_hdul[ext].data is None:
#                     print(f"⚠️ Ext {ext} ausente en {fname}. Saltando.")
#                     continue
#                 if ext >= len(bias_hdul) or bias_hdul[ext].data is None:
#                     print(f"⚠️ Ext {ext} ausente en bias. Saltando.")
#                     continue
#
#                 corrected = flat_hdul[ext].data.astype(float) - bias_hdul[ext].data.astype(float)
#                 stack.append(corrected)
#
#         stack = np.array(stack)
#
#         if sigma_clip_enabled:
#             from astropy.stats import sigma_clip
#             clipped = sigma_clip(stack, sigma=sigma, axis=0, maxiters=maxiters)
#             stack = clipped.data
#             print(f"✂️ Sigma clip aplicado en ext {ext} con σ={sigma}")
#
#         if combine_type == 'median':
#             combined = np.median(stack, axis=0)
#         elif combine_type == 'mean':
#             combined = np.mean(stack, axis=0)
#         else:
#             raise ValueError("Tipo de combinación inválido.")
#
#         norm = np.median(combined)
#         normalized_flat = combined / norm if norm > 0 else combined
#
#         with fits.open(flat_files[0]) as ref_hdul:
#             header = ref_hdul[ext].header.copy()
#
#         header['HISTORY'] = f"Master flat normalizado de {len(stack)} archivos"
#         header['EXTNAME'] = f'FLAT{ext}'
#
#         master_hdul.append(fits.ImageHDU(data=normalized_flat, header=header))
#         print(f"✅ Master flat generado para ext {ext}")
#
#     master_hdul.writeto(output_file, overwrite=True)
#     print(f"💾 Master flat guardado: {output_file}")

def create_master_flat_normalized(flat_files, master_bias_path, output_file, combine_type='median',
                                  sigma_clip_enabled=True, sigma=3.0, maxiters=5, use_dark=False,
                                  master_dark_path=None):
    """
    Crea un master flat normalizado a partir de archivos flat corregidos por bias y opcionalmente por dark.

    Args:
        flat_files (list): Lista de archivos flat.
        master_bias_path (str): Ruta al master bias (usado para corregir flats).
        output_file (str): Ruta para guardar el master flat normalizado.
        combine_type (str): Metodo de combinación: 'median' o 'mean'.
        sigma_clip_enabled (bool): Si aplicar sigma clipping antes de combinar.
        sigma (float): Sigma para el sigma clipping.
        maxiters (int): Iteraciones máximas para sigma clipping.
        use_dark (bool): Si restar el master dark a los flats.
        master_dark_path (str): Ruta al master dark (requerido si use_dark=True).

    Returns:
        None
    """
    if not flat_files:
        raise ValueError("No se entregaron archivos flat.")
    if use_dark and not master_dark_path:
        raise ValueError("Se requiere master_dark_path si use_dark=True.")

    print(f"📥 Procesando {len(flat_files)} archivos flat...")
    with fits.open(flat_files[0]) as hdul:
        n_ext = len(hdul) - 1
        if n_ext != 16:
            print(f"⚠️ El archivo flat {flat_files[0]} tiene {n_ext} extensiones, se esperaban 16.")

    master_hdul = fits.HDUList([fits.PrimaryHDU()])

    for ext in range(1, n_ext + 1):
        stack = []

        for fname in flat_files:
            with fits.open(fname) as flat_hdul, fits.open(master_bias_path) as bias_hdul:
                if ext >= len(flat_hdul) or flat_hdul[ext].data is None:
                    print(f"⚠️ Ext {ext} ausente en {fname}. Saltando.")
                    continue
                if ext >= len(bias_hdul) or bias_hdul[ext].data is None:
                    print(f"⚠️ Ext {ext} ausente en bias. Saltando.")
                    continue

                corrected = flat_hdul[ext].data.astype(float) - bias_hdul[ext].data.astype(float)

                if use_dark:
                    with fits.open(master_dark_path) as dark_hdul:
                        if ext >= len(dark_hdul) or dark_hdul[ext].data is None:
                            print(f"⚠️ Ext {ext} ausente en dark. Saltando.")
                            continue
                        dark_data = dark_hdul[ext].data.astype(float)
                        if corrected.shape != dark_data.shape:
                            print(
                                f"⚠️ Dimensiones incompatibles en ext {ext}: flat {corrected.shape}, dark {dark_data.shape}. Saltando.")
                            continue
                        corrected = corrected - dark_data

                stack.append(corrected)

        if not stack:
            print(f"❌ No se pudo crear master flat para ext {ext} (stack vacío).")
            continue

        stack = np.array(stack)

        if sigma_clip_enabled:
            clipped = sigma_clip(stack, sigma=sigma, axis=0, maxiters=maxiters)
            stack = clipped.data
            print(f"✂️ Sigma clip aplicado en ext {ext} con σ={sigma}")

        if combine_type == 'median':
            combined = np.median(stack, axis=0)
        elif combine_type == 'mean':
            combined = np.mean(stack, axis=0)
        else:
            raise ValueError("Tipo de combinación inválido. Usa 'mean' o 'median'.")

        norm = np.median(combined)
        normalized_flat = combined / norm if norm > 0 else combined

        with fits.open(flat_files[0]) as ref_hdul:
            header = ref_hdul[ext].header.copy()

        header['HISTORY'] = f"Master flat normalizado de {len(stack)} archivos"
        header['EXTNAME'] = f'FLAT{ext}'
        if use_dark:
            header['HISTORY'] = f"Master flat corregido por dark"

        master_hdul.append(fits.ImageHDU(data=normalized_flat, header=header))
        print(f"✅ Master flat normalizado generado para ext {ext}")

    master_hdul.writeto(output_file, overwrite=True)
    print(f"💾 Master flat guardado: {output_file}")


def flat_fielding(input_file, master_flat, output_file):
    """
    Aplica corrección de flat-fielding a un archivo FITS.

    Args:
        input_file (str): Ruta al archivo FITS de entrada (e.g., 'obd_input.fits').
        master_flat (str): Ruta al archivo master flat.
        output_file (str): Ruta al archivo FITS de salida (e.g., 'obdf_input.fits').

    Returns:
        None. Guarda el archivo corregido.
    """
    with fits.open(input_file, mode='readonly') as input_hdu, fits.open(master_flat, mode='readonly') as flat_hdu:
        hdu_list = fits.HDUList()
        hdu_list.append(input_hdu[0].copy())
        for ext in range(1, len(input_hdu)):
            if ext >= len(flat_hdu) or input_hdu[ext].data is None or flat_hdu[ext].data is None:
                continue
            flat_data = flat_hdu[ext].data
            flat_data = np.where(flat_data == 0, 1.0, flat_data)  # Evitar división por cero
            corrected_data = input_hdu[ext].data / flat_data
            hdu = fits.ImageHDU(data=corrected_data, header=input_hdu[ext].header)
            hdu.header['HISTORY'] = 'Flat fielding applied'
            hdu_list.append(hdu)
        hdu_list.writeto(output_file, overwrite=True)
    print(f"✅ Flat-fielding aplicado: {output_file}")

def estimate_readnoise(input_files, roi_vector=None, file=None, gain_vector=None):
    """
    Estima el ruido de lectura (en electrones) para cada extensión de archivos FITS raw.

    Args:
        input_files (str or list): Ruta a un archivo FITS o lista de archivos FITS raw.
        roi_vector (list): Lista de ROIs [col_start, col_end, row_start, row_end] para cada extensión.
        file (str): Ruta al archivo para estimar el ruido (opcional, e.g., bias).
        gain_vector (list): Lista de ganancias (e-/ADU) por extensión, o float si es igual para todas.

    Returns:
        list: Lista de valores de readnoise (en electrones) para cada extensión.
    """
    readnoise_vector = []
    readnoise_vector_ADU = []
    n_extensions = 16  # Asumimos 16 canales para MAS-Skipper CCD
    if isinstance(input_files, str):
        input_files = [input_files]  # Convertir a lista si es un solo archivo
    if isinstance(gain_vector, (int, float)):
        gain_vector = [gain_vector] * n_extensions
    elif gain_vector is None:
        gain_vector = [50] * n_extensions  # Valor por defecto para 'other'

    for ext in range(1, n_extensions + 1):
        variances = []
        for input_file in input_files:
            try:
                with fits.open(input_file, mode='readonly') as hdul:
                    if ext >= len(hdul) or hdul[ext].data is None:
                        print(f"⚠️ Extensión {ext} vacía en {input_file}. Saltando.")
                        continue
                    data = hdul[ext].data
                    if roi_vector and ext - 1 < len(roi_vector):
                        # Usar región de overscan específica para esta extensión
                        overscan_roi = roi_vector[ext - 1]
                        if (overscan_roi[3] > data.shape[0]) or (overscan_roi[1] > data.shape[1]):
                            print(f"⚠️ ROI overscan fuera de rango en ext {ext} de {input_file} (shape: {data.shape}, ROI: {overscan_roi}). Saltando.")
                            continue
                        overscan_data = data[overscan_roi[2]:overscan_roi[3], overscan_roi[0]:overscan_roi[1]]
                        readnoise_adu = np.std(sigma_clip(overscan_data, sigma=3.0, maxiters=5))  # Desviación estándar en ADU
                        variances.append(readnoise_adu ** 2)
                    elif file:
                        # Usar archivo bias
                        with fits.open(file, mode='readonly') as bias_hdu:
                            if ext < len(bias_hdu) and bias_hdu[ext].data is not None:
                                bias_data = bias_hdu[ext].data
                                readnoise_adu = np.std(sigma_clip(bias_data, sigma=3.0, maxiters=5))  # Desviación estándar en ADU
                                variances.append(readnoise_adu ** 2)
                            else:
                                print(f"⚠️ Extensión {ext} no válida en bias file {file}. Usando readnoise por defecto.")
                                readnoise_adu = 3.84 / gain_vector[ext - 1]  # Convertir default a ADU
                                variances.append(readnoise_adu ** 2)
                    else:
                        print(f"⚠️ No roi_vector ni bias file para extensión {ext} en {input_file}. Usando readnoise por defecto.")
                        readnoise_adu = 3.84 / gain_vector[ext - 1]  # Convertir default a ADU
                        variances.append(readnoise_adu ** 2)
            except Exception as e:
                print(f"⚠️ Error procesando {input_file}, ext {ext}: {e}")
                continue

        # Convertir varianza promedio a readnoise en electrones
        if variances:
            readnoise_adu = np.sqrt(np.mean(variances))
            gain = gain_vector[ext - 1] if ext - 1 < len(gain_vector) else gain_vector[0]
            readnoise_e = readnoise_adu / gain if gain != 0 else 3.84
            readnoise_vector.append(readnoise_e)
            readnoise_vector_ADU.append(readnoise_adu)
        else:
            print(f"⚠️ No se pudo calcular readnoise para ext {ext}. Usando 3.84.")
            readnoise_vector.append(3.84)

    print(f"📊 Readout-noise calculado por extensión (en e-): {readnoise_vector}")
    return readnoise_vector_ADU

def cosmic_ray_correction(input_file, output_file, sigclip=4.5, sigfrac=0.3, objlim=5.0,
                          gain_vector=None, satlevel_vector=None, roi_vector=None, file=None,
                          readnoise_vector=None):
    """
    Corrige rayos cósmicos en un archivo FITS usando LACosmic (astroscrappy).

    Args:
        input_file (str): Ruta al archivo FITS de entrada.
        output_file (str): Ruta al archivo FITS de salida.
        sigclip (float): Umbral de detección en desviaciones estándar.
        sigfrac (float): Fracción de sigclip para píxeles vecinos.
        objlim (float): Límite de contraste para objetos reales.
        gain_vector (list): Lista de ganancias (e-/ADU) por extensión, o float si es igual para todas.
        satlevel_vector (list): Lista de niveles de saturación (ADU) por extensión, o float si es igual.
        roi_vector (list): Lista de ROIs [col_start, col_end, row_start, row_end] para cada extensión (opcional).
        file (str): Ruta al archivo para calcular readnoise (opcional).
        readnoise_vector (list): Lista de valores de readnoise (en electrones) por extensión (opcional).

    Returns:
        None. Guarda el archivo corregido.
    """
    with fits.open(input_file, mode='readonly') as hdul:
        hdu_list = fits.HDUList()
        hdu_list.append(hdul[0].copy())  # Copiar header primario

        # Normalizar gain y satlevel como listas
        n_extensions = len(hdul) - 1
        if isinstance(gain_vector, (int, float)):
            gain_vector = [gain_vector] * n_extensions
        if isinstance(satlevel_vector, (int, float)):
            satlevel_vector = [satlevel_vector] * n_extensions

        # Usar readnoise_vector proporcionado o calcularlo
        if readnoise_vector is None:
            readnoise_vector = estimate_readnoise(input_file, roi_vector=roi_vector, file=file,
                                                  gain_vector=gain_vector)
        for ext in range(1, len(hdul)):
            data = hdul[ext].data
            if data is None:
                print(f"⚠️ Extensión {ext} vacía en {input_file}. Saltando.")
                continue

            # Obtener parámetros para esta extensión
            gain = gain_vector[ext - 1] if ext - 1 < len(gain_vector) else gain_vector[0]
            satlevel = satlevel_vector[ext - 1] if ext - 1 < len(satlevel_vector) else satlevel_vector[0]
            readnoise = readnoise_vector[ext - 1] if ext - 1 < len(readnoise_vector) else 3.84
            print(f"Ext {ext}: readnoise={readnoise}, gain={gain}, satlevel={satlevel}")
            # Aplicar LACosmic
            crmask, clean_data = detect_cosmics(
                data, sigclip=sigclip, sigfrac=sigfrac, objlim=objlim,
                gain=gain, readnoise=readnoise, satlevel=satlevel,
                cleantype='medmask', fsmode='median'
            )
            hdu = fits.ImageHDU(data=clean_data, header=hdul[ext].header)
            hdu.header[
                'HISTORY'] = f'Cosmic ray correction applied with LACosmic (gain={gain}, readnoise={readnoise}, satlevel={satlevel})'
            hdu_list.append(hdu)
        hdu_list.writeto(output_file, overwrite=True)
    print(f"✅ Rayos cósmicos corregidos: {output_file}")

def add_wcs(input_file, output_file):
    """
    Añade coordenadas WCS a una imagen FITS usando astropy.wcs, extrayendo RA y DEC del HDU[0].

    Args:
        input_file (str): Ruta al archivo FITS de entrada (e.g., fbo_*.fits o fdbo_*.fits).
        output_file (str): Ruta al archivo FITS de salida con WCS.

    Returns:
        None. Guarda el archivo con WCS actualizado.
    """
    with fits.open(input_file, mode='readonly') as hdul:
        hdu_list = fits.HDUList()
        hdu_list.append(hdul[0].copy())

        # Extraer RA y DEC del HDU[0]
        primary_header = hdul[0].header
        if 'RA' in primary_header and 'DEC' in primary_header:
            try:
                # Usar SkyCoord para parsear RA y DEC (sexagesimal o decimal)
                coord = SkyCoord(primary_header['RA'], primary_header['DEC'], unit=(u.hourangle, u.deg))
                ra = coord.ra.deg  # Convertir a grados
                dec = coord.dec.deg
            except ValueError as e:
                print(f"⚠️ Error al parsear RA/DEC en HDU[0] de {input_file}: {e}. Usando valores por defecto.")
                ra = 0.0  # RA en grados
                dec = 0.0  # Dec en grados
        else:
            # Coordenadas por defecto para punto vernal
            ra = 0.0  # RA en grados
            dec = 0.0  # Dec en grados
            print(f"⚠️ No se encontraron RA/DEC en HDU[0] de {input_file}. Usando RA={ra}, Dec={dec}.")

        # Escala fija
        scale = 0.2546  # arcsec/píxel

        for ext in range(1, len(hdul)):
            if hdul[ext].data is None:
                print(f"⚠️ Extensión {ext} vacía en {input_file}. Saltando.")
                continue

            data = hdul[ext].data
            header = hdul[ext].header

            # Verificar si ya hay WCS
            if 'CRVAL1' in header and 'CRVAL2' in header:
                print(f"📍 WCS ya presente en ext {ext} de {input_file}. Manteniendo header.")
                hdu = fits.ImageHDU(data=data, header=header)
                hdu_list.append(hdu)
                continue

            # Dimensiones de la imagen
            naxis2, naxis1 = data.shape

            # Crear WCS
            wcs = WCS(naxis=2)
            wcs.wcs.crpix = [naxis1 / 2, naxis2 / 2]  # Píxel de referencia en el centro
            wcs.wcs.crval = [ra, dec]  # Coordenadas del píxel de referencia
            wcs.wcs.cdelt = [(scale / 3600.0), scale / 3600.0]  # Escala en grados/píxel, RA hacia izquierda
            wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']  # Proyección tangencial
            wcs.wcs.cunit = ['deg', 'deg']  # Unidades
            wcs.wcs.pc = [[1.0, 0.0], [0.0, 1.0]]  # Sin rotación (montura ecuatorial)

            # Actualizar header con WCS
            header.update(wcs.to_header())
            header['HISTORY'] = f'WCS added using astropy.wcs (RA={ra}, Dec={dec}, scale={scale} arcsec/pix)'

            hdu = fits.ImageHDU(data=data, header=header)
            hdu_list.append(hdu)

        hdu_list.writeto(output_file, overwrite=True)
    print(f"✅ WCS añadido: {output_file}")
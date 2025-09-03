#redmas.py
#Reducción básica para imágenes MAS-Skipper CCD

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip


def overscan_correction_combined(input_file, output_file, roi_vector, method='mean'):
    """
    Aplica corrección de overscan a todas las extensiones de un archivo FITS.

    Args:
        input_file (str): Ruta al archivo FITS de entrada.
        output_file (str): Ruta al archivo FITS de salida (e.g., 'o_input.fits').
        roi_vector (list): Lista de ROIs para cada extensión.
        method (str): Método de corrección ('mean' o 'poly').

    Returns:
        None. Guarda el archivo corregido.
    """
    with fits.open(input_file, mode='readonly') as hdul:
        hdu_list = fits.HDUList()
        hdu_list.append(hdul[0].copy())  # Copiar header primario
        for ext, roi in enumerate(roi_vector, start=1):
            if ext >= len(hdul) or hdul[ext].data is None:
                continue
            data = hdul[ext].data
            overscan_region = data[roi[2]:roi[3], roi[0]:roi[1]]
            if method == 'mean':
                overscan_value = np.mean(overscan_region)
            else:
                # Implementación con ajuste polinómico (simplificado)
                x = np.arange(overscan_region.shape[1])
                coeffs = np.polyfit(x, np.mean(overscan_region, axis=0), 1)
                overscan_value = np.polyval(coeffs, x).mean()
            corrected_data = data - overscan_value
            hdu = fits.ImageHDU(data=corrected_data, header=hdul[ext].header)
            hdu.header['HISTORY'] = f'Overscan correction applied using {method}'
            hdu_list.append(hdu)
        hdu_list.writeto(output_file, overwrite=True)
    print(f"✅ Overscan corregido: {output_file}")

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

#redmas.py
#Reducción básica para imágenes MAS-Skipper CCD

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip

def overscan_correction_combined(file, output_file, roi_vector, method='mean'):
    """
    Aplica corrección de overscan a un archivo FITS con múltiples extensiones.

    Args:
        file (str): Ruta al archivo FITS de entrada.
        output_file (str): Ruta para guardar el archivo corregido.
        roi_vector (list): Lista con 16 ROIs, uno por extensión, como [col1, col2, row1, row2].
        method (str): Metodo para estimar el nivel de overscan ('mean' o 'poly').

    Returns:
        None. Guarda el archivo FITS corregido.
    """
    if len(roi_vector) < 1:
        raise ValueError("El vector ROI debe contener al menos una región.")

    with fits.open(file, mode='readonly') as hdul:
        corrected_hdul = fits.HDUList([fits.PrimaryHDU(header=hdul[0].header.copy())])

        for ext in range(1, len(hdul)):
            if hdul[ext].data is None:
                print(f"⚠️ Ext {ext} sin datos. Saltando.")
                continue

            if ext - 1 >= len(roi_vector):
                print(f"⚠️ No se definió ROI para ext {ext}. Saltando.")
                continue

            data = hdul[ext].data.astype(float)
            col1, col2, row1, row2 = roi_vector[ext - 1]

            if (row2 > data.shape[0]) or (col2 > data.shape[1]):
                print(f"⚠️ ROI fuera de rango en ext {ext}. Saltando.")
                corrected_hdul.append(fits.ImageHDU(data=data, header=hdul[ext].header))
                continue

            overscan_region = data[row1:row2, col1:col2]

            if overscan_region.size == 0:
                print(f"⚠️ Región de overscan vacía en ext {ext}. Saltando.")
                corrected_hdul.append(fits.ImageHDU(data=data, header=hdul[ext].header))
                continue

            if method == 'mean':
                overscan_value = np.mean(overscan_region)
            elif method == 'poly':
                x = np.arange(col2 - col1)
                y = np.mean(overscan_region, axis=0)
                poly_coef = np.polyfit(x, y, deg=2)
                poly_fit = np.polyval(poly_coef, x)
                overscan_value = np.mean(poly_fit)
            else:
                raise ValueError("Método debe ser 'mean' o 'poly'.")

            corrected_data = data - overscan_value
            corrected_hdul.append(fits.ImageHDU(data=corrected_data, header=hdul[ext].header))

            print(f"✅ Ext {ext}: Overscan = {overscan_value:.3f}")

    corrected_hdul.writeto(output_file, overwrite=True)
    print(f"💾 Guardado archivo corregido: {output_file}")

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


def bias_subtraction(path_files, master_bias_file, output_file):
    """
    Aplica sustracción de bias a un archivo FITS con múltiples extensiones.

    Args:
        path_files (str): Ruta al archivo de ciencia (overscan corregido).
        master_bias_file (str): Ruta al master bias.
        output_file (str): Ruta de salida.

    Returns:
        None
    """
    with fits.open(path_files, mode='readonly') as sci_hdul, fits.open(master_bias_file, mode='readonly') as bias_hdul:
        corrected_hdul = fits.HDUList([fits.PrimaryHDU(header=sci_hdul[0].header.copy())])

        for ext in range(1, 17):
            if ext >= len(sci_hdul) or sci_hdul[ext].data is None:
                print(f"⚠️ Ext {ext} no existe en ciencia. Saltando.")
                continue

            if ext >= len(bias_hdul) or bias_hdul[ext].data is None:
                print(f"⚠️ Ext {ext} no existe en bias. Saltando.")
                continue

            science_data = sci_hdul[ext].data.astype(float)
            bias_data = bias_hdul[ext].data.astype(float)

            corrected_data = science_data - bias_data
            corrected_hdu = fits.ImageHDU(data=corrected_data, header=sci_hdul[ext].header)
            corrected_hdul.append(corrected_hdu)

            print(f"🔧 Sustracción de bias aplicada a ext {ext}")

        corrected_hdul.writeto(output_file, overwrite=True)
        print(f"💾 Bias corregido guardado en: {output_file}")


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


def dark_subtraction(path_files, master_dark_file, output_file):
    """
    Aplica sustracción de dark a un archivo FITS con múltiples extensiones.

    Args:
        path_files (str): Ruta al archivo de ciencia (ya corregido por overscan/bias).
        master_dark_file (str): Ruta al master dark.
        output_file (str): Ruta de salida.

    Returns:
        None
    """
    with fits.open(path_files, mode='readonly') as sci_hdul, fits.open(master_dark_file, mode='readonly') as dark_hdul:
        # Copiar encabezado principal
        corrected_hdul = fits.HDUList([fits.PrimaryHDU(header=sci_hdul[0].header.copy())])

        # Iterar sobre extensiones de ciencia
        for ext in range(1, len(sci_hdul)):
            if sci_hdul[ext].data is None:
                print(f"⚠️ Ext {ext} no existe en ciencia. Saltando.")
                continue

            if ext >= len(dark_hdul) or dark_hdul[ext].data is None:
                print(f"⚠️ Ext {ext} no existe en dark. Saltando.")
                continue

            science_data = sci_hdul[ext].data.astype(float)
            dark_data = dark_hdul[ext].data.astype(float)

            # Sustracción de dark
            corrected_data = science_data - dark_data

            corrected_hdu = fits.ImageHDU(data=corrected_data, header=sci_hdul[ext].header)
            corrected_hdul.append(corrected_hdu)

            print(f"🔧 Sustracción de dark aplicada a ext {ext}")

        # Guardar archivo corregido
        corrected_hdul.writeto(output_file, overwrite=True)
        print(f"💾 Dark corregido guardado en: {output_file}")

def create_master_flat(flat_files, master_bias_path, output_file, combine_type='median', sigma_clip_enabled=True, sigma=3.0, maxiters=5):
    """
    Crea un master flat a partir de archivos flat corregidos por bias.

    Args:
        flat_files (list): Lista de archivos flat.
        master_bias_path (str): Ruta al master bias (se usará para corregir flats).
        output_file (str): Ruta para guardar el master flat.
        combine_type (str): A-
        sigma_clip_enabled (bool): A+
        sigma (float): B
        maxiters (int): B-
    """
    if not flat_files:
        raise ValueError("No se entregaron archivos flat.")

    print(f"📥 Procesando {len(flat_files)} archivos flat...")

    with fits.open(flat_files[0]) as hdul:
        n_ext = len(hdul) - 1

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
                stack.append(corrected)

        stack = np.array(stack)

        if sigma_clip_enabled:
            from astropy.stats import sigma_clip
            clipped = sigma_clip(stack, sigma=sigma, axis=0, maxiters=maxiters)
            stack = clipped.data
            print(f"✂️ Sigma clip aplicado en ext {ext} con σ={sigma}")

        if combine_type == 'median':
            combined = np.median(stack, axis=0)
        elif combine_type == 'mean':
            combined = np.mean(stack, axis=0)
        else:
            raise ValueError("Tipo de combinación inválido.")

        norm = np.median(combined)
        normalized_flat = combined / norm if norm > 0 else combined

        with fits.open(flat_files[0]) as ref_hdul:
            header = ref_hdul[ext].header.copy()

        header['HISTORY'] = f"Master flat normalizado de {len(stack)} archivos"
        header['EXTNAME'] = f'FLAT{ext}'

        master_hdul.append(fits.ImageHDU(data=normalized_flat, header=header))
        print(f"✅ Master flat generado para ext {ext}")

    master_hdul.writeto(output_file, overwrite=True)
    print(f"💾 Master flat guardado: {output_file}")


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

def flat_fielding(science_file, master_flat_file, output_file):
    """
    Aplica división por master flat normalizado (media ~ 1).

    Args:
        science_file (str): Ruta al archivo de ciencia.
        master_flat_file (str): Ruta al master flat.
        output_file (str): Ruta de salida.

    Returns:
        None
    """
    with fits.open(science_file) as sci_hdul, fits.open(master_flat_file) as flat_hdul:
        hdr0 = sci_hdul[0].header.copy()
        out_hdul = fits.HDUList([fits.PrimaryHDU(header=hdr0)])

        for ext in range(1, len(sci_hdul)):
            sci = sci_hdul[ext].data.astype(float)
            flat = flat_hdul[ext].data.astype(float)

            if flat is None:
                print(f"⚠️ Flat inexistente en ext {ext}. Saltando flat-field.")
                out_hdul.append(sci_hdul[ext])
                continue

            if sci.shape != flat.shape:
                print(f"⚠️ Dimensiones incompatibles en ext {ext}: ciencia {sci.shape}, flat {flat.shape}. Saltando.")
                continue

            corrected = sci / flat
            out_hdul.append(fits.ImageHDU(data=corrected, header=sci_hdul[ext].header))
            print(f"✅ Flat-field aplicado en ext {ext}")

        out_hdul.writeto(output_file, overwrite=True)
        print(f"💾 Flat-field completo guardado en: {output_file}")

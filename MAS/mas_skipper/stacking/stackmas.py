import numpy as np
import astroalign as aa
from astropy.io import fits
from astropy.stats import sigma_clip
import warnings
import os

# Suprimimos advertencias de NaN durante operaciones de slice, comunes al usar máscaras
warnings.filterwarnings("ignore", category=RuntimeWarning)

# CAMBIAR CON KEYWORDS REALES - Diccionario de Longitud de Onda Central (nm) aproximada
# Esto sirve para ordenar: El menor va al canal Azul, el medio al Verde, el mayor al Rojo.
FILTER_WAVELENGTHS = {
    # Johnson-Cousins
    'U': 365, 'B': 445, 'V': 551, 'R': 658, 'I': 806,
    # Sloan
    'u': 354, 'g': 475, 'r': 622, 'i': 763, 'z': 905,
    # Genericos/Otros
    'Luminance': 500, 'Clear': 500, 'Halpha': 656, 'OIII': 500, 'SII': 672
}


def validate_inputs(files, mode='deep', ext=1):
    """
    Verifica que los archivos cumplan las reglas del modo seleccionado.
    Returns:
        bool: True si pasa la validación.
        list/dict: Info procesada (lista de archivos o diccionario RGB).
    """
    filters_found = {}  # {filepath: filter_name}

    print(f"🔍 Validando {len(files)} archivos para modo {mode.upper()}...")

    for f in files:
        flt = get_filter_from_header(f, ext=ext)
        filters_found[f] = flt
        # print(f"   📄 {os.path.basename(f)} -> Filtro: {flt}")

    unique_filters = set(filters_found.values())

    if mode == 'deep':
        if len(unique_filters) > 1:
            print(f"❌ Error modo DEEP: Se encontraron múltiples filtros: {unique_filters}.")
            print("   Para apilado profundo, todas las imágenes deben ser del mismo filtro.")
            return False, None
        print(f"✅ Validación DEEP correcta. Filtro único: {list(unique_filters)[0]}")
        return True, files

    elif mode == 'rgb':
        if len(files) != 3:
            print(f"⚠️ Advertencia modo RGB: Se recomienda usar exactamente 3 imágenes. Se encontraron {len(files)}.")
            if len(files) < 3:
                print("❌ Error: Necesitas al menos 3 imágenes para RGB.")
                return False, None

        if len(unique_filters) < 3:
            print(f"❌ Error modo RGB: Se necesitan 3 filtros distintos. Encontrados: {unique_filters}")
            return False, None

        # Ordenar archivos por longitud de onda
        # 1. Asignar lambda a cada archivo
        files_with_lambda = []
        for f, flt in filters_found.items():
            # Buscar coincidencia parcial (ej 'B' in 'B_band')
            wl = 500  # Default green
            for key, val in FILTER_WAVELENGTHS.items():
                if key in flt or flt in key:
                    wl = val
                    break
            files_with_lambda.append((wl, f, flt))

        # 2. Ordenar de menor a mayor longitud de onda (Azul -> Rojo)
        sorted_files = sorted(files_with_lambda, key=lambda x: x[0])

        # Asumimos: Primero=Azul, Medio=Verde, Último=Rojo
        # Si hay más de 3, tomamos los extremos y el del medio.
        b_idx = 0
        r_idx = len(sorted_files) - 1
        g_idx = len(sorted_files) // 2

        rgb_dict = {
            'B': sorted_files[b_idx],  # (wl, path, filtername)
            'G': sorted_files[g_idx],
            'R': sorted_files[r_idx]
        }

        print(f"🌈 Asignación RGB Automática (basada en U->I):")
        print(f"   🔵 BLUE Channel:  {rgb_dict['B'][2]} ({os.path.basename(rgb_dict['B'][1])})")
        print(f"   🟢 GREEN Channel: {rgb_dict['G'][2]} ({os.path.basename(rgb_dict['G'][1])})")
        print(f"   🔴 RED Channel:   {rgb_dict['R'][2]} ({os.path.basename(rgb_dict['R'][1])})")

        # Devolvemos la lista ordenada para alinear: Referencia idealmente es el VERDE (medio)
        return True, rgb_dict

    return False, None

def get_filter_from_header(file_path, ext=0):
    """Obtiene el filtro del header intentando varias keywords comunes."""
    with fits.open(file_path) as hdul:
        header = hdul[ext if ext < len(hdul) else 0].header
        # Intenta buscar FILTER, o FILTER1, o FILTERS
        flt = header.get('FILTER') or header.get('FILTERS') or header.get('FILTER1')
        if flt:
            return flt.strip()
        return "Unknown"

def load_and_align_images(file_list, reference_idx=0, extension=1):
    """
    Carga una lista de archivos FITS y los alinea respecto a una imagen de referencia
    usando triangulación de fuentes (astroalign).

    Args:
        file_list (list): Lista de rutas a archivos FITS.
        reference_idx (int): Índice del archivo en file_list que servirá de referencia (WCS base).
        extension (int): Extensión FITS donde está la data (usualmente 1 para imágenes comprimidas/procesadas).

    Returns:
        tuple:
            - aligned_stack (np.array): Cubo 3D de imágenes alineadas (N_img, Y, X).
            - ref_header (fits.Header): Encabezado de la imagen de referencia.
            - success_list (list): Lista de nombres de archivos que se alinearon con éxito.
    """
    ref_file = file_list[reference_idx]
    print(f"⭐ Referencia de alineación: {ref_file}")

    # Cargar referencia
    with fits.open(ref_file) as hdul:
        # Intentar leer la extensión indicada, si falla probar la 0
        if extension < len(hdul):
            ref_data = hdul[extension].data.astype("float32")
            ref_header = hdul[extension].header.copy()
        else:
            print(f"⚠️ Extensión {extension} no encontrada en referencia. Usando Ext 0.")
            ref_data = hdul[0].data.astype("float32")
            ref_header = hdul[0].header.copy()

    # La referencia siempre es la primera capa del stack
    aligned_images = [ref_data]
    success_files = [ref_file]

    # Iterar sobre el resto
    for i, file_path in enumerate(file_list):
        if i == reference_idx:
            continue

        try:
            with fits.open(file_path) as hdul:
                # Selección de extensión robusta
                target_ext = extension if extension < len(hdul) else 0
                source_data = hdul[target_ext].data.astype("float32")

            # --- AQUI OCURRE LA MAGIA DE ASTROALIGN ---
            # register devuelve la imagen transformada.
            # fill_value=np.nan es CRÍTICO para que los bordes negros no afecten la estadística.
            registered_image, _ = aa.register(source_data, ref_data, fill_value=np.nan)

            aligned_images.append(registered_image)
            success_files.append(file_path)
            print(f"✅ Alineado: {file_path}")

        except aa.MaxIterError:
            print(f"⚠️ Fallo de alineación (no se hallaron suficientes triángulos): {file_path}")
        except Exception as e:
            print(f"❌ Error procesando {file_path}: {e}")

    if len(aligned_images) < 2:
        print("⚠️ Advertencia: No hay suficientes imágenes para apilar.")
        return None, None, []

    # Convertir lista a array numpy 3D
    aligned_stack = np.array(aligned_images)
    return aligned_stack, ref_header, success_files

def combine_stack(stack, method='median', sigma=3.0, maxiters=5):
    """
    Combina un cubo de imágenes alineadas en una imagen 2D profunda.

    Args:
        stack (np.array): Cubo 3D (N, Y, X) con valores float y NaNs en bordes.
        method (str): 'median', 'mean', o 'sigmaclip'.
        sigma (float): Umbral para sigma clipping.
        maxiters (int): Iteraciones para sigma clipping.

    Returns:
        np.array: Imagen 2D combinada (Master Science).
    """
    print(f"📚 Combinando stack de {stack.shape[0]} imágenes usando método: {method.upper()}...")

    if method == 'median':
        # Usamos nanmedian para ignorar los bordes rellenos de NaN por la rotación
        final_image = np.nanmedian(stack, axis=0)

    elif method == 'mean':
        final_image = np.nanmean(stack, axis=0)

    elif method == 'sigmaclip':
        print(f"✂️ Ejecutando Sigma Clipping (σ={sigma}, iters={maxiters}). Esto puede tardar...")
        # astropy.stats.sigma_clip devuelve un array enmascarado (MaskedArray)
        # axis=0 opera a través del tiempo (z-axis del cubo)
        clipped_stack = sigma_clip(stack, sigma=sigma, maxiters=maxiters, axis=0, copy=False)

        # Calculamos la media ignorando los valores enmascarados (clippeados) y los NaNs
        # np.ma.mean calcula la media sobre la máscara
        final_image = np.ma.mean(clipped_stack, axis=0).data

        # Si quedan NaNs (zonas sin cobertura), rellenar o dejar como nan
        # (El .data de un masked array pone valores por defecto donde hay mascara, cuidado)
        # Una forma más segura para astronomía es convertir la máscara a NaN y usar nanmean:
        filled_data = np.where(clipped_stack.mask, np.nan, clipped_stack.data)
        final_image = np.nanmean(filled_data, axis=0)

    else:
        raise ValueError(f"Método desconocido: {method}")

    return final_image

def save_coadd(output_path, data, header, input_files, method):
    """
    Guarda la imagen final actualizando el header con metadatos del proceso.
    """
    header['HISTORY'] = f"Stacked {len(input_files)} images using MAS-STACK method {method}"
    header['NCOMBINE'] = (len(input_files), "Number of images combined")

    # Opcional: Listar archivos usados en HISTORY (puede ser largo)
    # for f in input_files:
    #     header['HISTORY'] = f"Used: {os.path.basename(f)}"

    fits.writeto(output_path, data, header, overwrite=True)
    print(f"💾 Imagen profunda guardada en: {output_path}")
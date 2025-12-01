import numpy as np
import astroalign as aa
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.visualization import ZScaleInterval, make_lupton_rgb
import warnings
import os
import matplotlib.pyplot as plt

# Suprimimos advertencias de NaN durante operaciones de slice, comunes al usar máscaras
warnings.filterwarnings("ignore", category=RuntimeWarning)

# CAMBIAR CON KEYWORDS REALES - Diccionario de Longitud de Onda Central (nm) aproximada
# Esto sirve para ordenar: El menor va al canal Azul, el medio al Verde, el mayor al Rojo.
# --- DICCIONARIO DE FILTROS MAESTRO ---
# Mapea nombres de filtros a longitud de onda central (nm).
# Usado para ordenar canales RGB (Azul -> Rojo).
FILTER_WAVELENGTHS = {
    # --- Johnson-Cousins ---
    'u': 365, 'b': 445, 'nv': 551, 'r': 658, 'i': 806,
    'ov': 550,  # Old visual in CTIO

    # --- SDSS (Sloan) ---
    'u_sdss': 354,
    'g_sdss': 477,
    'r_sdss': 623,
    'i_sdss': 763,
    'z_sdss': 913,

    # --- Gunn-Thuan ---
    # Similares a SDSS pero históricos.
    'g_gunnthuan': 490,
    'r_gunnthuan': 655,
    'i_gunnthuan': 780,
    'z_gunnthuan': 900,

    # --- MCELS / Banda Estrecha ---
    'o3_winkler': 500.7,  # OIII Oxígeno
    'grn_cont': 520,  # Green Continuum (usualmente desplazado de OIII)
    'ha': 656.3,  # H-Alpha
    'red_cont': 660,  # Red Continuum (si existiera)
    's2': 672,  # Sulfur II (común en MCELS)

    # --- Otros / Genéricos ---
    'clear': 500, 'lum': 500
}

def validate_inputs(files, mode='deep', ext=1):
    """Verifica filtros y organiza archivos para Deep o RGB."""
    filters_found = {}
    print(f"🔍 Validando {len(files)} archivos para modo {mode.upper()}...")

    for f in files:
        flt = get_filter_from_header(f, ext=ext) # Usamos ext para leer header
        filters_found[f] = flt

    unique_filters = set(filters_found.values())
    print(f"   ℹ️ Filtros detectados (limpios): {unique_filters}")

    if mode == 'deep':
        if len(unique_filters) > 1:
            print(f"❌ Error modo DEEP: Múltiples filtros encontrados: {unique_filters}")
            return False, None
        print(f"✅ Validación DEEP correcta. Filtro: {list(unique_filters)[0]}")
        return True, files

    elif mode == 'rgb':
        if len(unique_filters) < 3:
            print(f"⚠️ Advertencia RGB: Se recomiendan 3 filtros distintos. Encontrados: {unique_filters}")

        files_with_lambda = []
        for f, flt in filters_found.items():
            wl = 500
            if flt in FILTER_WAVELENGTHS:
                wl = FILTER_WAVELENGTHS[flt]
            else:
                for key_wl, val_wl in FILTER_WAVELENGTHS.items():
                    if key_wl in flt:
                        wl = val_wl
                        break
            files_with_lambda.append((wl, f, flt))

        sorted_files = sorted(files_with_lambda, key=lambda x: x[0])
        b_idx, r_idx = 0, len(sorted_files) - 1
        g_idx = len(sorted_files) // 2

        rgb_dict = {'B': sorted_files[b_idx], 'G': sorted_files[g_idx], 'R': sorted_files[r_idx]}
        print(f"🌈 Asignación RGB: B={rgb_dict['B'][2]}, G={rgb_dict['G'][2]}, R={rgb_dict['R'][2]}")
        return True, rgb_dict

    return False, None

def get_filter_from_header(file_path, ext=0):
    """
    Analiza el keyword FILTERS y extrae el filtro real ignorando posiciones vacías.
    Ejemplos de entrada en header:
      'OPEN5 g_gunnthuan' -> Retorna 'g_gunnthuan'
      'grn_cont OPEN2'    -> Retorna 'grn_cont'
      'dia ov'            -> Retorna 'ov'
      'dia ha'            -> Retorna 'ha'
    """
    # Lista negra de palabras que NO son filtros reales
    ignore_keyword_values = ['open', 'dia', "nd", 'empty', 'blank', 'air', 'clear']

    with fits.open(file_path) as hdul:
        target_ext = ext if ext < len(hdul) else 0
        header = hdul[target_ext].header

        # Obtener valor crudo
        raw_val = header.get('FILTERS') or header.get('FILTER') or header.get('FILTER1') or "unknown"

        if not isinstance(raw_val, str):
            return "unknown"

        # 1. Limpiar y dividir por espacios
        # Ej: 'OPEN5 g_gunnthuan ' -> ['open5', 'g_gunnthuan']
        tokens = raw_val.lower().strip().split()

        valid_filters = []

        for token in tokens:
            # Verificar si el token empieza con alguna palabra ignorada
            # (ej. 'open5' empieza con 'open')
            is_ignored = False
            for ignore_kw in ignore_keyword_values:
                if token.startswith(ignore_kw):
                    is_ignored = True
                    break

            if not is_ignored:
                valid_filters.append(token)

        # Lógica de retorno
        if len(valid_filters) == 0:
            #Sitodo fue ignorado (ej: OPEN1 OPEN2), devolver unknown o raw.
            return "unknown"
        elif len(valid_filters) == 1:
            return valid_filters[0]  # El caso ideal
        else:
            # Si hay dos filtros reales (ej: 'U B'), unirlos o tomar el primero
            # Para este pipeline, asumimos el primero como dominante o los unimos
            return "_".join(valid_filters)


def load_and_align_images(file_list, reference_idx=0, extension=1):
    """
    Alinea imágenes y devuelve headers primario y secundario de la referencia.
    """
    ref_file = file_list[reference_idx]
    print(f"⭐ Referencia de alineación: {ref_file}")

    with fits.open(ref_file) as hdul:
        # Guardamos AMBOS headers para reconstruir la estructura después
        ref_primary_header = hdul[0].header.copy()

        if extension < len(hdul):
            ref_data = hdul[extension].data.astype("float32")
            ref_sci_header = hdul[extension].header.copy()
        else:
            print(f"⚠️ Extensión {extension} no hallada. Usando 0.")
            ref_data = hdul[0].data.astype("float32")
            ref_sci_header = hdul[0].header.copy()

    aligned_images = [ref_data]
    success_files = [ref_file]

    for i, file_path in enumerate(file_list):
        if i == reference_idx: continue
        try:
            with fits.open(file_path) as hdul:
                target_ext = extension if extension < len(hdul) else 0
                source_data = hdul[target_ext].data.astype("float32")

            registered_image, _ = aa.register(source_data, ref_data, fill_value=np.nan)
            aligned_images.append(registered_image)
            success_files.append(file_path)
            print(f"✅ Alineado: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"❌ Error en {os.path.basename(file_path)}: {e}")

    if len(aligned_images) < 2:
        return None, None, None, []  # Cambiado return signature

    aligned_stack = np.array(aligned_images)
    # Devolvemos (stack, primary_header, sci_header, lista_archivos)
    return aligned_stack, ref_primary_header, ref_sci_header, success_files

def combine_stack(stack, method='median', sigma=3.0, maxiters=5):
    """Combina el cubo de imágenes."""
    print(f"📚 Combinando {stack.shape[0]} imágenes ({method.upper()})...")
    if method == 'median':
        final_image = np.nanmedian(stack, axis=0)
    elif method == 'mean':
        final_image = np.nanmean(stack, axis=0)
    elif method == 'sigmaclip':
        print(f"✂️ Sigma Clipping (σ={sigma})...")
        clipped_stack = sigma_clip(stack, sigma=sigma, maxiters=maxiters, axis=0, copy=False)
        filled_data = np.where(clipped_stack.mask, np.nan, clipped_stack.data)
        final_image = np.nanmean(filled_data, axis=0)
    else:
        raise ValueError(f"Método desconocido: {method}")
    return final_image

def save_coadd(output_path, data, primary_header, sci_header, input_files, method):
    """
    Guarda la imagen manteniendo la estructura PrimaryHDU + ImageHDU.
    """
    # Actualizar historial en el header científico (donde está la data)
    sci_header['HISTORY'] = f"Stacked {len(input_files)} images using MAS-STACK method {method}"
    sci_header['NCOMBINE'] = (len(input_files), "Number of images combined")
    sci_header['EXTNAME'] = 'COADD'

    # Crear estructura FITS: [0] Primary (vacío o metadata) + [1] Image (Data)
    hdu0 = fits.PrimaryHDU(header=primary_header)
    hdu1 = fits.ImageHDU(data=data, header=sci_header)

    hdul = fits.HDUList([hdu0, hdu1])
    hdul.writeto(output_path, overwrite=True)
    print(f"💾 Imagen Coadd guardada: {output_path} (HDU structure preserved)")


def create_rgb_product(rgb_dict, output_base, extension=1, stretch=0.5, Q=10, do_scaling=True):
    """
    Alinea canales, normaliza intensidades y genera un PNG RGB estilo Lupton.

    Args:
        stretch (float): Linear stretch de Lupton (intensidad).
        Q (float): Asinh softening parameter (suavidad de zonas brillantes).
        do_scaling (bool): Si True, normaliza cada canal para equilibrar colores.
    """
    # 1. Definir Referencia (Verde)
    path_g = rgb_dict['G'][1]
    path_b = rgb_dict['B'][1]
    path_r = rgb_dict['R'][1]

    print(f"🔄 Generando RGB (Stretch={stretch}, Q={Q})...")

    # --- Función auxiliar de carga y pre-procesamiento ---
    def load_and_prep(path, ref_data=None):
        with fits.open(path) as hdul:
            ext = extension if extension < len(hdul) else 0
            data = hdul[ext].data.astype("float32")
            header = hdul[ext].header.copy()
            prim_header = hdul[0].header.copy()

        # Alinear si hay referencia
        if ref_data is not None:
            try:
                # fill_value=np.nan para no afectar estadística del fondo
                data, _ = aa.register(data, ref_data, fill_value=np.nan)
            except Exception as e:
                print(f"❌ Error alineando {os.path.basename(path)}: {e}")
                return None, None, None

        # --- CORRECCIÓN DE FONDO Y ESCALA (La clave del éxito) ---
        # 1. Calcular estadísticas robustas (ignorando bordes NaN)
        valid_pixels = data[~np.isnan(data)]
        if len(valid_pixels) == 0: return data, header, prim_header

        # Usamos sigma clipping para hallar el fondo real
        mean_bg, median_bg, std_bg = sigma_clipped_stats(valid_pixels, sigma=3.0)

        # 2. Restar el fondo (Pedestal a 0)
        data_sub = data - median_bg

        # 3. Reemplazar NaNs por 0 para Lupton
        data_sub = np.nan_to_num(data_sub, nan=0.0)

        # 4. Normalización (Scaling)
        # Si un filtro tiene mucha más señal, lo dividimos por su desviación estándar o ZScale
        if do_scaling:
            # Opción A: Usar ZScale (mejor contraste visual)
            z = ZScaleInterval()
            vmin, vmax = z.get_limits(data_sub)
            scale_factor = vmax if vmax > 0 else 1.0

            # Opción B: Usar STDEV (más físico, pero arriesgado si hay mucha diferencia)
            # scale_factor = std_bg

            data_norm = data_sub / scale_factor
            print(f"   ⚖️ Canal {os.path.basename(path)}: Background={median_bg:.1f}, ScaleFactor={scale_factor:.1f}")
            return data_norm, header, prim_header
        else:
            print(f"   ⚖️ Canal {os.path.basename(path)}: Background={median_bg:.1f} (Sin rescalado)")
            return data_sub, header, prim_header

    # --- PROCESO ---

    # 1. Cargar G (Base)
    print("   🟢 Procesando Canal Verde (Referencia)...")
    img_g, hdr_sci_g, hdr_prim_g = load_and_prep(path_g, ref_data=None)

    # 2. Cargar B (Alineado a G)
    print("   🔵 Procesando Canal Azul...")
    img_b, _, _ = load_and_prep(path_b,
                                ref_data=img_g)  # Pasamos img_g SIN normalizar idealmente, pero astroalign aguanta

    # 3. Cargar R (Alineado a G)
    print("   🔴 Procesando Canal Rojo...")
    img_r, _, _ = load_and_prep(path_r,
                                ref_data=img_g)  # Nota: Usar el img_g original para alinear sería más purista, pero esto funciona.

    if img_g is None or img_b is None or img_r is None:
        print("❌ Fallo en la preparación de canales.")
        return

    # --- GUARDAR FITS ALINEADOS (Opcional, guardamos los datos PRE-Lupton pero POST-alineación) ---
    # Nota: Estos fits tendrán el fondo restado y escalado. Si quieres los originales alineados,
    # tendrías que separar la lógica. Para visualización rápida, esto sirve.

    # --- GENERAR RGB ---
    try:
        # Lupton espera valores positivos y relaciones coherentes.
        # Como ya normalizamos con ZScale, los valores están aprox entre 0 y 1 (o un poco más en estrellas).
        # Un stretch de 0.5 suele funcionar bien con datos normalizados.

        rgb_image = make_lupton_rgb(img_r, img_g, img_b, minimum=0, stretch=stretch, Q=Q)

        plt.figure(figsize=(12, 12))
        plt.imshow(rgb_image, origin='lower')
        plt.axis('off')

        # Texto informativo en la imagen
        info_txt = (f"R: {rgb_dict['R'][2]}\nG: {rgb_dict['G'][2]}\nB: {rgb_dict['B'][2]}\n"
                    f"S={stretch}, Q={Q}")
        plt.text(0.02, 0.02, info_txt, transform=plt.gca().transAxes, color='white', fontsize=10, alpha=0.7)

        png_name = f"{output_base}_rgb_s{stretch}_Q{Q}.png"
        plt.savefig(png_name, bbox_inches='tight', dpi=150)
        print(f"🖼️ RGB guardado: {png_name}")

    except Exception as e:
        print(f"⚠️ Error generando Lupton RGB: {e}")
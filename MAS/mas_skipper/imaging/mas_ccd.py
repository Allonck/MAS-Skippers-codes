#!/usr/bin/env python3

import argparse
import os
import glob
from astropy.io import fits
from ..core.core import roi_shifting, combine_science_images
from .redmas import overscan_correction_combined, create_master_bias, bias_subtraction, create_master_dark, \
    dark_subtraction, create_master_flat_normalized, flat_fielding, cosmic_ray_correction, estimate_readnoise, add_wcs

CAMERA_CONFIGS = {
    'v5+hh2d7': {
        'gain': [52.0] * 16,  # Ejemplo: 4 extensiones, ganancia en e-/ADU
        'satlevel': [2860000] * 16  # Ejemplo: nivel de saturación en ADU. Asumir 55ke-.
    },
    'v5+hh3': {
        'gain': 50.0,  # [1.2, 1.2, 1.1, 1.1],
        'satlevel': 1400000  # [65000, 65000, 65000, 65000] #Asumir 28 ke-.
    },
    'seq+hh2d7': {
        'gain': 48.0,  # Valor único si son similares
        'satlevel': 2640000  # Asumir 55 ke-.
    },
    'seq+hh3': {
        'gain': 50.0,
        'satlevel': 1400000  # Asumir 28 ke-.
    },
    'v5+ulensing': {
        'gain': 52.0,
        'satlevel': 260000  # Asumir 5ke-.
    },
    'other': {
        'gain': 50.0,
        'satlevel': 1400000  # Asumir 28ke-.
    }
}

def detect_camera_config(file_path):
    """
    Detecta la configuración de cámara basada en keywords del header primario.

    Args:
        file_path (str): Ruta al archivo FITS.

    Returns:
        str: Configuración detectada (e.g., 'v5+hh2d7').
    """
    with fits.open(file_path) as hdul:
        hdr = hdul[0].header

        # Verificar VR clock (VR_H y VR_L distintas)
        vr_h = hdr.get('VR_H', None)
        vr_l = hdr.get('VR_L', None)
        if vr_h is not None and vr_l is not None and vr_h != vr_l:
            print(f"VR clock detectado: VR_H={vr_h}, VR_L={vr_l}")
        else:
            print("No se detectó VR clocking.")

        # Verificar H1A_H para hh2d7 o hh3
        h1a_h = hdr.get('H1A_H', None)
        if h1a_h is not None:
            h1a_h_str = str(h1a_h)
            if h1a_h_str == '2.7000':
                hh_type = 'hh2d7'
            elif h1a_h_str == '3.0000':
                hh_type = 'hh3'
            elif h1a_h_str == '-2.0000':
                return 'v5+ulensing'
            else:
                return 'other+other'

        # Verificar HOVER, INTPED, INTSIG para v5 o seq
        hover = hdr.get('HOVER', None)
        intped = hdr.get('INTPED', None)
        intsig = hdr.get('INTSIG', None)
        if hover is not None and intped is not None and intsig is not None:
            hover_str = str(hover)
            intped_str = str(intped)
            intsig_str = str(intsig)
            if hover_str == '4.0000' and intped_str == '89.0000' and intsig_str == '82.0000':
                return f'v5+{hh_type}' if 'hh_type' in locals() else 'other+other'
            elif hover_str == '10.0000' and intped_str == '130.0000' and intsig_str == '90.0000':
                return f'seq+{hh_type}' if 'hh_type' in locals() else 'other+other'
            else:
                return 'other'
        else:
            return 'other'

def main():
    parser = argparse.ArgumentParser(description="MASSKIP v0.4.0 - No warranty of results.")

    parser.add_argument("--raw", type=str, default=".", help="Carpeta con los FITS raw.")
    parser.add_argument("--output", type=str, default="./reduced", help="Carpeta de salida.")
    parser.add_argument("--full-reduction", action="store_true",
                        help="Ejecutar reducción completa (overscan, bias, dark, flat).")
    parser.add_argument("--reduction", action="store_true", help="Ejecutar reducción sin darks (overscan, bias, flat).")
    parser.add_argument("--bias-pattern", type=str, default="frame_bias*.fits", help="Patrón de archivos bias.")
    parser.add_argument("--dark-pattern", type=str, default="frame_dark*.fits", help="Patrón de archivos dark.")
    parser.add_argument("--flat-pattern", type=str, default="frame_flat*.fits", help="Patrón de archivos flat.")
    parser.add_argument("--sci-pattern", type=str, default="sci*.fits", help="Patrón de archivos ciencia.")
    parser.add_argument("--make-master-bias", action="store_true", help="Crear master bias.")
    parser.add_argument("--make-master-dark", action="store_true", help="Crear master dark.")
    parser.add_argument("--do-bias-subtraction", action="store_true", help="Aplicar master bias a ciencia.")
    parser.add_argument("--roi-overscan", type=int, nargs=4, default=[575, 600, 10, 1000],
                        help="ROI de overscan: col_start col_end row_start row_end (1st ext)")
    parser.add_argument("--method", choices=["mean", "poly"], default="mean", help="Método de corrección overscan.")
    parser.add_argument("--do-dark-subtraction", action="store_true", help="Aplicar master dark a ciencia.")
    parser.add_argument("--make-master-flat", action="store_true", help="Crear master flat.")
    parser.add_argument("--do-flat-fielding", action="store_true", help="Aplicar flat fielding.")
    parser.add_argument("--do-cosmic-ray-correction", action="store_true",
                        help="Aplicar corrección de rayos cósmicos con LACosmic.")
    parser.add_argument("--master-bias-name", default="master_bias.fits", help="Nombre del archivo master bias.")
    parser.add_argument("--master-dark-name", default="master_dark.fits", help="Nombre del archivo master dark.")
    parser.add_argument("--master-flat-name", default="master_flat.fits", help="Nombre del archivo master flat.")
    parser.add_argument("--combined-prefix", type=str, default="comb_",
                        help="Prefijo para los archivos combinados de ciencia (e.g., 'comb_').")
    parser.add_argument("--do-wcs", action="store_true", help="Añadir coordenadas WCS usando astropy.wcs.")
    parser.add_argument("--remove-ext", type=int, nargs='*', default=[],
                        help="Extensiones a excluir en la combinación (e.g., --remove-ext 14 15 16).")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Determinar pasos a ejecutar
    make_master_bias = args.make_master_bias or args.full_reduction or args.reduction or args.do_bias_subtraction or args.make_master_dark or args.do_dark_subtraction or args.make_master_flat or args.do_flat_fielding
    make_master_dark = args.make_master_dark or args.full_reduction or args.do_dark_subtraction
    do_bias_subtraction = args.do_bias_subtraction or args.full_reduction or args.reduction or args.do_dark_subtraction or args.do_flat_fielding
    do_dark_subtraction = args.make_master_dark or args.do_dark_subtraction or args.full_reduction
    make_master_flat = args.make_master_flat or args.full_reduction or args.reduction or args.do_flat_fielding
    do_flat_fielding = args.do_flat_fielding or args.full_reduction or args.reduction
    do_cosmic_ray_correction = args.do_cosmic_ray_correction or args.full_reduction
    do_wcs = args.do_wcs or args.full_reduction

    # Generar vector de ROIs para overscan
    roi_vector = roi_shifting(args.roi_overscan)

    # Detectar configuración de cámara automáticamente ---------------------------------
    # Usar la primera imagen científica o bias para detectar
    first_file = sorted(glob.glob(os.path.join(args.raw, args.sci_pattern))) or sorted(
        glob.glob(os.path.join(args.raw, args.bias_pattern)))
    print(f"Usando archivo {first_file} para encontrar configuración de cámara.")
    if not first_file:
        print("⚠️ No se encontraron archivos para detectar la configuración de cámara. Usando 'other'.")
        camera_config = 'other'
    else:
        camera_config = detect_camera_config(first_file[0])
        print(f"✅ Configuración de cámara detectada: {camera_config}")

    # Obtener parámetros de la configuración
    config = CAMERA_CONFIGS.get(camera_config, {'gain': [50] * 16, 'satlevel': [1400000] * 16})
    gain_vector = config.get('gain', [50] * 16)
    satlevel_vector = config.get('satlevel', [1400000] * 16)

    # Obtener NSAMP (HDU[1]) y EXPTIME (HDU[0]) de referencia desde el primer archivo científico
    sci_files = sorted(glob.glob(os.path.join(args.raw, args.sci_pattern)))
    reference_nsamp = None
    reference_exptime = None
    if sci_files:
        with fits.open(sci_files[0]) as hdul:
            reference_nsamp = hdul[1].header.get('NSAMP', None) if len(hdul) > 1 else None
            reference_exptime = hdul[0].header.get('EXPTIME', None)
        print(f"📋 NSAMP de referencia (ciencia, HDU[1]): {reference_nsamp}")
        print(f"📋 EXPTIME de referencia (ciencia, HDU[0]): {reference_exptime}")

    # Calcular readnoise una vez al inicio
    sci_files = sorted(glob.glob(os.path.join(args.raw, args.sci_pattern)))
    read_noise = None
    if sci_files and (args.do_cosmic_ray_correction or args.full_reduction or args.reduction or do_bias_subtraction):
        print(f"📊 Calculando readout noise usando archivos {sci_files}...")
        roi_vector = roi_shifting(args.roi_overscan)
        read_noise = estimate_readnoise(sci_files, roi_vector=roi_vector)

    # 1. Master bias
    mbias_path = os.path.join(args.output, args.master_bias_name)
    if make_master_bias:
        bias_files = sorted(glob.glob(os.path.join(args.raw, args.bias_pattern)))
        if not bias_files:
            print("⚠️ No se encontraron archivos bias con el patrón especificado. Saltando creación de master bias.")
            make_master_bias = False
            do_bias_subtraction = False
        else:
            consistent_bias_files = []
            for f in bias_files:
                with fits.open(f) as hdul:
                    bias_nsamp = hdul[1].header.get('NSAMP', None) if len(hdul) > 1 else None
                    if reference_nsamp is not None and bias_nsamp != reference_nsamp:
                        print(f"⚠️ Archivo bias {f} tiene NSAMP={bias_nsamp} (HDU[1]), no coincide con NSAMP={reference_nsamp}. Omitiendo.")
                        continue
                    consistent_bias_files.append(f)
            if not consistent_bias_files:
                print("⚠️ No se encontraron archivos bias con NSAMP consistente. Saltando creación de master bias.")
                make_master_bias = False
                do_bias_subtraction = False
            else:
                overscan_bias_files = []
                print(f"📥 Procesando {len(consistent_bias_files)} archivos bias con NSAMP consistente...")
                for f in consistent_bias_files:
                    out_file = os.path.join(args.output, f"o_{os.path.basename(f)}")
                    overscan_correction_combined(f, out_file, roi_vector, method=args.method)
                    overscan_bias_files.append(out_file)
                create_master_bias(overscan_bias_files, mbias_path)
                print(f"✅ Master bias creado: {mbias_path}")

    # 2. Master dark
    mdark_path = os.path.join(args.output, args.master_dark_name)
    if make_master_dark:
        dark_files = sorted(glob.glob(os.path.join(args.raw, args.dark_pattern)))
        if not dark_files:
            print("⚠️ No se encontraron archivos dark con el patrón especificado. Saltando creación de master dark.")
            make_master_dark = False
            do_dark_subtraction = False
        elif not os.path.exists(mbias_path):
            print("⚠️ Master bias no encontrado. Necesario para corregir darks. Saltando creación de master dark.")
            make_master_dark = False
            do_dark_subtraction = False
        else:
            consistent_dark_files = []
            for f in dark_files:
                with fits.open(f) as hdul:
                    dark_nsamp = hdul[1].header.get('NSAMP', None) if len(hdul) > 1 else None
                    dark_exptime = hdul[0].header.get('EXPTIME', None)
                    if reference_nsamp is not None and dark_nsamp != reference_nsamp:
                        print(f"⚠️ Archivo dark {f} tiene NSAMP={dark_nsamp} (HDU[1]), no coincide con NSAMP={reference_nsamp}. Omitiendo.")
                        continue
                    if reference_exptime is not None and dark_exptime != reference_exptime:
                        print(f"⚠️ Archivo dark {f} tiene EXPTIME={dark_exptime} (HDU[0]), no coincide con EXPTIME={reference_exptime}. Omitiendo.")
                        continue
                    consistent_dark_files.append(f)
            if not consistent_dark_files:
                print("⚠️ No se encontraron archivos dark con NSAMP y EXPTIME consistentes. Saltando creación de master dark.")
                make_master_dark = False
                do_dark_subtraction = False
            else:
                overscan_dark_files = []
                bias_corrected_dark_files = []
                print(f"📥 Procesando {len(consistent_dark_files)} archivos dark con NSAMP y EXPTIME consistentes...")
                for f in consistent_dark_files:
                    overscan_file = os.path.join(args.output, f"o_{os.path.basename(f)}")
                    bias_corrected_file = os.path.join(args.output, f"bo_{os.path.basename(f)}")
                    overscan_correction_combined(f, overscan_file, roi_vector, method=args.method)
                    bias_subtraction(overscan_file, mbias_path, bias_corrected_file)
                    overscan_dark_files.append(overscan_file)
                    bias_corrected_dark_files.append(bias_corrected_file)
                create_master_dark(bias_corrected_dark_files, mdark_path)
                print(f"✅ Master dark creado: {mdark_path}")

    # 3. Sustracción de bias y dark en ciencia
    if do_bias_subtraction or do_dark_subtraction:
        sci_files = sorted(glob.glob(os.path.join(args.raw, args.sci_pattern)))
        mbias_path = os.path.join(args.output, args.master_bias_name)
        mdark_path = os.path.join(args.output, args.master_dark_name)

        if not sci_files:
            print("⚠️ No se encontraron archivos de ciencia con el patrón especificado.")
            return
        if do_bias_subtraction and not os.path.exists(mbias_path):
            print("⚠️ Master bias no encontrado. Necesario para sustracción de bias.")
            return
        if do_dark_subtraction and not os.path.exists(mdark_path):
            print("⚠️ Master dark no encontrado. Necesario para sustracción de dark.")
            return

        dark_files = sorted(glob.glob(os.path.join(args.raw, args.dark_pattern)))
        if not dark_files:
            do_dark_subtraction = False
            print("⚠️ No se encontraron archivos dark. Saltando sustracción de dark.")

        print(f"🔬 Procesando ciencia: {len(sci_files)} archivos")
        for f in sci_files:
            overscan_file = os.path.join(args.output, f"o_{os.path.basename(f)}")
            bias_corrected_file = os.path.join(args.output, f"bo_{os.path.basename(f)}")
            dark_corrected_file = os.path.join(args.output, f"dbo_{os.path.basename(f)}")

            overscan_correction_combined(f, overscan_file, roi_vector, method=args.method)
            if do_bias_subtraction:
                bias_subtraction(overscan_file, mbias_path, bias_corrected_file)
            if do_dark_subtraction:
                input_file = bias_corrected_file if do_bias_subtraction else overscan_file
                dark_subtraction(input_file, mdark_path, dark_corrected_file)

    # 4. Master flat por filtro
    if make_master_flat:
        flat_files = sorted(glob.glob(os.path.join(args.raw, args.flat_pattern)))
        if not flat_files:
            print("⚠️ No se encontraron archivos flat con el patrón especificado. Saltando creación de master flat.")
            make_master_flat = False
            do_flat_fielding = False
        elif not os.path.exists(mbias_path):
            print("⚠️ Master bias no encontrado. Necesario para corregir flats. Saltando creación de master flat.")
            make_master_flat = False
            do_flat_fielding = False
        elif do_dark_subtraction and not os.path.exists(mdark_path):
            print("⚠️ Master dark no encontrado. Necesario para corregir flats con dark. Saltando creación de master flat.")
            make_master_flat = False
            do_flat_fielding = False
        else:
            print(f"📥 Procesando {len(flat_files)} archivos flat...")
            flat_groups = {}
            for f in flat_files:
                with fits.open(f) as hdul:
                    filters = hdul[0].header.get('FILTERS', 'unknown').strip()
                    filter_key = filters.split()[-1] if filters != 'unknown' else 'unknown'
                    if filter_key not in flat_groups:
                        flat_groups[filter_key] = []
                    flat_groups[filter_key].append(f)
            for filter_key, group_files in flat_groups.items():
                overscan_group_files = []
                for f in group_files:
                    overscan_file = os.path.join(args.output, f"o_{os.path.basename(f)}")
                    overscan_correction_combined(f, overscan_file, roi_vector, method=args.method)
                    overscan_group_files.append(overscan_file)
                mflat_path = os.path.join(args.output, f"{args.master_flat_name.split('.fits')[0]}_{filter_key}.fits")
                create_master_flat_normalized(overscan_group_files, mbias_path, mflat_path,
                                             use_dark=do_dark_subtraction, master_dark_path=mdark_path)
                print(f"✅ Master flat creado para filtro {filter_key}: {mflat_path}")

    # 5. Verificación de master flats por filtro
    if do_flat_fielding:
        sci_files = sorted(glob.glob(os.path.join(args.raw, args.sci_pattern)))
        if not sci_files:
            print("⚠️ No se encontraron archivos de ciencia con el patrón especificado.")
            return
        dark_files = sorted(glob.glob(os.path.join(args.raw, args.dark_pattern)))
        use_dark = do_dark_subtraction and dark_files and os.path.exists(mdark_path)
        print(f"🌈 Verificando master flats para {len(sci_files)} archivos de ciencia...")
        for f in sci_files:
            with fits.open(f) as hdul:
                filters = hdul[0].header.get('FILTERS', 'unknown').strip()
                filter_key = filters.split()[-1] if filters != 'unknown' else 'unknown'
            mflat_path = os.path.join(args.output, f"{args.master_flat_name.split('.fits')[0]}_{filter_key}.fits")
            if not os.path.exists(mflat_path):
                print(f"⚠️ Master flat para filtro {filter_key} no encontrado. Saltando flat-fielding para {f}.")
                do_flat_fielding = False
                break

    # 6. Procesamiento de imágenes de ciencia
    do_bias_subtraction = args.do_bias_subtraction or args.reduction or args.full_reduction
    do_dark_subtraction = args.do_dark_subtraction or args.full_reduction
    do_flat_fielding = args.do_flat_fielding or args.full_reduction or args.reduction
    do_cosmic_ray_correction = args.do_cosmic_ray_correction or args.full_reduction
    do_wcs = args.do_wcs or args.full_reduction or args.reduction
    use_dark = do_dark_subtraction and dark_files and os.path.exists(mdark_path)
    if sci_files:
        print(f"🔬 Procesando {len(sci_files)} imágenes de ciencia...")
        for i, f in enumerate(sci_files, 1):
            print(f"[{i}/{len(sci_files)}] {f}")
            prefix = ""
            overscan_file = os.path.join(args.output, f"o_{os.path.basename(f)}")
            if not os.path.exists(overscan_file):
                overscan_correction_combined(f, overscan_file, roi_vector, method=args.method)
            prefix = "o"
            if do_bias_subtraction:
                bias_file = os.path.join(args.output, f"bo_{os.path.basename(f)}")
                if not os.path.exists(bias_file):
                    bias_subtraction(overscan_file, mbias_path, bias_file)
                prefix = "bo"
            if use_dark:
                dark_file = os.path.join(args.output, f"dbo_{os.path.basename(f)}")
                if not os.path.exists(dark_file):
                    dark_subtraction(bias_file if do_bias_subtraction else overscan_file, mdark_path, dark_file)
                prefix = "dbo"
            if do_flat_fielding:
                with fits.open(f) as hdul:
                    filters = hdul[0].header.get('FILTERS', 'unknown').strip()
                    filter_key = filters.split()[-1] if filters != 'unknown' else 'unknown'
                mflat_path = os.path.join(args.output, f"{args.master_flat_name.split('.fits')[0]}_{filter_key}.fits")
                if not os.path.exists(mflat_path):
                    print(f"⚠️ Master flat para filtro {filter_key} no encontrado. Saltando flat-fielding para {f}.")
                    continue
                flat_file = os.path.join(args.output, f"f{prefix}_{os.path.basename(f)}")
                if not os.path.exists(flat_file):
                    flat_fielding(
                        dark_file if use_dark else (bias_file if do_bias_subtraction else overscan_file),
                        mflat_path, flat_file
                    )
                prefix = f"f{prefix}"
                if do_wcs:
                    add_wcs(flat_file, flat_file)
            if do_cosmic_ray_correction:
                cosmic_file = os.path.join(args.output, f"c{prefix}_{os.path.basename(f)}")
                if not os.path.exists(cosmic_file):
                    cosmic_ray_correction(
                        flat_file,
                        cosmic_file, sigclip=5.0, sigfrac=0.3, objlim=6.0,
                        readnoise_vector=read_noise, gain_vector=gain_vector, satlevel_vector=satlevel_vector
                    )
                prefix = f"c{prefix}"

    # 7. Combinar imágenes de ciencia
    if sci_files:
        dark_files = sorted(glob.glob(os.path.join(args.raw, args.dark_pattern)))
        use_dark = do_dark_subtraction and dark_files and os.path.exists(mdark_path)
        prefix = ""
        if do_cosmic_ray_correction:
            prefix = "c"
        if do_flat_fielding:
            prefix += "f"
        if use_dark:
            prefix += "dbo"
        elif do_bias_subtraction:
            prefix += "bo"
        else:
            prefix += "o"
        input_pattern = f"{prefix}_*.fits"
        corrected_files = sorted(glob.glob(os.path.join(args.output, input_pattern)))
        if not corrected_files:
            print(
                f"⚠️ No se encontraron archivos corregidos con el patrón {input_pattern}. Intentando con archivos menos procesados.")
            alternative_patterns = [
                f"c{'f' if do_flat_fielding else ''}{'dbo' if use_dark else 'bo'}_*.fits" if do_cosmic_ray_correction else None,
                f"{'f' if do_flat_fielding else ''}{'dbo' if use_dark else 'bo'}_*.fits",
                f"{'f' if do_flat_fielding else ''}o_*.fits" if do_flat_fielding else None,
                "cdbo_*.fits" if do_cosmic_ray_correction and use_dark else None,
                "dbo_*.fits" if use_dark else None,
                "cbo_*.fits" if do_cosmic_ray_correction and do_bias_subtraction else None,
                "bo_*.fits" if do_bias_subtraction else None,
                "co_*.fits" if do_cosmic_ray_correction else None,
                "o_*.fits"
            ]
            for pattern in [p for p in alternative_patterns if p]:
                corrected_files = sorted(glob.glob(os.path.join(args.output, pattern)))
                if corrected_files:
                    input_pattern = pattern
                    break
            if not corrected_files:
                print("⚠️ No se encontraron archivos corregidos para combinar. Generando archivos overscan.")
                for f in sci_files:
                    overscan_file = os.path.join(args.output, f"o_{os.path.basename(f)}")
                    if not os.path.exists(overscan_file):
                        overscan_correction_combined(f, overscan_file, roi_vector, method=args.method)
                corrected_files = sorted(glob.glob(os.path.join(args.output, "o_*.fits")))
                input_pattern = "o_*.fits"
        if corrected_files:
            print(
                f"🔗 Combinando los 16 canales de {len(corrected_files)} imágenes científicas con patrón {input_pattern}...")
            combined_count = 0
            processed_files = set()
            for corrected_file in corrected_files:
                # Encontrar el archivo científico original a partir del nombre del archivo corregido
                base_name = os.path.basename(corrected_file).replace(f"{prefix}_", "")
                original_file = os.path.join(args.raw, base_name)
                # Extraer filtro, tiempo de exposición y NSAMP del archivo original
                filter_key = "unknown"
                exptime = "unknown"
                nsamp = "unknown"
                if os.path.exists(original_file):
                    with fits.open(original_file) as hdul:
                        filters = hdul[0].header.get('FILTERS', 'unknown').strip()
                        filter_key = filters.split()[-1] if filters != 'unknown' else 'unknown'
                        exptime = int(hdul[0].header.get('EXPTIME', 0))
                        nsamp = hdul[1].header.get('NSAMP', 'unknown') if len(hdul) > 1 else 'unknown'
                else:
                    print(f"⚠️ Archivo original {original_file} no encontrado. Usando filtro 'unknown', EXPTIME 'unknown' y NSAMP 'unknown'.")
                # Construir el nombre del archivo combinado
                combined_output = os.path.join(args.output, f"{args.combined_prefix}{prefix}_{base_name.split('.fits')[0]}_{filter_key}_{exptime}s_{nsamp}.fits")
                if corrected_file not in processed_files and not os.path.exists(combined_output):
                    combine_science_images(
                        [corrected_file],  # Solo la imagen actual
                        combined_output,
                        roi_base=[1, 512, 0, 1024],
                        exclude_extensions=args.remove_ext
                    )
                    combined_count += 1
                    processed_files.add(corrected_file)
                    print(f"✅ Imagen combinada guardada: {combined_output}")
            print(f"✅ Total de imágenes combinadas generadas: {combined_count}")

if __name__ == "__main__":
    main()
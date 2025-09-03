#!/usr/bin/env python3

import argparse
import os
import glob
from astropy.io import fits
from mas_skipper import roi_shifting, combine_science_images
from .redmas import overscan_correction_combined, create_master_bias, bias_subtraction, create_master_dark, dark_subtraction, create_master_flat_normalized, flat_fielding

def main():
    parser = argparse.ArgumentParser(description="Reducción MAS Skipper CCD (overscan + bias + dark + flat)")

    parser.add_argument("--raw", type=str, default=".", help="Carpeta con los FITS raw.")
    parser.add_argument("--output", type=str, default="./reduced", help="Carpeta de salida.")
    parser.add_argument("--full-reduction", action="store_true", help="Ejecutar reducción completa (overscan, bias, dark, flat).")
    parser.add_argument("--reduction", action="store_true", help="Ejecutar reducción sin darks (overscan, bias, flat).")
    parser.add_argument("--bias-pattern", type=str, default="frame_bias*.fits", help="Patrón de archivos bias.")
    parser.add_argument("--dark-pattern", type=str, default="frame_dark*.fits", help="Patrón de archivos dark.")
    parser.add_argument("--flat-pattern", type=str, default="frame_flat*.fits", help="Patrón de archivos flat.")
    parser.add_argument("--sci-pattern", type=str, default="sci*.fits", help="Patrón de archivos ciencia.")
    parser.add_argument("--make-master-bias", action="store_true", help="Crear master bias.")
    parser.add_argument("--make-master-dark", action="store_true", help="Crear master dark.")
    parser.add_argument("--do-bias-subtraction", action="store_true", help="Aplicar master bias a ciencia.")
    parser.add_argument("--do-dark-subtraction", action="store_true", help="Aplicar master dark a ciencia.")
    parser.add_argument("--make-master-flat", action="store_true", help="Crear master flat.")
    parser.add_argument("--do-flat-fielding", action="store_true", help="Aplicar flat fielding.")
    parser.add_argument("--method", choices=["mean", "poly"], default="mean", help="Método de corrección overscan.")
    parser.add_argument("--master-bias-name", default="master_bias.fits", help="Nombre del archivo master bias.")
    parser.add_argument("--master-dark-name", default="master_dark.fits", help="Nombre del archivo master dark.")
    parser.add_argument("--master-flat-name", default="master_flat.fits", help="Nombre del archivo master flat.")
    parser.add_argument("--combined-prefix", type=str, default="comb_",
                        help="Prefijo para los archivos combinados de ciencia (e.g., 'comb_').")
    parser.add_argument("--roi-overscan", type=int, nargs=4, default=[575, 600, 10, 1000],
                        help="ROI de overscan: col_start col_end row_start row_end (1st ext)")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Determinar pasos a ejecutar
    make_master_bias = args.make_master_bias or args.full_reduction or args.reduction or args.do_bias_subtraction or args.make_master_dark or args.do_dark_subtraction or args.make_master_flat or args.do_flat_fielding
    make_master_dark = args.make_master_dark or args.full_reduction or args.do_dark_subtraction
    do_bias_subtraction = args.do_bias_subtraction or args.full_reduction or args.reduction or args.do_dark_subtraction or args.do_flat_fielding
    do_dark_subtraction = args.make_master_dark or args.do_dark_subtraction or args.full_reduction
    make_master_flat = args.make_master_flat or args.full_reduction or args.reduction or args.do_flat_fielding
    do_flat_fielding = args.do_flat_fielding or args.full_reduction or args.reduction

    # Generar vector de ROIs para overscan
    roi_vector = roi_shifting(args.roi_overscan)

    # 1. Master bias
    if make_master_bias:
        bias_files = sorted(glob.glob(os.path.join(args.raw, args.bias_pattern)))
        overscan_bias_files = []

        if not bias_files:
            print("⚠️ No se encontraron archivos bias con el patrón especificado.")
            return

        print(f"📥 Procesando {len(bias_files)} archivos bias...")
        for f in bias_files:
            out_file = os.path.join(args.output, f"o_{os.path.basename(f)}")
            overscan_correction_combined(f, out_file, roi_vector, method=args.method)
            overscan_bias_files.append(out_file)

        mbias_path = os.path.join(args.output, args.master_bias_name)
        create_master_bias(overscan_bias_files, mbias_path)
        print(f"✅ Master bias creado: {mbias_path}")

    # 2. Master dark
    if make_master_dark:
        dark_files = sorted(glob.glob(os.path.join(args.raw, args.dark_pattern)))
        overscan_dark_files = []
        bias_corrected_dark_files = []
        mbias_path = os.path.join(args.output, args.master_bias_name)

        if not dark_files:
            print("⚠️ No se encontraron archivos dark con el patrón especificado.")
            return
        if not os.path.exists(mbias_path):
            print("⚠️ Master bias no encontrado. Necesario para corregir darks.")
            return

        print(f"📥 Procesando {len(dark_files)} archivos dark...")
        for f in dark_files:
            overscan_file = os.path.join(args.output, f"o_{os.path.basename(f)}")
            bias_corrected_file = os.path.join(args.output, f"b_{os.path.basename(f)}")
            overscan_correction_combined(f, overscan_file, roi_vector, method=args.method)
            bias_subtraction(overscan_file, mbias_path, bias_corrected_file)
            overscan_dark_files.append(overscan_file)
            bias_corrected_dark_files.append(bias_corrected_file)

        mdark_path = os.path.join(args.output, args.master_dark_name)
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

        # Verificar si hay archivos dark para ajustar do_dark_subtraction
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
        overscan_flat_files = []
        mbias_path = os.path.join(args.output, args.master_bias_name)
        mdark_path = os.path.join(args.output, args.master_dark_name)

        if not flat_files:
            print("⚠️ No se encontraron archivos flat con el patrón especificado.")
            return
        if not os.path.exists(mbias_path):
            print("⚠️ Master bias no encontrado. Necesario para corregir flats.")
            return
        if do_dark_subtraction and not os.path.exists(mdark_path):
            print("⚠️ Master dark no encontrado. Necesario para corregir flats con dark.")
            return

        print(f"📥 Procesando {len(flat_files)} archivos flat...")
        # Agrupar flats por la última palabra de 'FILTERS'
        flat_groups = {}
        for f in flat_files:
            with fits.open(f) as hdul:
                filters = hdul[0].header.get('FILTERS', 'unknown').strip()
                # Usar solo la última palabra de 'FILTERS'
                filter_key = filters.split()[-1] if filters != 'unknown' else 'unknown'
                if filter_key not in flat_groups:
                    flat_groups[filter_key] = []
                flat_groups[filter_key].append(f)

        # Crear master_flat por grupo
        for filter_key, group_files in flat_groups.items():
            overscan_group_files = []
            for f in group_files:
                overscan_file = os.path.join(args.output, f"o_{os.path.basename(f)}")
                overscan_correction_combined(f, overscan_file, roi_vector, method=args.method)
                overscan_group_files.append(overscan_file)

            mflat_path = os.path.join(args.output, f"{args.master_flat_name.split('.fits')[0]}_{filter_key}.fits")
            create_master_flat_normalized(overscan_group_files, mbias_path, mflat_path, use_dark=do_dark_subtraction,
                                          master_dark_path=mdark_path)
            print(f"✅ Master flat creado para filtro {filter_key}: {mflat_path}")

    # 5. Flat fielding en ciencia
    if do_flat_fielding:
        sci_files = sorted(glob.glob(os.path.join(args.raw, args.sci_pattern)))
        mbias_path = os.path.join(args.output, args.master_bias_name)
        mdark_path = os.path.join(args.output, args.master_dark_name)

        if not sci_files:
            print("⚠️ No se encontraron archivos de ciencia con el patrón especificado.")
            return

        # Verificar si hay archivos dark para ajustar el prefijo
        dark_files = sorted(glob.glob(os.path.join(args.raw, args.dark_pattern)))
        use_dark = do_dark_subtraction and dark_files and os.path.exists(mdark_path)

        print(f"🌈 Aplicando master flat a {len(sci_files)} archivos")
        for f in sci_files:
            with fits.open(f) as hdul:
                filters = hdul[0].header.get('FILTERS', 'unknown').strip()
                # Usar solo la última palabra de 'FILTERS'
                filter_key = filters.split()[-1] if filters != 'unknown' else 'unknown'
            mflat_path = os.path.join(args.output, f"{args.master_flat_name.split('.fits')[0]}_{filter_key}.fits")
            if not os.path.exists(mflat_path):
                print(f"⚠️ Master flat para filtro {filter_key} no encontrado. Saltando {f}.")
                continue

            overscan_file = os.path.join(args.output, f"o_{os.path.basename(f)}")
            bias_corrected_file = os.path.join(args.output, f"bo_{os.path.basename(f)}")
            dark_corrected_file = os.path.join(args.output, f"dbo_{os.path.basename(f)}")
            # Determinar prefijo según pasos aplicados
            prefix = "f"
            if use_dark:
                prefix = "fdbo"
            elif do_bias_subtraction:
                prefix = "fbo"
            else:
                prefix = "fo"
            flat_corrected_file = os.path.join(args.output, f"{prefix}_{os.path.basename(f)}")

            overscan_correction_combined(f, overscan_file, roi_vector, method=args.method)
            if do_bias_subtraction:
                bias_subtraction(overscan_file, mbias_path, bias_corrected_file)
            if use_dark:
                input_file = bias_corrected_file if do_bias_subtraction else overscan_file
                dark_subtraction(input_file, mdark_path, dark_corrected_file)
            input_file = dark_corrected_file if use_dark else (
                bias_corrected_file if do_bias_subtraction else overscan_file)
            flat_fielding(input_file, mflat_path, flat_corrected_file)

    # 6. Combinar imágenes de ciencia.
    sci_files = sorted(glob.glob(os.path.join(args.raw, args.sci_pattern)))
    if sci_files:
        # Verificar si hay archivos dark para ajustar el prefijo
        dark_files = sorted(glob.glob(os.path.join(args.raw, args.dark_pattern)))
        use_dark = do_dark_subtraction and dark_files and os.path.exists(
            os.path.join(args.output, args.master_dark_name))

        # Seleccionar los archivos más procesados disponibles
        input_pattern = (
            "fdbo_*.fits" if do_flat_fielding and use_dark else
            "fbo_*.fits" if do_flat_fielding and do_bias_subtraction else
            "fo_*.fits" if do_flat_fielding else
            "dbo_*.fits" if use_dark else
            "bo_*.fits" if do_bias_subtraction else
            "o_*.fits"
        )
        corrected_files = sorted(glob.glob(os.path.join(args.output, input_pattern)))

        if not corrected_files:
            print(f"⚠️ No se encontraron archivos corregidos con el patrón {input_pattern}. No se puede combinar.")
        else:
            print(f"🔗 Combinando {len(corrected_files)} imágenes científicas...")
            for corrected_file in corrected_files:
                combined_output = os.path.join(args.output, f"{args.combined_prefix}{os.path.basename(corrected_file)}")
                combine_science_images([corrected_file], combined_output)
            print(f"✅ Proceso de combinación completado.")

if __name__ == "__main__":
    main()
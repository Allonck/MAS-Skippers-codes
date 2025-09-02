#!/usr/bin/env python3

import argparse
import os
import glob

from mas_skipper import roi_shifting
from .redmas import overscan_correction_combined, create_master_bias, bias_subtraction, create_master_flat, flat_fielding

def main():
    parser = argparse.ArgumentParser(description="Reducción MAS Skipper CCD (overscan + masterbias)")

    parser.add_argument("--raw", type=str, default=".", help="Carpeta con los FITS raw.")
    parser.add_argument("--bias-pattern", type=str, default="frame_bias*.fits", help="Patrón de archivos bias.")
    parser.add_argument("--sci-pattern", type=str, default="sci*.fits", help="Patrón de archivos ciencia.")
    parser.add_argument("--output", type=str, default="./reduced", help="Carpeta de salida.")
    parser.add_argument("--make-master-bias", action="store_true", help="Crear master bias.")
    parser.add_argument("--do-bias-subtraction", action="store_true", help="Aplicar master bias a ciencia.")
    parser.add_argument("--method", choices=["mean", "poly"], default="mean", help="Método de corrección overscan.")
    parser.add_argument("--master-bias-name", default="master_bias.fits", help="Nombre del archivo master bias.")

    parser.add_argument("--make-master-flat", action="store_true", help="Crear master flat.")
    parser.add_argument("--do-flat-fielding", action="store_true", help="Aplicar flat fielding.")
    parser.add_argument("--flat-pattern", type=str, default="frame_flat*.fits", help="Patrón de archivos flat.")
    parser.add_argument("--master-flat-name", default="master_flat.fits", help="Nombre del archivo master flat.")

    # ⚠️ NUEVO: ROI fijo que se aplica a todas las extensiones
    parser.add_argument("--roi-overscan", type=int, nargs=4, default=[575, 600, 10, 1000],
                        help="ROI de overscan: col_start col_end row_start row_end (1st ext)")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # CORRECTED: Pass args.roi_overscan directly as it's already a list of 4 integers
    roi_vector = roi_shifting(args.roi_overscan)
    # 1. Master bias
    if args.make_master_bias:
        bias_files = sorted(glob.glob(os.path.join(args.raw, args.bias_pattern)))
        overscan_bias_files = []

        print(f"📥 Procesando {len(bias_files)} archivos bias...")

        for f in bias_files:
            out_file = os.path.join(args.output, f"overscan_{os.path.basename(f)}")
            overscan_correction_combined(f, out_file, roi_vector, method=args.method)
            overscan_bias_files.append(out_file)

        mbias_path = os.path.join(args.output, args.master_bias_name)
        create_master_bias(overscan_bias_files, mbias_path)

    # 2. Substracción de bias en ciencia
    if args.do_bias_subtraction:
        sci_files = sorted(glob.glob(os.path.join(args.raw, args.sci_pattern)))
        mbias_path = os.path.join(args.output, args.master_bias_name)

        print(f"🔬 Procesando ciencia: {len(sci_files)} archivos")

        for f in sci_files:
            overscan_file = os.path.join(args.output, f"overscan_{os.path.basename(f)}")
            corrected_file = os.path.join(args.output, f"biascorr_{os.path.basename(f)}")

            overscan_correction_combined(f, overscan_file, roi_vector, method=args.method)
            bias_subtraction(overscan_file, mbias_path, corrected_file)

    # 3. Master flat
    if args.make_master_flat:
        flat_files = sorted(glob.glob(os.path.join(args.raw, args.flat_pattern)))
        mbias_path = os.path.join(args.output, args.master_bias_name)
        mflat_path = os.path.join(args.output, args.master_flat_name)

        create_master_flat(flat_files, mbias_path, mflat_path)

    # 4. Flat fielding en ciencia

    if args.do_flat_fielding:
        corrected_files = sorted(glob.glob(os.path.join(args.output, "biascorr_*.fits")))
        mflat_path = os.path.join(args.output, args.master_flat_name)

        print(f" Aplicando master flat a {len(corrected_files)} archivos")

        for f in corrected_files:
            output_flat = os.path.join(args.output, f"flatcorr_{os.path.basename(f)}")
            flat_fielding(f, mflat_path, output_flat)

#X: Falta añadir la correción por darks.

if __name__ == "__main__":
    main()
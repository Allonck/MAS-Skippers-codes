#!/usr/bin/env python3
import os
import glob
from astropy.io import fits
import argparse
from .photmas import perform_photometry, visualize_photometry

import importlib.metadata
__version__ = importlib.metadata.version('mas_skipper_pipeline')  # Nombre de tu paquete en pyproject.toml

def main():
    parser = argparse.ArgumentParser(description=f"MASSKIP Photometry Module v{__version__} - Detect sources and perform aperture photometry.")

    parser.add_argument("--input", type=str, required=True, help="Input FITS file or pattern (e.g., 'comb_cfbo_*.fits').")
    parser.add_argument("--output", type=str, default="phot_cat.fits", help="Output catalog file (FITS table).")
    parser.add_argument("--format", type=str, choices=['fits', 'ascii'], default='fits',
                        help="Output format (default: fits).")
    parser.add_argument("--aperture-radius", type=float, default=15.0, help="Aperture radius in pixels (default: 15).")
    parser.add_argument("--threshold", type=float, default=5.0, help="Detection threshold (x background std; default: 5.0).")
    parser.add_argument("--fwhm", type=float, default=3.0, help="FWHM in pixels for detection (default: 3.0).")
    parser.add_argument("--box-size", type=int, nargs=2, default=[50, 50], help="Background box size (default: 50,50).")
    parser.add_argument("--do-visualize", action="store_true", help="Generate visualization plot.")
    parser.add_argument("--zeropoint", type=float, default=0.0, help="Zeropoint for magnitudes (default: 0.0).")

    args = parser.parse_args()

    # Process input files
    files = sorted(glob.glob(args.input)) if '*' in args.input else [args.input]
    if not files:
        print("⚠️ No files found matching pattern.")
        return

    for file_path in files:
        print(f"Procesando: {file_path}")
        all_tables = perform_photometry(file_path, aperture_radius=args.aperture_radius, threshold=args.threshold,
                                        fwhm=args.fwhm, box_size=args.box_size, zeropoint=args.zeropoint)

        if not all_tables:
            print("No HDUs with data; skipping.")
            continue

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        for i, hdu_table in enumerate(all_tables, 1):
            out_name = f"phot_cat_{base_name}_ext{i}.{args.format}"
            hdu_table.write(out_name, format='ascii.basic' if args.format == 'ascii' else args.format, overwrite=True)
            num_sources = len(hdu_table)
            print(f"HDU {i}: Catálogo guardado: {out_name} (detectadas {num_sources} fuentes)")

        if args.do_visualize:
            # Visualize the first HDU or brightest
            visualize_photometry(file_path, all_tables[0] if all_tables else Table(), zoom_size=100, aperture_radius=args.aperture_radius)

if __name__ == "__main__":
    main()
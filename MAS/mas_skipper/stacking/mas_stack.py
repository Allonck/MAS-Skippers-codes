#!/usr/bin/env python3
import argparse
import glob
import os
import sys
import importlib.metadata

from .stackmas import load_and_align_images, combine_stack, save_coadd

# Intentar obtener versión, fallback si no está instalado
try:
    __version__ = importlib.metadata.version('mas_skipper_pipeline')
except importlib.metadata.PackageNotFoundError:
    __version__ = "dev"

def main():
    parser = argparse.ArgumentParser(
        description=f"MAS-STACK v{__version__} Module - Align and Stack module using Astroalign."
    )

    # Argumentos obligatorios
    parser.add_argument("--input", type=str, required=True,
                        help="Patrón de entrada (ej. 'reduced/comb_*.fits'). Usa comillas si usas wildcards (*).")

    # Argumentos opcionales
    parser.add_argument("--output", type=str, default="coadd.fits",
                        help="Nombre del archivo de salida (default: coadd.fits).")

    parser.add_argument("--method", type=str, choices=['median', 'mean', 'sigmaclip'], default='median',
                        help="Método de combinación. 'median' es robusto y rápido. 'sigmaclip' es mejor para limpiar trazas de satélites pero lento.")

    parser.add_argument("--ref-index", type=int, default=0,
                        help="Índice de la imagen a usar como referencia de alineación (0-based). Default: 0 (la primera).")

    parser.add_argument("--ext", type=int, default=1,
                        help="Número de extensión FITS a leer. Para archivos comb_*.fits de mas-ccd es 1.")

    # Parámetros avanzados para sigma clipping
    parser.add_argument("--sigma", type=float, default=3.0,
                        help="Umbral sigma para rejection (solo si method='sigmaclip'). Default: 3.0.")
    parser.add_argument("--iters", type=int, default=5,
                        help="Número de iteraciones para clipping (solo si method='sigmaclip'). Default: 5.")

    args = parser.parse_args()

    # 1. Expandir lista de archivos
    # Soporta tanto "carpeta/*.fits" como una lista directa si el shell la expande
    if '*' in args.input:
        files = sorted(glob.glob(args.input))
    else:
        files = [args.input]  # Caso de un solo archivo (inútil para stack, pero para evitar errores)
        # O si el shell ya expandió los argumentos, argparse solo ve el primero.
        # Nota: Para wildcards en terminal, normalmente sys.argv ya trae la lista.
        # Pero argparse con type=str toma solo uno si no usas nargs='+'.
        # Ajuste para robustez:

    # Si el usuario ejecutó: python mas_stack.py --input *.fits (el shell expande antes)
    # argparse fallará si no usamos nargs='+' o el usuario debe poner comillas.
    # Para simplificar, asumimos que el usuario pone comillas: --input "dir/*.fits"

    if not files:
        print(f"❌ No se encontraron archivos con el patrón: {args.input}")
        sys.exit(1)

    if len(files) < 2:
        print(f"⚠️ Se encontró solo {len(files)} archivo. Se necesitan al menos 2 para apilar.")
        sys.exit(1)

    print(f"🚀 Iniciando MAS-STACK con {len(files)} imágenes.")
    print(f"   Método: {args.method.upper()}")
    if args.method == 'sigmaclip':
        print(f"   Configuración Clip: σ={args.sigma}, iters={args.iters}")

    # 2. Cargar y Alinear
    aligned_stack, ref_header, success_files = load_and_align_images(
        files,
        reference_idx=args.ref_index,
        extension=args.ext
    )

    if aligned_stack is None:
        print("❌ Fallo crítico en alineación. Abortando.")
        sys.exit(1)

    print(f"✅ Alineación exitosa de {len(success_files)}/{len(files)} imágenes.")

    # 3. Combinar
    final_image = combine_stack(
        aligned_stack,
        method=args.method,
        sigma=args.sigma,
        maxiters=args.iters
    )

    # 4. Guardar
    save_coadd(args.output, final_image, ref_header, success_files, args.method)

    print(f"✨ Proceso finalizado. Output: {args.output}")

if __name__ == "__main__":
    main()
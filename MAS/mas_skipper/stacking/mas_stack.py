#!/usr/bin/env python3
import argparse
import glob
import os
import sys
import importlib.metadata

from .stackmas import validate_inputs, load_and_align_images, combine_stack, save_coadd, create_rgb_product

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

    parser.add_argument("--mode", type=str, choices=['deep', 'rgb'], default='deep',
                        help="DEV:Modo de operación: 'deep' (apilar mismo filtro) o 'rgb' (alinear 3 filtros distintos).")

    # --- Configuración FITS (HDU 0 vs 1) ---
    parser.add_argument("--ext", type=int, default=0,
                        help="Extensión donde leer el HEADER de FILTROS. Tus archivos suelen tenerlo en 0.")
    parser.add_argument("--data-ext", type=int, default=1,
                        help="Extensión donde está la IMAGEN (PIXELES). Tus archivos 'comb_' suelen tenerla en 1.")

    parser.add_argument("--ref-index", type=int, default=0,
                        help="Índice de la imagen a usar como referencia de alineación (0-based). Default: 0 (la primera).")

    parser.add_argument("--method", type=str, choices=['median', 'mean', 'sigmaclip'], default='median',
                        help="Método de combinación (SOLO DEEP). 'median' es robusto y rápido. 'sigmaclip' es mejor para limpiar trazas de satélites pero lento.")

    # Parámetros avanzados para sigma clipping
    parser.add_argument("--sigma", type=float, default=3.0,
                        help="Umbral sigma para rejection (solo DEEP y si method='sigmaclip'). Default: 3.0.")
    parser.add_argument("--iters", type=int, default=5,
                        help="Número de iteraciones para clipping (solo DEEP y si method='sigmaclip'). Default: 5.")
    parser.add_argument("--rgb-stretch", type=float, default=0.5,
                        help="Lupton stretch (contraste lineal). Default: 0.5 (para datos normalizados, sino 8).")
    parser.add_argument("--rgb-q", type=float, default=8.0,
                        help="Lupton Q (suavidad asinh). Valores altos (10) suavizan, bajos (0.1) resaltan tenues. Default: 8.0.")
    parser.add_argument("--no-rgb-scale", action="store_true",
                        help="Desactivar el auto-balance de canales. Úsalo si quieres ver la diferencia real de flujo entre filtros.")
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

    # 2. VALIDACIÓN DE FILTROS Y MODO
    #    Usamos args.ext (0) para leer los filtros
    valid, processed_info = validate_inputs(files, mode=args.mode, ext=args.ext)

    if not valid:
        print("❌ La validación de entradas falló. Revisa los mensajes anteriores.")
        sys.exit(1)

    # 3. EJECUCIÓN SEGÚN MODO

    # --- MODO DEEP (Apilado) ---
    if args.mode == 'deep':
        files_to_stack = processed_info  # validate_inputs devuelve la lista limpia en este modo

        if len(files_to_stack) < 2:
            print(f"⚠️ Se encontró solo {len(files_to_stack)} archivo válido. Se necesitan al menos 2 para apilar.")
            sys.exit(1)

        print(f"🚀 Iniciando DEEP STACK con {len(files_to_stack)} imágenes.")
        print(f"   Método: {args.method.upper()} | Data Ext: {args.data_ext}")

        # A. Cargar y Alinear (Ahora devuelve 4 valores)
        aligned_stack, ref_prim_header, ref_sci_header, success_files = load_and_align_images(
            files_to_stack,
            reference_idx=args.ref_index,
            extension=args.data_ext  # Usamos data-ext (1) para leer imagen
        )

        if aligned_stack is None:
            print("❌ Fallo crítico en alineación (insuficientes imágenes exitosas).")
            sys.exit(1)

        print(f"✅ Alineación exitosa de {len(success_files)}/{len(files_to_stack)} imágenes.")

        # B. Combinar
        final_image = combine_stack(
            aligned_stack,
            method=args.method,
            sigma=args.sigma,
            maxiters=args.iters
        )

        # C. Guardar (Pasando ambos headers)
        # Asegurar extensión .fits
        out_name = args.output if args.output.endswith('.fits') else f"{args.output}.fits"

        save_coadd(out_name, final_image, ref_prim_header, ref_sci_header, success_files, args.method)
        print(f"✨ Deep Field finalizado: {out_name}")

    # --- MODO RGB (Color) ---
    elif args.mode == 'rgb':
        rgb_dict = processed_info  # validate_inputs devuelve un diccionario {'R':.., 'G':.., 'B':..}

        print(f"🚀 Iniciando RGB ALIGNMENT.")
        print(f"   Base de salida: {args.output}")

        # Quitamos extensión si el usuario la puso por error, ya que generamos _R.fits, _G.fits, etc.
        base_name = os.path.splitext(args.output)[0]

        create_rgb_product(
            rgb_dict,
            base_name,
            extension=args.data_ext,
            stretch=args.rgb_stretch,
            Q=args.rgb_q,
            do_scaling=not args.no_rgb_scale  # True por defecto
        )
        print(f"✨ Proceso RGB finalizado.")
#--------------------------------------------------------------------------------------------

    # # 2. Cargar y Alinear
    # aligned_stack, ref_prim_header, ref_sci_header, success = load_and_align_images(
    #     files_to_stack,
    #     reference_idx=args.ref_index,
    #     extension=args.data_ext
    # )
    #
    # if aligned_stack is None:
    #     print("❌ Fallo crítico en alineación. Abortando.")
    #     sys.exit(1)
    #
    # print(f"✅ Alineación exitosa de {len(success_files)}/{len(files)} imágenes.")
    #
    # # 3. Combinar
    # final_image = combine_stack(
    #     aligned_stack,
    #     method=args.method,
    #     sigma=args.sigma,
    #     maxiters=args.iters
    # )
    #
    # # 4. Guardar
    # save_coadd(out_name, final_image, ref_prim_header, ref_sci_header, success, args.method)
    # print(f"✨ Proceso finalizado. Output: {args.output}")

if __name__ == "__main__":
    main()
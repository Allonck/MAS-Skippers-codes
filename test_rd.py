#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval

# Orden correcto de extensiones del MAS-CCD
MAS_EXT_ORDER = [1, 14, 16, 15, 13, 11, 12, 10, 5, 2, 4, 3, 9, 6, 8, 7]
GAP = 15  # Desplazamiento horizontal entre canales

def roi_shifting(roi):
    """Genera ROIs por extensión aplicando el corrimiento."""
    shifted = []
    for ext in MAS_EXT_ORDER:
        dx = GAP * (ext - 1)
        roi_shifted = [roi[0] + dx, roi[1] + dx, roi[2], roi[3]]
        shifted.append(roi_shifted)
    return shifted

def extract_noise(fits_file, roi_list):
    """Extrae ruido (std) de cada extensión dada una lista de ROIs."""
    noise_values = []
    with fits.open(fits_file) as hdul:
        for i, roi in enumerate(roi_list):
            data = hdul[i + 1].data
            if data is None:
                noise_values.append(np.nan)
                continue
            x1, x2, y1, y2 = roi
            roi_data = data[y1:y2, x1:x2]
            noise = np.std(roi_data)
            noise_values.append(noise)
    print(noise_values)
    return noise_values

def plot_noise(noise_values, output=None):
    """Grafica el ruido por extensión."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, 17), noise_values, tick_label=MAS_EXT_ORDER)
    ax.set_xlabel("Extensión MAS")
    ax.set_ylabel("Ruido (STD, ADU)")
    ax.set_title("Ruido de Lectura por Extensión")
    ax.grid(True)

    if output:
        fig.savefig(output)
        print(f"📁 Figura guardada en {output}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualiza el ruido (STD) por extensión del MAS CCD.")
    parser.add_argument("fits_file", type=str, help="Ruta al archivo FITS")
    parser.add_argument("--roi", type=int, nargs=4, metavar=('X1', 'X2', 'Y1', 'Y2'),
                        default=[545, 635, 540, 640],
                        help="ROI base a aplicar (se corregirá por canal)")
    parser.add_argument("--save", type=str, default=None,
                        help="Ruta para guardar la figura (opcional)")

    args = parser.parse_args()

    shifted_rois = roi_shifting(args.roi)
    noise = extract_noise(args.fits_file, shifted_rois)
    plot_noise(noise, output=args.save)

if __name__ == "__main__":
    main()

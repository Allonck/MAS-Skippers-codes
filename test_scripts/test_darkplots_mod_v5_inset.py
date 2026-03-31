#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def process_directory(path, roi):
    data_points = []
    fits_files = [f for f in os.listdir(path) if f.lower().endswith(('.fit', '.fits')) and not 'comb' in f.lower()]
    for fits_file in fits_files:
        full_path = os.path.join(path, fits_file)
        try:
            with fits.open(full_path) as hdul:
                header = hdul[0].header
                exp_time = header.get('EXPTIME') or header.get('EXPOSURE') or header.get('ITIME')
                if exp_time is None: continue
                data = hdul[1].data if len(hdul) > 1 else hdul[0].data
                y_min, y_max, x_min, x_max = roi
                roi_data = data[y_min:y_max, x_min:x_max]
                mean_roi = np.mean(roi_data)
                data_points.append((exp_time, mean_roi))
        except Exception: pass
    data_points.sort(key=lambda x: x[0])
    if len(data_points) >= 2:
        x = np.array([dp[0] for dp in data_points])
        y = np.array([dp[1] for dp in data_points])
        m, b = np.polyfit(x, y, 1)
        return x, y, m, b
    return None, None, None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi", type=int, nargs=4, default=[30, 130, 0, 100])
    args = parser.parse_args()

    directories = {
        'realNOAB': 'NO-AB (DG High = -2.0V)',
        'realAB_DGM2': 'AB ON (DG High = -2.0V)',
        'realAB_DGM2d3': 'AB ON (DG High = -2.3V)',
        'realAB_DGM2d5': 'AB ON (DG High = -2.5V)',
        'realAB_DGM3': 'AB ON (DG High = -3.0V)'
    }

    base_path = "." 
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # AJUSTE 1: Recuadro más arriba y a la izquierda para esquivar la línea roja
    axins = ax.inset_axes([0.05, 0.55, 0.4, 0.4])

    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    
    for i, (dir_name, label) in enumerate(directories.items()):
        dir_path = os.path.join(base_path, dir_name)
        if os.path.isdir(dir_path):
            x, y, m, b = process_directory(dir_path, args.roi)
            if m is not None:
                y_neto = y - b
                fit_neto = (m * x + b) - b
                
                ax.scatter(x, y_neto, color=colors[i], s=40, zorder=3)
                ax.plot(x, fit_neto, color=colors[i], linestyle='-', linewidth=2, label=f"{label} [m={m:.1f} ADU/s]", zorder=2)
                
                axins.scatter(x, y_neto, color=colors[i], s=20, zorder=3)
                axins.plot(x, fit_neto, color=colors[i], linestyle='-', linewidth=2, zorder=2)

    ax.set_xlabel("Exposure Time (s)", fontsize=15)
    ax.set_ylabel("Accumulated Glow in ROI (ADU)", fontsize=15)
    ax.grid(True, which="major", ls="--", alpha=0.5)
    
    # AJUSTE 2: Leyenda anclada manualmente en la esquina inferior derecha, pero elevada
    # El primer valor (0.98) es el eje X (casi pegado a la derecha).
    # El segundo valor (0.25) es el eje Y (25% hacia arriba desde el fondo).

    ax.legend(fontsize=10, loc='lower right', bbox_to_anchor=(0.98, 0.15))
    ax.set_xlim(-2, 65)
    ax.set_ylim(bottom=-5000, top=150000)
    
    # Zoom del recuadro
    x1, x2, y1, y2 = -2, 62, -20, 450 
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.grid(True, ls="--", alpha=0.5)
    axins.set_title("Zoom on Auto-Bias modes", fontsize=10)
    
    ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.tight_layout()
    plt.savefig('combined_amp_glow_inset.pdf', bbox_inches='tight')
    plt.savefig('combined_amp_glow_inset.png', bbox_inches='tight', dpi=300)
    print("Gráfico con recuadro guardado exitosamente.")

if __name__ == "__main__":
    main()
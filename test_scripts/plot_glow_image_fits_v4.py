#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_glow_gradient(fits_file, output_name="glow_image.pdf"):
    # Configurar estilo para paper (A&A)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 15

    # Cargar los datos
    try:
        with fits.open(fits_file) as hdul:
            data = hdul[1].data if len(hdul) > 1 else hdul[0].data
    except Exception as e:
        print(f"Error al leer el archivo FITS: {e}")
        return

    # --- RECORTAR LA REGIÓN DE INTERÉS ---
    prescan = 27
    active_area = 512
    overscan_to_show = 50
    
    x_start = prescan
    x_end = prescan + active_area + overscan_to_show
    
    data_cropped = data[:, x_start:x_end]
    
    # Calcular límites de color
    vmin = np.percentile(data_cropped, 5)
    vmax = np.percentile(data_cropped, 99.5)

    # Crear la figura
    fig, (ax_img, ax_prof) = plt.subplots(2, 1, figsize=(8, 7), 
                                          gridspec_kw={'height_ratios': [3, 1]}, 
                                          sharex=True)

    # --- Panel Superior: Imagen 2D ---
    im = ax_img.imshow(data_cropped, cmap='inferno', origin='lower', aspect='auto', 
                       vmin=vmin, vmax=vmax, extent=[x_start, x_end, 0, data_cropped.shape[0]])
    
    ax_img.set_ylabel('Row Index (Pixels)')
    
    # Barra de color superior
    divider_img = make_axes_locatable(ax_img)
    cax_img = divider_img.append_axes("right", size="2.5%", pad=0.03)
    cbar = fig.colorbar(im, cax=cax_img)
    cbar.set_label('Signal (ADU)')

    # --- Panel Inferior: Perfil 1D ---
    profile = np.median(data_cropped, axis=0)
    x_axis = np.arange(x_start, x_end)
    
    ax_prof.plot(x_axis, profile, color='black', linewidth=1.5)
    ax_prof.set_xlabel('Column Index (Pixels)')
    ax_prof.set_ylabel('Median ADU')
    ax_prof.grid(True, which='major', linestyle='--', alpha=0.5)
    
    # Límite X
    ax_prof.set_xlim(x_start, x_end)
    
    # Límite Y
    rango_y = np.max(profile) - np.min(profile)
    margen = max(10, rango_y * 0.05)
    ax_prof.set_ylim(bottom=np.min(profile) - margen, top=np.max(profile) + margen)

    # ALINEACIÓN PERFECTA: Crear un bloque invisible del mismo tamaño que la barra de color
    divider_prof = make_axes_locatable(ax_prof)
    cax_prof = divider_prof.append_axes("right", size="2.5%", pad=0.03)
    cax_prof.axis('off') # Ocultar este eje auxiliar

    # Eliminar espacio entre paneles
    plt.subplots_adjust(hspace=0.05)
    
    plt.savefig(output_name, bbox_inches='tight')
    plt.savefig(output_name.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"Figuras guardadas como {output_name} y su versión .png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera mapa 2D y perfil 1D del Amp Glow perfectamente alineados.")
    parser.add_argument("-f", "--file", type=str, 
                        default="/media/allon/HDD R2/MAS_DATA/06_19_2025/darks/realNOAB/darksDGM2_NOAB_0001.fits",
                        help="Ruta al archivo FITS")
    parser.add_argument("-o", "--output", type=str, default="glow_image.pdf",
                        help="Nombre del archivo de salida")
    args = parser.parse_args()
    
    plot_glow_gradient(args.file, args.output)
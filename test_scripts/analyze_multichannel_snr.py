import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import os

# --- CONFIGURACIÓN ---
# Archivo a analizar (debe existir en todas las carpetas)
FILENAME = "comb_cfbo_ESO342-11_0180_g_sdss_30s_1_e.fits"

# Mapeo de Canales -> Carpeta
# Asumo que 'reduced' es la versión completa (15 o 16 canales). Ajusta el número si es 16.
DIRS = {
    4:  'reduced_4',
    8:  'reduced_8',
    12: 'reduced_12',
    15: 'reduced'     # Asumiendo que esta es la de 15 canales (o 16)
}

# --- ¡¡AJUSTA ESTAS COORDENADAS VIENDO TU IMAGEN!! ---
GALAXY_CENTER = (329, 412)  # Centro del núcleo (x,y)
ARM_OFFSET    = (5, 80)    # Desplazamiento en píxeles hacia un brazo débil
SKY_CENTER    = (430, 405)  # (x, y) de una zona vacía sin estrellas

# Radios de apertura (pix)
R_CORE = 15
R_ARM  = 15
R_SKY  = 25

def get_flux_and_noise(data, center_obj, center_sky, r_obj, r_sky):
    """Calcula flujo neto y ruido de fondo local."""
    # 1. Estadística del Cielo (Ruido)
    y_sky, x_sky = np.ogrid[:data.shape[0], :data.shape[1]]
    mask_sky = ((x_sky - center_sky[0])**2 + (y_sky - center_sky[1])**2) <= r_sky**2
    
    sky_pixels = data[mask_sky]
    # Sigma clipping para eliminar rayos cósmicos/estrellas del fondo
    mean_bg, median_bg, std_bg = sigma_clipped_stats(sky_pixels, sigma=3.0)
    
    # 2. Flujo del Objeto
    y_obj, x_obj = np.ogrid[:data.shape[0], :data.shape[1]]
    mask_obj = ((x_obj - center_obj[0])**2 + (y_obj - center_obj[1])**2) <= r_obj**2
    
    obj_pixels = data[mask_obj]
    flux_raw = np.sum(obj_pixels)
    n_pix = len(obj_pixels)
    
    # Restar fondo
    flux_net = flux_raw - (median_bg * n_pix)
    
    return flux_net, std_bg

def analyze_channels():
    results = []

    print(f"Analizando imagen: {FILENAME}")
    print(f"Comparando configuraciones: {list(DIRS.keys())} canales")

    # Preparar visualización
    fig_vis, axes = plt.subplots(1, len(DIRS), figsize=(16, 4))
    
    for i, (n_ch, directory) in enumerate(sorted(DIRS.items())):
        filepath = os.path.join(directory, FILENAME)
        
        if not os.path.exists(filepath):
            print(f"⚠️ No encontrado: {filepath}")
            continue
            
        try:
            with fits.open(filepath) as hdul:
                # Buscar datos en ext 0 o 1
                data = hdul[0].data if hdul[0].data is not None else hdul[1].data
                data = data.astype(float)
                
                # Definir coordenadas brazo
                arm_x = GALAXY_CENTER[0] + ARM_OFFSET[0]
                arm_y = GALAXY_CENTER[1] + ARM_OFFSET[1]
                
                # Calcular SNR Núcleo
                flux_core, noise_core = get_flux_and_noise(data, GALAXY_CENTER, SKY_CENTER, R_CORE, R_SKY)
                snr_core = flux_core / noise_core
                
                # Calcular SNR Brazo Débil
                flux_arm, noise_arm = get_flux_and_noise(data, (arm_x, arm_y), SKY_CENTER, R_ARM, R_SKY)
                snr_arm = flux_arm / noise_arm
                
                results.append({
                    'N': n_ch,
                    'RON': noise_arm,  # Desviación estándar del cielo
                    'SNR_Core': snr_core,
                    'SNR_Arm': snr_arm
                })
                
                # --- VISUALIZACIÓN ---
                # Zoom centrado en la galaxia
                size = 180
                y1, y2 = int(GALAXY_CENTER[1]-size), int(GALAXY_CENTER[1]+size)
                x1, x2 = int(GALAXY_CENTER[0]-size), int(GALAXY_CENTER[0]+size)
                cutout = data[y1:y2, x1:x2]
                
                # Normalización (zscale-ish)
                vmin = np.median(data) - 1*noise_arm
                vmax = np.median(data) + 5*noise_arm
                
                ax = axes[i] if len(DIRS) > 1 else axes
                ax.imshow(cutout, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
                ax.set_title(f"M={n_ch} Canales\nSky $\sigma$={noise_arm:.2f}")
                ax.axis('off')
                
                # Dibujar círculos donde medimos
                circle_core = plt.Circle((size, size), R_CORE, color='r', fill=False)
                circle_arm = plt.Circle((size+ARM_OFFSET[0], size+ARM_OFFSET[1]), R_ARM, color='g', fill=False)
                ax.add_patch(circle_core)
                ax.add_patch(circle_arm)

        except Exception as e:
            print(f"Error leyendo {filepath}: {e}")

    plt.tight_layout()
    #plt.savefig('galaxy_channel_comparison.png')
    plt.savefig('galaxy_channel_comparison.png', bbox_inches='tight', pad_inches=0.1)
    print("✅ Guardado mapa visual: galaxy_channel_comparison.png")

    if not results: return

    # --- GRÁFICOS DE ANÁLISIS ---
    N_vals = np.array([r['N'] for r in results])
    RON_vals = np.array([r['RON'] for r in results])
    SNR_Core = np.array([r['SNR_Core'] for r in results])
    SNR_Arm = np.array([r['SNR_Arm'] for r in results])

    plt.figure(figsize=(12, 5))

    # Panel 1: Reducción de Ruido (RON vs sqrt(N))
    plt.subplot(1, 2, 1)
    plt.plot(np.sqrt(N_vals), RON_vals, 'o-', color='red', label='Ruido Medido ($\sigma_{sky}$)')
    
    # Línea teórica (ajustada al primer punto)
    # Teoría: Noise = k / sqrt(N)
    k = RON_vals[0] * np.sqrt(N_vals[0])
    y_theoretical = k / np.sqrt(N_vals)
    plt.plot(np.sqrt(N_vals), y_theoretical, 'k--', alpha=0.6, label=r'Teoría ($1/\sqrt{N}$)')
    
    plt.xlabel(r'$\sqrt{N_{canales}}$')
    plt.ylabel('Ruido de Fondo (ADU/e-)')
    plt.title('Reducción de Ruido por Promedio de Canales')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Panel 2: Mejora de SNR
    plt.subplot(1, 2, 2)
    # Normalizamos a N=4 para ver el factor de mejora
    plt.plot(np.sqrt(N_vals), SNR_Core / SNR_Core[0], 's-', color='blue', label='SNR Núcleo (Brillante)')
    plt.plot(np.sqrt(N_vals), SNR_Arm / SNR_Arm[0], 'o-', color='green', label='SNR Brazo (Débil)')
    
    # Línea de referencia (Mejora ideal)
    # Si N pasa de 4 a 16 (x4), sqrt(N) pasa de 2 a 4 (x2). La mejora debería ser lineal con sqrt(N)
    plt.plot(np.sqrt(N_vals), np.sqrt(N_vals)/np.sqrt(N_vals[0]), 'k--', label='Mejora Ideal (Limitado por Lectura)')
    
    plt.xlabel(r'$\sqrt{N_{canales}}$')
    plt.ylabel('Mejora Relativa de SNR')
    plt.title('Ganancia de Señal-Ruido')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('galaxy_snr_channels.png')
    print("✅ Guardado análisis numérico: galaxy_snr_channels.png")
    
    # Tabla
    print("\n--- RESULTADOS ---")
    print(f"{'Canales':<8} | {'Ruido':<8} | {'SNR Core':<10} | {'SNR Arm':<10}")
    for r in results:
        print(f"{r['N']:<8} | {r['RON']:.2f}     | {r['SNR_Core']:.1f}       | {r['SNR_Arm']:.1f}")

if __name__ == "__main__":
    analyze_channels()

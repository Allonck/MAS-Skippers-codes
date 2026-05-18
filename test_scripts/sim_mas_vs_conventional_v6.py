import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.stats import mad_std
from astropy.io import fits
import glob
import os
import warnings

# Configuración Estética
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 14)}) # Un poco más alto para info
warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN ---
LC_FILE = "lightcurve_data_R.fits"
COMB_PATTERN = "comb_cfbo_ARSAT1_*.fits"

# VALORES DUROS
RON_MAS_E = 1.56    # Ruido efectivo MAS (N=1, M=15)
RON_CONV_E = 4.51   # Ruido convencional (N=1, M=1)
READ_TIME_MAS = 10.0 # Segundos
DARK_CURRENT_E = 0.05 # Teórico conservador (<0.01 e-/s * 5s)

def get_sky_noise():
    comb_files = sorted(glob.glob(COMB_PATTERN))
    if comb_files:
        with fits.open(comb_files[0]) as hdul:
            try:
                data = hdul[1].data
            except:
                data = hdul[0].data
            h, w = data.shape
            center = data[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
            return mad_std(center, ignore_nan=True)
    return 10.0

def simulate_temporal_degradation(mjd, mag, t_read_mas, exptime, slow_factor):
    """Simula el aliasing temporal por penalización Iso-Noise."""
    # Ciclo MAS (Rápido)
    cycle_mas = exptime + t_read_mas
    
    # Ciclo Convencional (Lento por Skipper para bajar ruido)
    # T_conv = T_read + (T_read * (N-1)) aprox, o simplificado T_read*N
    read_time_conv = t_read_mas * slow_factor
    cycle_conv = exptime + read_time_conv
    
    print(f"\n⏱️ SIMULACIÓN TEMPORAL (Iso-Noise):")
    print(f"   Ciclo MAS: {cycle_mas:.1f} s")
    print(f"   Ciclo Conv: {cycle_conv:.1f} s (Factor {slow_factor:.1f}x)")
    
    # Binning
    t0 = mjd[0]
    t_sec = (mjd - t0) * 24 * 3600
    bins = np.arange(t_sec[0], t_sec[-1] + cycle_conv, cycle_conv)
    digitized = np.digitize(t_sec, bins)
    
    t_sim, mag_sim = [], []
    for i in range(1, len(bins)):
        mask = digitized == i
        if np.sum(mask) > 0:
            fluxes = 10**(-0.4 * mag[mask])
            avg_mag = -2.5 * np.log10(np.mean(fluxes))
            mid_time = (bins[i-1] + bins[i])/2 / 3600/24 + t0
            t_sim.append(mid_time)
            mag_sim.append(avg_mag)
            
    return np.array(t_sim), np.array(mag_sim), cycle_conv, cycle_mas

def main():
    print("--- ANÁLISIS INTEGRAL: MAS vs CONVENCIONAL ---")
    
    # 1. ANÁLISIS DE SENSIBILIDAD (PRESUPUESTO DE RUIDO)
    sky_e = get_sky_noise()
    
    # Ruido Total = sqrt(Sky^2 + RON^2 + Dark^2)
    noise_conv = np.sqrt(sky_e**2 + RON_CONV_E**2 + DARK_CURRENT_E) 
    noise_mas = np.sqrt(sky_e**2 + RON_MAS_E**2 + DARK_CURRENT_E)
    
    snr_gain = ((noise_conv / noise_mas) - 1) * 100

    print("\n📉 PRESUPUESTO DE RUIDO (Escenario Real):")
    print(f"   RON Convencional: {RON_CONV_E:.2f} e-")
    print(f"   RON MAS-CCD:      {RON_MAS_E:.2f} e-")
    print("-" * 30)
    print(f"   Ruido Cielo:      {sky_e:.2f} e-")
    print(f"   Dark (Teórico):   {np.sqrt(DARK_CURRENT_E):.2f} e-")
    print("-" * 30)
    print(f"   Ruido Total Conv: {noise_conv:.2f} e-")
    print(f"   Ruido Total MAS:  {noise_mas:.2f} e-")
    print(f"   >> MEJORA DE SENSIBILIDAD (SNR): +{snr_gain:.1f}%")

    if sky_e > 3 * RON_CONV_E:
        regime = "SKY-LIMITED"
    elif sky_e < RON_MAS_E:
        regime = "RON-LIMITED"
    else:
        regime = "TRANSITION ZONE"
    
    print(f"   >> RÉGIMEN: {regime}")

    # 2. ANÁLISIS DE VELOCIDAD (ISO-NOISE)
    # Cuántas muestras se necesitan para que el Convencional iguale al MAS?
    n_samples_req = (RON_CONV_E / RON_MAS_E) ** 2
    
    print(f"\n🐢 REQUISITOS ISO-NOISE (Para igualar {RON_MAS_E} e-):")
    print(f"   Muestras Skipper necesarias (N): {n_samples_req:.2f}")
    print(f"   >> FACTOR DE RALENTIZACIÓN: {n_samples_req:.2f}x")

    # 3. CARGA DE DATOS Y SIMULACIÓN
    if not os.path.exists(LC_FILE): return
    tbl = Table.read(LC_FILE)
    mjd, mag = tbl['MJD'], tbl['MAG_CAL']
    exptime = np.median(tbl['EXPTIME'])
    
    t_sim, mag_sim, cycle_conv, cycle_mas = simulate_temporal_degradation(
        mjd, mag, READ_TIME_MAS, exptime, n_samples_req
    )

    # 4. GRAFICAR
    t0 = mjd[0]
    mins_mas = (mjd - t0) * 24 * 60
    mins_sim = (t_sim - t0) * 24 * 60
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 12))
    
    # Panel A: MAS
    ax1.plot(mins_mas, mag, 'o-', color='tab:blue', markersize=4, label='MAS-CCD (Real)')
    ax1.invert_yaxis()
    ax1.set_ylabel("Magnitud Calibrada (R)")
    ax1.set_title(rf"A) MAS-CCD ($N=1$): Ciclo $\Delta t \approx {cycle_mas:.0f}$ s ($\sigma_{{tot}} \approx {noise_mas:.1f} e^-$)")
    ax1.grid(True, alpha=0.3); ax1.legend(loc='upper right')
    
    # Panel B: Convencional ISO-NOISE
    ax2.plot(mins_sim, mag_sim, 's--', color='tab:red', markersize=8, label='Skipper Convencional (Simulado)')
    ax2.plot(mins_mas, mag, '-', color='tab:blue', alpha=0.15)
    ax2.invert_yaxis()
    ax2.set_ylabel("Magnitud Calibrada (R)")
    ax2.set_xlabel("Tiempo (minutos)")
    ax2.set_title(rf"B) Convencional Iso-Noise: Ciclo $\Delta t \approx {cycle_conv:.0f}$ s (Factor {n_samples_req:.1f}x)")
    ax2.grid(True, alpha=0.3); ax2.legend(loc='upper right')
    
    # Info Box Consolidada
    info_text = (
        f"PERFORMANCE SUMMARY\n"
        f"Sensitivity Gain: +{snr_gain:.1f}%\n"
        f"Speed Advantage: {n_samples_req:.1f}x\n"
        f"Regime: {regime}"
    )
    ax2.text(0.02, 0.05, info_text, transform=ax2.transAxes, fontsize=11,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.savefig("mas_vs_conv_comprehensive_v6.png")
    print("\n✅ Gráfico guardado: mas_vs_conv_comprehensive_v6.png")
    plt.show()

if __name__ == "__main__":
    main()

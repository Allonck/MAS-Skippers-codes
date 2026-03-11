from astropy.io import fits
from astropy.table import Table
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURACIÓN ---
FITS_FILE = "lightcurve_data_R.fits"

# RANGO DE BÚSQUEDA (El "Zoom")
# Forzamos a buscar solo oscilaciones rápidas
SEARCH_MIN_PERIOD_MIN = 1.0   # Mínimo 1 minuto
SEARCH_MAX_PERIOD_MIN = 15.0  # Máximo 15 minutos (Ignora la tendencia de 40 min)

plt.style.use('seaborn-v0_8-paper')

def analyze_arsat1_zoom():
    if not os.path.exists(FITS_FILE):
        print(f"¡Error! No encuentro '{FITS_FILE}'")
        return

    # 1. Cargar datos
    tbl = Table.read(FITS_FILE)
    t_days = tbl['MJD'].value 
    mag = tbl['MAG_CAL'].value
    mag_err = tbl['MAG_ERR'].value
    
    # Filtrar
    mask = np.isfinite(mag)
    t_days = t_days[mask]
    mag = mag[mask]
    mag_err = mag_err[mask]
    
    # 2. DETRENDING (OPCIONAL PERO RECOMENDADO)
    # Ajustamos un polinomio de grado 2 para quitar la tendencia lenta
    # y dejar solo las oscilaciones rápidas.
    t_hours = (t_days - t_days[0]) * 24.0
    coeffs = np.polyfit(t_hours, mag, 2)
    poly = np.poly1d(coeffs)
    trend = poly(t_hours)
    mag_detrended = mag - trend
    
    print("Tendencia lenta sustraída para resaltar periodicidad rápida.")

    # 3. PERIODOGRAMA (Con rango restringido)
    # Frecuencias correspondientes a los periodos deseados
    min_freq = 60.0 / SEARCH_MAX_PERIOD_MIN  # Ciclos por hora
    max_freq = 60.0 / SEARCH_MIN_PERIOD_MIN
    
    # Usamos mag_detrended para el análisis de frecuencia
    frequency, power = LombScargle(t_hours, mag_detrended, dy=mag_err).autopower(
        minimum_frequency=min_freq, 
        maximum_frequency=max_freq
    )
    
    # Encontrar pico
    best_freq = frequency[np.argmax(power)]
    best_period_h = 1.0 / best_freq
    best_period_min = best_period_h * 60.0
    
    print(f"--- RESULTADOS NUEVOS ---")
    print(f"Periodo Rápido Detectado: {best_period_min:.4f} minutos")

    # 4. GRAFICAR
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])

    # Panel A: Curva Original vs Tendencia
    ax1 = fig.add_subplot(gs[0])
    ax1.errorbar(t_hours, mag, yerr=mag_err, fmt='.', color='gray', alpha=0.3, label='Data Cruda')
    ax1.plot(t_hours, trend, 'r--', label='Tendencia Lenta (Sustraída)')
    ax1.invert_yaxis()
    ax1.set_ylabel("Magnitud R")
    ax1.set_title("1. Eliminación de Tendencia Geométrica")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel B: Periodograma (Zoom)
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(1/frequency * 60, power, color='tab:blue')
    ax2.set_xlabel("Periodo (minutos)")
    ax2.set_ylabel("Potencia")
    ax2.set_xlim(SEARCH_MIN_PERIOD_MIN, SEARCH_MAX_PERIOD_MIN)
    
    # Marcar pico
    ax2.axvline(best_period_min, color='red', linestyle='--')
    ax2.text(best_period_min, max(power)*0.9, 
             f" T = {best_period_min:.2f} min", color='red', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_title("2. Periodograma (Búsqueda 1-15 min)")

    # Panel C: Curva Doblada (Usando data detrended)
    ax3 = fig.add_subplot(gs[2])
    phase = (t_hours / best_period_h) % 1
    
    # Graficar 2 ciclos
    ax3.errorbar(np.concatenate([phase, phase+1]), 
                 np.concatenate([mag_detrended, mag_detrended]), 
                 fmt='o', color='tab:purple', markersize=4, alpha=0.6)
    
    # Sin invertir eje Y aquí porque son residuos (oscilan alrededor de 0)
    # O invertimos si queremos mantener la lógica "arriba es más brillante" (valores negativos son más brillantes en delta mag)
    ax3.invert_yaxis() 
    ax3.set_xlabel("Fase Orbital")
    ax3.set_ylabel("Delta Magnitud (Residuo)")
    ax3.set_title(f"3. Curva Doblada (T = {best_period_min:.2f} min)")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='black', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig("arsat1_period_zoom.png")
    print("Gráfico guardado: arsat1_period_zoom.png")
    plt.show()

if __name__ == "__main__":
    analyze_arsat1_zoom()

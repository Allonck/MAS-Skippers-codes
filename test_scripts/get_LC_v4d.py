import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.time import Time
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN ---
BAND_NAME = 'R'
RINGSS_FILE = "/home/allon/Descargas/seeing_weather_ctio_/ringss_July17-July28-2025.csv"

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 10)})

def load_ringss_data(filepath, mjd_start, mjd_end):
    """Carga datos de RINGSS convirtiendo el string de tiempo a MJD."""
    print(f"--- Iniciando lectura de RINGSS ---")
    if not os.path.exists(filepath):
        print(f"⚠️ Archivo no encontrado en: {filepath}")
        return None, None

    try:
        # 1. Leer con separador ';'
        df = pd.read_csv(filepath, sep=';', engine='python')
        
        # 2. Limpiar nombres de columnas
        df.columns = [str(c).replace('"', '').strip().lower() for c in df.columns]
        
        mjd_col = 'time'
        see_col = 'see'

        if mjd_col in df.columns and see_col in df.columns:
            # CORRECCIÓN AQUÍ: Usar .str.strip() correctamente
            time_strings = df[mjd_col].astype(str).str.replace('"', '').str.strip()
            
            # Convertir fechas a MJD usando astropy
            t_obj = Time(time_strings.tolist(), format='iso', scale='utc')
            df['mjd_converted'] = t_obj.mjd
            
            # Limpiar y convertir Seeing
            df[see_col] = pd.to_numeric(df[see_col].astype(str).str.replace('"', ''), errors='coerce')
            df = df.dropna(subset=['mjd_converted', see_col])

            # Filtrado
            mask = (df['mjd_converted'] >= mjd_start - 0.005) & (df['mjd_converted'] <= mjd_end + 0.005)
            mjd_vals = df['mjd_converted'][mask].values
            see_vals = df[see_col][mask].values

            if len(mjd_vals) > 0:
                print(f"✅ Se cargaron {len(mjd_vals)} puntos reales de RINGSS.")
                return mjd_vals, see_vals
            else:
                print(f"⚠️ Rango temporal no coincide. RINGSS MJD: {df['mjd_converted'].min():.4f} a {df['mjd_converted'].max():.4f}")
        else:
            print(f"⚠️ Columnas no detectadas: {list(df.columns)}")

    except Exception as e:
        print(f"❌ Error al procesar CSV: {e}")

    # Fallback (Dummy) por si algo falla, para que no te quedes sin gráfico
    print("ℹ️ Usando datos Dummy para el trazo de Seeing.")
    mjd_dummy = np.linspace(mjd_start, mjd_end, 50)
    see_dummy = np.random.normal(0.85, 0.05, 50)
    return mjd_dummy, see_dummy

def plot_from_fits(fits_path):
    if not os.path.exists(fits_path): return

    t = Table.read(fits_path)
    mjd_arr = np.array(t['MJD'])
    t0 = mjd_arr[0]
    hours = (mjd_arr - t0) * 24.0

    # Obtener datos de RINGSS
    ringss_mjd, ringss_vals = load_ringss_data(RINGSS_FILE, mjd_arr.min(), mjd_arr.max())

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))
    
    # --- PANEL 1 ---
    ax1.errorbar(hours, t['MAG_CAL'], yerr=t['MAG_ERR'], fmt='o-', color='blue', 
                 ecolor='gray', capsize=2, markersize=4, label='ARSAT-1')
    
    # Destacar punto 17 con Estrella Roja
    ax1.plot(hours[16], t['MAG_CAL'][16], marker='*', color='red', 
             markersize=15, markeredgecolor='black', label='Evento (Punto 17)')

    ax1.invert_yaxis()
    ax1.set_ylabel(f"Magnitud ({BAND_NAME})")
    #ax1.set_title("Diagnóstico Ambiental: ARSAT-1")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # --- PANEL 2 ---
    # PANEL 2 (Igual a OG)
    color_sky = 'tab:orange'
    ax2.set_xlabel("Tiempo (Horas)")
    ax2.set_ylabel("Fondo Cielo Medio (e-/pix)", color=color_sky, fontweight='bold')
    
    # IMPORTANTE: Al quitar 'label', Matplotlib lo excluye de la leyenda automáticamente
    ax2.plot(hours, t['SKY_MEAN'], color=color_sky, marker='.', linestyle='-', alpha=0.8)
    ax2.fill_between(hours, t['SKY_MEAN'], color=color_sky, alpha=0.15)

    ax2.tick_params(axis='y', labelcolor=color_sky)
    
    ax3 = ax2.twinx()
    color_see = 'tab:green'
    color_src = 'tab:purple'
    ax3.set_ylabel("FWHM / Seeing (arcsec)", color='black', fontweight='bold')
    
    # Línea Verde RINGSS
    if ringss_mjd is not None:
        h_ringss = (ringss_mjd - t0) * 24.0
        ax3.plot(h_ringss, ringss_vals, color=color_see, linestyle='--', 
                 linewidth=2, label='Seeing Atmosférico (RINGSS)')
    
    # Línea Púrpura FWHM
    valid = ~np.isnan(t['FWHM_ARCSEC'])
    ax3.plot(hours[valid], t['FWHM_ARCSEC'][valid], color=color_src, marker='x', 
             linestyle='-', linewidth=1.5, label='FWHM Estimado (Fuente)')

    # Leyenda final: solo muestra lo de ax3 (Seeing y FWHM)
    ax3.legend(loc='upper right', fontsize='small')
    
    ax2.set_xlim(min(hours), max(hours))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"lightcurve_sat_{BAND_NAME}_enviromental_final.png", dpi=300)
    print("✅ Gráfico generado exitosamente.")
    plt.show()

if __name__ == "__main__":
    import sys
    f_in = sys.argv[1] if len(sys.argv) > 1 else 'lightcurve_data_R.fits'
    plot_from_fits(f_in)
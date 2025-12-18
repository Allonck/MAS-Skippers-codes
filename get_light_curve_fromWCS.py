from astropy.io import fits
from astropy.table import Table, Column
from astropy.time import Time
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

obj = "TOI-6805"

# --- CONFIGURACIÓN ---
ZP_FINAL = 0     
K_FINAL  = 0        
BAND_NAME = 'V'         

PIXEL_SCALE = 0.2546 

# CAMBIO 1: EN LUGAR DE PIXELES, USAMOS COORDENADAS CELESTES (J2000)
# Coordenadas aproximadas de TOI-6805 (Verifica en ExoFOP si quieres más precisión)
TARGET_RA  = 292.39216  # RA en Grados
TARGET_DEC = -34.09315  # Dec en Grados

# Radio de coincidencia (en píxeles) respecto a la predicción WCS
MATCH_RADIUS = 20.0                         
SKY_SIGMA_THRESHOLD = 3.0 

CATALOG_PATTERN = "phot_cat_comb_cfbo_*.fits" 
RINGSS_FILE = "/home/allon/Descargas/seeing_weather_ctio_/ringss_July17-July28-2025.csv" 

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 10)})

def load_ringss_data(filepath, mjd_start, mjd_end):
    if not os.path.exists(filepath):
        print(f"⚠️ Archivo RINGSS no encontrado: {filepath}")
        return None, None
    try:
        df = pd.read_csv(filepath)
        # Ajusta nombres si es necesario
        # mjd_ringss = df['MJD'].values
        # seeing_vals = df['Seeing'].values
        
        # DUMMY 
        mjd_ringss = np.linspace(mjd_start, mjd_end, 50)
        seeing_vals = np.random.normal(0.8, 0.05, 50)
        return mjd_ringss, seeing_vals
    except Exception as e:
        print(f"Error leyendo RINGSS: {e}")
        return None, None

def process_lightcurve():
    catfiles = sorted(glob.glob(CATALOG_PATTERN))
    if not catfiles: catfiles = sorted(glob.glob("phot_cat_comb_cfbo_*.fits"))
    if not catfiles: raise FileNotFoundError("No encontré catálogos de fotometría.")

    print(f"Procesando {len(catfiles)} catálogos usando WCS...")

    data = {
        'mjd': [], 'mag_cal': [], 'mag_inst': [], 'mag_err': [],
        'sky': [], 'fwhm': [], 'airmass': [], 'exptime': [], 'filename': []
    }
    
    # Objeto SkyCoord fijo
    target_coord = SkyCoord(ra=TARGET_RA*u.deg, dec=TARGET_DEC*u.deg)
    
    found_fwhm_col = None

    for catfile in catfiles:
        try:
            # CAMBIO 2: LEER WCS DE LA IMAGEN ORIGINAL
            # Reconstruimos el nombre de la imagen a partir del catálogo
            basename = os.path.basename(catfile)
            base_img_name = basename.replace("phot_cat_", "").replace("_ext1.fits", ".fits")
            
            # Buscamos la imagen para leer el header
            if os.path.exists(base_img_name):
                img_path = base_img_name
            elif os.path.exists(base_img_name.replace(".fits", "_ext1.fits")):
                 img_path = base_img_name.replace(".fits", "_ext1.fits")
            else:
                # Si no está la imagen, no podemos saber dónde está la estrella
                print(f"⚠️ Imagen original no encontrada para {basename}, saltando.")
                continue

            # Leer Header y WCS
            with fits.open(img_path) as hdul:
                header = hdul[0].header
                # A veces el WCS está en la extensión 1, prueba hdul[1] si falla
                try:
                    wcs = WCS(header)
                except:
                    wcs = WCS(hdul[1].header)
                
                exptime = float(header.get('EXPTIME', 1.0))
                airmass = float(header.get('AIRMASS', 1.0))
                date_obs = header.get('DATE-OBS', None)
                mjd = Time(date_obs, format='isot', scale='utc').mjd if date_obs else 0.0

            # CALCULAR POSICIÓN PIXEL ESPERADA
            target_pix = wcs.world_to_pixel(target_coord)
            expected_x, expected_y = float(target_pix[0]), float(target_pix[1])

            # --- A PARTIR DE AQUÍ ES IGUAL, PERO USANDO expected_x/y ---
            tbl = Table.read(catfile)
            cols = tbl.colnames
            
            x_key = next((k for k in ['x_fit', 'xcentroid', 'x'] if k in cols), None)
            y_key = next((k for k in ['y_fit', 'ycentroid', 'y'] if k in cols), None)
            mag_key = next((k for k in ['mag_fit', 'mag', 'mag_inst'] if k in cols), None)
            err_key = next((k for k in ['mag_err', 'mag_error'] if k in cols), None)
            sky_key = next((k for k in ['sky_mean', 'sky', 'background_mean'] if k in cols), None)
            fwhm_key = next((k for k in ['fwhm_est', 'fwhm', 'fwhm_image', 'ISOAREAF_IMAGE'] if k in cols), None)
            
            if fwhm_key and found_fwhm_col is None: found_fwhm_col = fwhm_key

            skies = np.array(tbl[sky_key]) if sky_key else np.zeros(len(tbl))
            fwhms = np.array(tbl[fwhm_key]) if fwhm_key else np.full(len(tbl), np.nan)
            inst_errs = np.array(tbl[err_key]) if err_key else np.full(len(tbl), 0.02)
            
            x_c = np.array(tbl[x_key])
            y_c = np.array(tbl[y_key])
            mags = np.array(tbl[mag_key])
            
            # MATCH usando la posición WCS calculada
            dist = np.hypot(x_c - expected_x, y_c - expected_y)
            min_idx = np.argmin(dist)
            
            if dist[min_idx] > MATCH_RADIUS: 
                # Estrella no encontrada cerca de la predicción WCS
                continue 
            
            mag_inst = mags[min_idx]
            this_sky = skies[min_idx]
            this_fwhm_raw = fwhms[min_idx]

            # Conversiones
            this_fwhm_arcsec = np.nan
            if not np.isnan(this_fwhm_raw) and this_fwhm_raw > 0:
                this_fwhm_arcsec = this_fwhm_raw * PIXEL_SCALE
            
            mag_norm = mag_inst + 2.5 * np.log10(exptime)
            mag_cal = mag_norm - (K_FINAL * airmass) - ZP_FINAL 
            
            data['mjd'].append(mjd)
            data['mag_cal'].append(mag_cal)
            data['mag_inst'].append(mag_inst)
            data['mag_err'].append(inst_errs[min_idx])
            data['sky'].append(this_sky)
            data['fwhm'].append(this_fwhm_arcsec)
            data['airmass'].append(airmass)
            data['exptime'].append(exptime)
            data['filename'].append(basename)
            
            # No actualizamos current_pos, confiamos en el WCS para el siguiente frame

        except Exception as e:
            continue

    if not data['mjd']:
        print("❌ ERROR: No se extrajeron datos.")
        return

    # Arrays
    arr_mjd = np.array(data['mjd'])
    arr_mag = np.array(data['mag_cal'])
    arr_sky = np.array(data['sky'])
    arr_err = np.array(data['mag_err'])
    arr_fwhm = np.array(data['fwhm'])
    
    # 4. FILTRADO
    median_sky = np.median(arr_sky)
    std_sky = np.std(arr_sky)
    threshold = median_sky + (SKY_SIGMA_THRESHOLD * std_sky)
    good_mask = arr_sky < threshold
    
    print(f"\n🔍 DIAGNÓSTICO FWHM: {found_fwhm_col}")
    
    mjd_clean = arr_mjd[good_mask]
    mag_clean = arr_mag[good_mask]
    err_clean = arr_err[good_mask]
    sky_clean = arr_sky[good_mask]
    fwhm_clean = arr_fwhm[good_mask]
    
    output_filename = f"lightcurve_data_{obj}_{BAND_NAME}_CLEAN.fits"
    t_out = Table()
    t_out['MJD'] = Column(mjd_clean, unit='d')
    t_out['MAG_CAL'] = Column(mag_clean, unit='mag')
    t_out['MAG_ERR'] = Column(err_clean, unit='mag')
    t_out['SKY_MEAN'] = Column(sky_clean, unit='adu')
    t_out['FWHM_ARCSEC'] = Column(fwhm_clean, unit='arcsec')
    t_out.write(output_filename, overwrite=True)

    # 5. GRAFICAR
    t_start, t_end = min(mjd_clean), max(mjd_clean)
    t0 = mjd_clean[0]
    hours = (mjd_clean - t0) * 24.0
    
    ringss_mjd, ringss_vals = load_ringss_data(RINGSS_FILE, t_start, t_end)
    hours_ringss = []
    vals_ringss_cut = []
    if ringss_mjd is not None:
        mask = (ringss_mjd >= t_start) & (ringss_mjd <= t_end)
        if np.any(mask):
            hours_ringss = (ringss_mjd[mask] - t0) * 24.0
            vals_ringss_cut = ringss_vals[mask]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))
    
    # Panel 1
    ax1.errorbar(hours, mag_clean, yerr=err_clean, fmt='o', color='royalblue', 
                 ecolor='gray', alpha=0.8, capsize=0, markersize=5, label=f'{obj}')
    y_low = np.percentile(mag_clean, 1); y_high = np.percentile(mag_clean, 99)
    margin = (y_high - y_low) * 0.2
    ax1.set_ylim(y_high + margin, y_low - margin)
    ax1.set_ylabel(f"Magnitud Instrumental ({BAND_NAME})")
    ax1.set_title(f"Curva de Luz WCS: {obj}")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # Panel 2
    color_sky = 'tab:orange'
    ax2.set_xlabel("Tiempo desde inicio (Horas)")
    ax2.set_ylabel("Fondo Cielo (e-/pix)", color=color_sky, fontweight='bold')
    ax2.plot(hours, sky_clean, color=color_sky, marker='.', linestyle='-', alpha=0.6, label='Sky Mean')
    ax2.tick_params(axis='y', labelcolor=color_sky)
    
    ax3 = ax2.twinx()
    
    if len(hours_ringss) > 0:
        ax3.plot(hours_ringss, vals_ringss_cut, color='tab:green', linestyle=':', linewidth=2, label='Seeing RINGSS')
    
    valid_fwhm = ~np.isnan(fwhm_clean)
    if np.any(valid_fwhm):
        ax3.plot(hours[valid_fwhm], fwhm_clean[valid_fwhm], 
                 color='tab:purple', marker='o', markersize=6, linestyle='--', alpha=0.9, label='FWHM Fuente')
    
    ax3.set_ylabel("FWHM (arcsec)", color='black')
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='upper right', frameon=True)
    
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"lightcurve_{obj}_{BAND_NAME}_filtered.png")
    print("✅ Gráfico generado.")

if __name__ == "__main__":
    process_lightcurve()

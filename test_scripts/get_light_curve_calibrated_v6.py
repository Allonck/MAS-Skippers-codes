from astropy.io import fits
from astropy.table import Table, Column
from astropy.time import Time
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

obj = "TOI-4948"
# --- CONFIGURACIÓN ---
ZP_FINAL = 0    
K_FINAL  = 0       
BAND_NAME = 'V'        

PIXEL_SCALE = 0.2546 
OBJ_START_POS = (177.0, 44.0) 
MATCH_RADIUS = 50.0                       
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
        # Ajusta esto según tu CSV real si falla
        # mjd_ringss = df['MJD'].values
        # seeing_vals = df['Seeing'].values
        
        # DUMMY (Borrar si tienes el CSV real funcionando)
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

    print(f"Procesando {len(catfiles)} catálogos...")

    data = {
        'mjd': [], 'mag_cal': [], 'mag_inst': [], 'mag_err': [],
        'sky': [], 'fwhm': [], 'airmass': [], 'exptime': [], 'filename': []
    }
    
    current_pos = list(OBJ_START_POS)
    
    # Variable para diagnóstico
    found_fwhm_col = None

    for catfile in catfiles:
        try:
            tbl = Table.read(catfile)
            cols = tbl.colnames
            
            x_key = next((k for k in ['x_fit', 'xcentroid', 'x'] if k in cols), None)
            y_key = next((k for k in ['y_fit', 'ycentroid', 'y'] if k in cols), None)
            mag_key = next((k for k in ['mag_fit', 'mag', 'mag_inst'] if k in cols), None)
            err_key = next((k for k in ['mag_err', 'mag_error'] if k in cols), None)
            sky_key = next((k for k in ['sky_mean', 'sky', 'background_mean'] if k in cols), None)
            
            # Búsqueda ampliada de FWHM
            fwhm_key = next((k for k in ['fwhm_est', 'fwhm', 'fwhm_image', 'fwhm_world', 'ISOAREAF_IMAGE'] if k in cols), None)
            
            if fwhm_key and found_fwhm_col is None:
                found_fwhm_col = fwhm_key # Guardamos el nombre para reportar

            skies = np.array(tbl[sky_key]) if sky_key else np.zeros(len(tbl))
            
            # Extracción FWHM
            if fwhm_key:
                fwhms = np.array(tbl[fwhm_key])
            else:
                fwhms = np.full(len(tbl), np.nan)
            
            inst_errs = np.array(tbl[err_key]) if err_key else np.full(len(tbl), 0.02)
            x_c = np.array(tbl[x_key])
            y_c = np.array(tbl[y_key])
            mags = np.array(tbl[mag_key])
            
            dist = np.hypot(x_c - current_pos[0], y_c - current_pos[1])
            min_idx = np.argmin(dist)
            
            if dist[min_idx] > MATCH_RADIUS: continue 
            
            mag_inst = mags[min_idx]
            this_sky = skies[min_idx]
            this_fwhm_raw = fwhms[min_idx]
            
            # 2. METADATOS
            basename = os.path.basename(catfile)
            base_img_name = basename.replace("phot_cat_", "")
            candidates = [base_img_name, base_img_name.replace("_ext1.fits", ".fits")]
            img_path = next((c for c in candidates if os.path.exists(c)), None)
            
            if img_path:
                with fits.open(img_path) as hdul:
                    header = hdul[0].header
                    exptime = float(header.get('EXPTIME', 1.0))
                    airmass = float(header.get('AIRMASS', 1.0))
                    date_obs = header.get('DATE-OBS', None)
                    mjd = Time(date_obs, format='isot', scale='utc').mjd if date_obs else 0.0
            else:
                mjd = 0.0; exptime = 10.0; airmass = 1.0

            # 3. CÁLCULO FWHM (Detectar si está en pixeles o arcsec)
            # A veces 'fwhm_world' ya está en grados/arcsec. Asumimos pixeles si valor > 0.1
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
            current_pos = [x_c[min_idx], y_c[min_idx]]

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
    
    print(f"\n🔍 DIAGNÓSTICO FWHM:")
    if found_fwhm_col:
        print(f"   Columna detectada: '{found_fwhm_col}'")
    else:
        print(f"   ⚠️ NO SE DETECTÓ COLUMNA DE FWHM EN LOS CATÁLOGOS.")
    
    valid_fwhm_count = np.count_nonzero(~np.isnan(arr_fwhm[good_mask]))
    print(f"   Puntos válidos de FWHM (post-filtrado): {valid_fwhm_count}")
    if valid_fwhm_count > 0:
        print(f"   Rango FWHM: {np.nanmin(arr_fwhm):.2f}\" - {np.nanmax(arr_fwhm):.2f}\"")

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
    ax1.set_title(f"Curva de Luz: {obj}")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # Panel 2
    color_sky = 'tab:orange'
    ax2.set_xlabel("Tiempo desde inicio (Horas)")
    ax2.set_ylabel("Fondo Cielo (e-/pix)", color=color_sky, fontweight='bold')
    ax2.plot(hours, sky_clean, color=color_sky, marker='.', linestyle='-', alpha=0.6, label='Sky Mean')
    ax2.tick_params(axis='y', labelcolor=color_sky)
    
    ax3 = ax2.twinx()
    color_see = 'tab:green'
    color_src = 'tab:purple'
    
    # Graficar RINGSS si existe
    if len(hours_ringss) > 0:
        ax3.plot(hours_ringss, vals_ringss_cut, color=color_see, linestyle=':', linewidth=2, label='Seeing RINGSS')
    
    # Graficar FWHM Fuente (MEJORADO)
    valid_fwhm = ~np.isnan(fwhm_clean)
    if np.any(valid_fwhm):
        ax3.plot(hours[valid_fwhm], fwhm_clean[valid_fwhm], 
                 color=color_src, marker='o', markersize=6, markeredgecolor='white',
                 linestyle='--', linewidth=1.5, alpha=0.9, label='FWHM Fuente')
    
    ax3.set_ylabel("FWHM (arcsec)", color='black')
    
    # Unir leyendas
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='upper right', frameon=True)
    
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"lightcurve_{obj}_{BAND_NAME}_filtered.png")
    print("✅ Gráfico generado.")

if __name__ == "__main__":
    process_lightcurve()

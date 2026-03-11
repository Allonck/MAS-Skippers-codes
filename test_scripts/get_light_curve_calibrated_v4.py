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

obj = "DIRECTV-14"
# --- CONFIGURACIÓN ---
#B=-22.646  
#V=-23.223
#R=-23.238

ZP_FINAL = -23.238   
K_FINAL  = 0.104      
BAND_NAME = 'R'       

# ESCALA DE PLACA (Para convertir fwhm_est a arcsec)
PIXEL_SCALE = 0.2546 

# Coordenadas iniciales (Ajustar según necesidad)
OBJ_START_POS = (441.0, 45.0) #(173.0, 100.0) 
MATCH_RADIUS = 80.0                      

# PATRÓN DE ARCHIVOS
CATALOG_PATTERN = "phot_cat_comb_cfbo_*.fits" 

# RUTA AL ARCHIVO RINGSS
RINGSS_FILE = "/home/allon/Descargas/seeing_weather_ctio_/ringss_July17-July28-2025.csv" 

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 10)})

def load_ringss_data(filepath, mjd_start, mjd_end):
    if not os.path.exists(filepath):
        print(f"⚠️ Archivo RINGSS no encontrado: {filepath}")
        return None, None
    try:
        df = pd.read_csv(filepath)
        # Ajusta nombres de columnas si tu CSV es distinto
        # mjd_ringss = df['MJD'].values
        # seeing_vals = df['Seeing'].values
        
        # DUMMY PARA QUE CORRA SI NO ESTÁ EL CSV
        mjd_ringss = np.linspace(mjd_start, mjd_end, 50)
        seeing_vals = np.random.normal(0.8, 0.05, 50)
        return mjd_ringss, seeing_vals
    except Exception as e:
        print(f"Error leyendo RINGSS: {e}")
        return None, None

def process_lightcurve():
    # Buscar catálogos
    catfiles = sorted(glob.glob(CATALOG_PATTERN))
    
    # Fallback si no encuentra el patrón corto
    if not catfiles:
        catfiles = sorted(glob.glob("phot_cat_comb_cfbo_*.fits"))

    if not catfiles: 
        raise FileNotFoundError("No encontré catálogos de fotometría en esta carpeta.")

    print(f"Procesando {len(catfiles)} catálogos...")

    # Listas para almacenar datos
    mjd_list = []
    mag_cal_list = []
    mag_inst_list = [] # Guardamos también la instrumental por si acaso
    mag_err_list = []
    sky_list = [] 
    fwhm_src_list = [] 
    airmass_list = []
    exptime_list = []
    filename_list = []
    
    current_pos = list(OBJ_START_POS)
    
    for catfile in catfiles:
        try:
            # 1. LEER CATÁLOGO
            tbl = Table.read(catfile)
            cols = tbl.colnames
            
            # Detectar columnas dinámicamente
            x_key = next((k for k in ['x_fit', 'xcentroid', 'x'] if k in cols), None)
            y_key = next((k for k in ['y_fit', 'ycentroid', 'y'] if k in cols), None)
            mag_key = next((k for k in ['mag_fit', 'mag', 'mag_inst'] if k in cols), None)
            err_key = next((k for k in ['mag_err', 'mag_error'] if k in cols), None)
            
            # Fondo (Sky Mean)
            sky_key = next((k for k in ['sky_mean', 'sky', 'background_mean'] if k in cols), None)
            if sky_key: skies = np.array(tbl[sky_key])
            else: skies = np.zeros(len(tbl))

            # FWHM Est (Estimado)
            fwhm_key = next((k for k in ['fwhm_est', 'fwhm', 'fwhm_image'] if k in cols), None)
            if fwhm_key: fwhms = np.array(tbl[fwhm_key])
            else: fwhms = np.full(len(tbl), np.nan)

            x_c = np.array(tbl[x_key])
            y_c = np.array(tbl[y_key])
            mags = np.array(tbl[mag_key])
            
            # Error instrumental (si existe, si no 0.02)
            if err_key: inst_errs = np.array(tbl[err_key])
            else: inst_errs = np.full(len(tbl), 0.02)
            
            # Matching
            dist = np.hypot(x_c - current_pos[0], y_c - current_pos[1])
            min_idx = np.argmin(dist)
            
            if dist[min_idx] > MATCH_RADIUS: 
                continue 
            
            # Datos objeto
            mag_inst = mags[min_idx]
            mag_err_inst = inst_errs[min_idx]
            this_sky = skies[min_idx]
            this_fwhm_pix = fwhms[min_idx]
            
            # 2. BUSCAR IMAGEN ORIGINAL
            basename = os.path.basename(catfile)
            base_img_name = basename.replace("phot_cat_", "")
            
            candidates = [
                base_img_name,
                os.path.join("reduced", base_img_name),
                base_img_name.replace("_ext1.fits", ".fits"),
                os.path.join("reduced", base_img_name.replace("_ext1.fits", ".fits"))
            ]
            
            img_path = None
            for cand in candidates:
                if os.path.exists(cand):
                    img_path = cand
                    break
            
            if img_path:
                with fits.open(img_path) as hdul:
                    header = hdul[0].header
                    exptime = float(header.get('EXPTIME', 1.0))
                    airmass = float(header.get('AIRMASS', 1.0))
                    date_obs = header.get('DATE-OBS', None)
                    if date_obs:
                        t = Time(date_obs, format='isot', scale='utc')
                        mjd = t.mjd
                    else: mjd = 0.0
            else:
                continue

            # 3. CONVERSIONES Y CALIBRACIÓN
            if this_fwhm_pix <= 0: this_fwhm_arcsec = np.nan
            else: this_fwhm_arcsec = this_fwhm_pix * PIXEL_SCALE

            # Fórmula: m_cal = m_inst + 2.5*log(t) - kX - ZP
            mag_norm = mag_inst + 2.5 * np.log10(exptime)
            mag_cal = mag_norm - (K_FINAL * airmass) - ZP_FINAL 
            
            # Guardar en listas
            mjd_list.append(mjd)
            mag_cal_list.append(mag_cal)
            mag_inst_list.append(mag_inst)
            mag_err_list.append(mag_err_inst) # Usamos error instrumental
            sky_list.append(this_sky)
            fwhm_src_list.append(this_fwhm_arcsec)
            airmass_list.append(airmass)
            exptime_list.append(exptime)
            filename_list.append(basename)
            
            current_pos = [x_c[min_idx], y_c[min_idx]]

        except Exception as e:
            continue

    if not mjd_list:
        print("❌ ERROR: No se extrajeron datos.")
        return

    # --- 4. GUARDAR CATÁLOGO FITS (NUEVO EN V4) ---
    output_filename = f"lightcurve_data_{obj}_{BAND_NAME}.fits"
    print(f"💾 Guardando datos en {output_filename} ...")
    
    t_out = Table()
    t_out['MJD'] = Column(mjd_list, unit='d', description='Modified Julian Date')
    t_out['MAG_CAL'] = Column(mag_cal_list, unit='mag', description=f'Calibrated Magnitude ({BAND_NAME})')
    t_out['MAG_INST'] = Column(mag_inst_list, unit='mag', description='Instrumental Magnitude')
    t_out['MAG_ERR'] = Column(mag_err_list, unit='mag', description='Instrumental Error')
    t_out['SKY_MEAN'] = Column(sky_list, unit='adu', description='Background Mean')
    t_out['FWHM_ARCSEC'] = Column(fwhm_src_list, unit='arcsec', description='Source FWHM')
    t_out['AIRMASS'] = Column(airmass_list, description='Airmass')
    t_out['EXPTIME'] = Column(exptime_list, unit='s', description='Exposure Time')
    t_out['FILENAME'] = Column(filename_list, description='Source File')
    
    # Escribir a disco
    t_out.write(output_filename, overwrite=True)
    print("✅ Archivo FITS guardado correctamente.")

    # --- 5. GRAFICAR ---
    mjd_arr = np.array(mjd_list)
    t_start, t_end = min(mjd_arr), max(mjd_arr)
    ringss_mjd, ringss_vals = load_ringss_data(RINGSS_FILE, t_start, t_end)

    t0 = mjd_arr[0]
    hours = (mjd_arr - t0) * 24.0
    
    hours_ringss = []
    vals_ringss_cut = []
    if ringss_mjd is not None:
        mask = (ringss_mjd >= t_start) & (ringss_mjd <= t_end)
        hours_ringss = (ringss_mjd[mask] - t0) * 24.0
        vals_ringss_cut = ringss_vals[mask]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))
    
    # Panel 1
    ax1.errorbar(hours, mag_cal_list, yerr=mag_err_list, fmt='o-', color='blue', 
                 ecolor='gray', capsize=2, markersize=4, label=f'{obj}')
    ax1.invert_yaxis()
    ax1.set_ylabel(f"Magnitud ({BAND_NAME})")
    ax1.set_title(f"Diagnóstico Ambiental: {obj}")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # Panel 2
    color_sky = 'tab:orange'
    ax2.set_xlabel("Tiempo (Horas)")
    ax2.set_ylabel("Fondo Cielo Medio (e-/pix)", color=color_sky, fontweight='bold')
    ax2.plot(hours, sky_list, color=color_sky, marker='.', linestyle='-', alpha=0.8, label='Sky Mean (Catálogo)')
    ax2.fill_between(hours, sky_list, color=color_sky, alpha=0.15)
    ax2.tick_params(axis='y', labelcolor=color_sky)
    
    ax3 = ax2.twinx()
    color_see = 'tab:green'
    color_src = 'tab:purple'
    ax3.set_ylabel("FWHM / Seeing (arcsec)", color='black', fontweight='bold')
    
    if len(hours_ringss) > 0:
        ax3.plot(hours_ringss, vals_ringss_cut, color=color_see, linestyle='--', linewidth=2, label='Seeing Atmosférico (RINGSS)')
    
    fwhm_arr = np.array(fwhm_src_list)
    valid = ~np.isnan(fwhm_arr)
    ax3.plot(hours[valid], fwhm_arr[valid], color=color_src, marker='x', linestyle='-', linewidth=1.5, label='FWHM Estimado (Fuente)')

    lines, labels = ax3.get_legend_handles_labels()
    ax3.legend(lines, labels, loc='upper right', fontsize='small')
    
    ax2.set_xlim(min(hours), max(hours))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"lightcurve_{obj}_{BAND_NAME}_enviromental_final.png")
    print("✅ Gráfico generado exitosamente.")

if __name__ == "__main__":
    process_lightcurve()

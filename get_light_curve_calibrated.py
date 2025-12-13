from astropy.io import fits
from astropy.table import Table, Column
from astropy.time import Time
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN DE CALIBRACIÓN (RESULTADOS TESIS) ---
ZP_FINAL = -23.274   
K_FINAL  = 0.104     
BAND_NAME = 'R'      

# --- CONFIGURACIÓN DE RASTREO ---
# CAMBIA ESTO CON LA POSICIÓN INICIAL REAL QUE VISTE EN DS9
OBJ_START_POS = (173.0, 100.0) 
MATCH_RADIUS = 70.0                      
CATALOG_PATTERN = "phot_cat_comb_cfbo_*.fits" 

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

def process_lightcurve():
    catfiles = sorted(glob.glob(CATALOG_PATTERN))
    if not catfiles:
        raise FileNotFoundError("No encontré catálogos.")

    print(f"Procesando {len(catfiles)} épocas para STAR ONE D2...")

    # Usaremos listas y luego convertiremos a Tabla Astropy directamente
    mjd_list = []
    mag_cal_list = []
    mag_err_list = []
    airmass_list = []
    exptime_list = []
    
    current_pos = list(OBJ_START_POS)
    
    for catfile in catfiles:
        try:
            basename = os.path.basename(catfile)
            parts = basename.split('_')
            
            # Buscar ID numérico
            img_id = [p for p in parts if p.isdigit() and len(p)==4]
            if not img_id: continue
            img_id = img_id[0]
            
            # Buscar FITS original
            original_candidates = glob.glob(f"*{img_id}*.fits")
            original_candidates = [c for c in original_candidates if "phot" not in c and "comb" not in c]
            
            if not original_candidates: continue
            original_fits = original_candidates[0]

            with fits.open(original_fits) as hdul:
                header = hdul[0].header
                exptime = float(header.get('EXPTIME', 1.0))
                # Forzar 1.0 si es 0 por error
                if exptime <= 0: exptime = 1.0
                
                airmass = float(header.get('AIRMASS', header.get('SECZ', 1.0)))
                date_obs = header.get('DATE-OBS', header.get('UTSHUT', None))
                
                if date_obs:
                    try:
                        t = Time(date_obs, format='isot', scale='utc')
                        mjd = t.mjd
                    except:
                        t = Time(date_obs, format='fits', scale='utc')
                        mjd = t.mjd
                else:
                    mjd = 0.0

            # Leer tabla
            tbl = Table.read(catfile)
            
            # Convertir a numpy arrays float
            x_centroids = np.array(tbl['xcentroid'], dtype=float)
            y_centroids = np.array(tbl['ycentroid'], dtype=float)
            mags = np.array(tbl['mag'], dtype=float)
            mag_errs = np.array(tbl['mag_err'], dtype=float)
            
            # Matching
            dx = x_centroids - current_pos[0]
            dy = y_centroids - current_pos[1]
            dist = np.hypot(dx, dy)
            
            min_dist_idx = np.argmin(dist)
            
            if dist[min_dist_idx] > MATCH_RADIUS:
                print(f"ID {img_id}: Perdido")
                continue 
            
            mag_inst = mags[min_dist_idx]
            mag_err_inst = mag_errs[min_dist_idx]
            
            # --- FÓRMULA CORREGIDA ---
            # 1. Normalizar flujo a 1s: m_norm = m_raw + 2.5 log(t)
            mag_norm = mag_inst + 2.5 * np.log10(exptime)
            
            # 2. Corregir atmósfera y ZP: m_cal = m_norm - kX - ZP
            # (Restamos ZP porque ZP es negativo, así que -(-23) = +23)
            mag_cal = mag_norm - (K_FINAL * airmass) - ZP_FINAL 
            
            mjd_list.append(mjd)
            mag_cal_list.append(mag_cal)
            mag_err_list.append(mag_err_inst) # Asumimos error instrumental ~ error calibrado
            airmass_list.append(airmass)
            exptime_list.append(exptime)
            
            current_pos = [x_centroids[min_dist_idx], y_centroids[min_dist_idx]]
            
            print(f"ID {img_id}: Mag={mag_cal:.3f}")

        except Exception as e:
            print(f"Error {e}")
            continue

    if not mjd_list:
        print("No se generaron datos.")
        return

    # --- GUARDAR A FITS ---
    t_out = Table()
    t_out['MJD'] = Column(mjd_list, unit='day')
    t_out['MAG_CAL'] = Column(mag_cal_list, unit='mag')
    t_out['MAG_ERR'] = Column(mag_err_list, unit='mag')
    t_out['AIRMASS'] = Column(airmass_list)
    t_out['EXPTIME'] = Column(exptime_list, unit='s')
    
    fits_name = f"lightcurve_starone_{BAND_NAME}.fits"
    t_out.write(fits_name, overwrite=True)
    print(f"\nGuardado FITS: {fits_name}")

    # --- GRAFICAR ---
    t0 = mjd_list[0]
    hours = (np.array(mjd_list) - t0) * 24.0
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.errorbar(hours, mag_cal_list, yerr=mag_err_list, 
                 fmt='o-', color='blue', ecolor='gray', capsize=2, markersize=4)
    plt.gca().invert_yaxis()
    plt.ylabel(f"Magnitud Calibrada ({BAND_NAME})")
    plt.title(f"Curva de Luz: STAR ONE D2")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(hours, airmass_list, 'r--', label='Airmass')
    plt.ylabel("Masa de Aire (X)")
    plt.xlabel("Tiempo (Horas)")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"lightcurve_starone_{BAND_NAME}.png")
    print("Gráfico guardado.")

if __name__ == "__main__":
    process_lightcurve()

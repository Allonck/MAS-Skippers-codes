from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN ---
OBJ_NAME = "TOI-4948"

# COORDENADAS (Del debug anterior)
OBJ_START_POS  = (175.57, 43.37)   # Target

# OPCIÓN A: La estrella brillante (la que intentamos antes)
# COMP_START_POS = (83.32, 13.61)    

# OPCIÓN B: La segunda estrella más brillante (Intentemos esta si la A falla)
# Es un poco menos brillante (Mag -9 vs -11) pero podría ser más estable.
COMP_START_POS = (73.81, 69.74) 

MATCH_RADIUS = 15.0  
SKY_SIGMA_THRESHOLD = 3.0 
CATALOG_PATTERN = "phot_cat_comb_cfbo_*.fits" 

plt.style.use('seaborn-v0_8-paper')

def process_differential_photometry():
    catfiles = sorted(glob.glob(CATALOG_PATTERN))
    if not catfiles: 
        print("❌ No encontré archivos.")
        return

    print(f"Procesando {len(catfiles)} catálogos...")
    print(f"   Target: {OBJ_START_POS}")
    print(f"   Comp:   {COMP_START_POS}")

    # Listas
    data = {'mjd': [], 'mag_obj': [], 'mag_comp': [], 'sky': []}
    
    curr_obj = list(OBJ_START_POS)
    curr_comp = list(COMP_START_POS)

    for i, catfile in enumerate(catfiles):
        try:
            tbl = Table.read(catfile)
            cols = tbl.colnames
            
            x_col = next((k for k in ['xcentroid', 'xcenter', 'x'] if k in cols), None)
            y_col = next((k for k in ['ycentroid', 'ycenter', 'y'] if k in cols), None)
            mag_col = next((k for k in ['mag', 'mag_inst'] if k in cols), None)
            sky_col = next((k for k in ['sky_mean', 'sky'] if k in cols), None)

            if not (x_col and y_col and mag_col): continue

            X = tbl[x_col].data
            Y = tbl[y_col].data
            MAG = tbl[mag_col].data
            SKY = tbl[sky_col].data if sky_col else np.zeros(len(X))
            
            # MATCH TARGET
            dist_obj = np.hypot(X - curr_obj[0], Y - curr_obj[1])
            idx_obj = np.argmin(dist_obj)
            if dist_obj[idx_obj] > MATCH_RADIUS: continue 
            
            # MATCH COMP
            dist_comp = np.hypot(X - curr_comp[0], Y - curr_comp[1])
            idx_comp = np.argmin(dist_comp)
            if dist_comp[idx_comp] > MATCH_RADIUS: continue

            # VALIDACIÓN DE DATOS (Anti-NaN)
            val_obj = MAG[idx_obj]
            val_comp = MAG[idx_comp]
            
            if np.isnan(val_obj) or np.isnan(val_comp):
                # Si alguna es NaN, saltamos este frame
                continue
                
            if np.isinf(val_obj) or np.isinf(val_comp):
                continue

            # MJD
            basename = os.path.basename(catfile)
            core_name = basename.replace("phot_cat_", "").replace("_ext1.fits", ".fits")
            mjd = 0
            if os.path.exists(core_name):
                with fits.open(core_name) as h:
                    date_obs = h[0].header.get('DATE-OBS')
                    if date_obs: mjd = Time(date_obs).mjd
            if mjd == 0: mjd = len(data['mjd']) 

            # Guardar
            data['mjd'].append(mjd)
            data['mag_obj'].append(val_obj)
            data['mag_comp'].append(val_comp)
            data['sky'].append(SKY[idx_obj])
            
            # Actualizar trackers
            curr_obj = [X[idx_obj], Y[idx_obj]]
            curr_comp = [X[idx_comp], Y[idx_comp]]
            
            # DEBUG: Imprimir los primeros 3 valores para ver qué está leyendo
            if i < 3:
                print(f"Frame {i}: Obj={val_obj:.3f} | Comp={val_comp:.3f}")

        except Exception:
            continue

    # Convertir a numpy
    mjd = np.array(data['mjd'])
    mag_obj = np.array(data['mag_obj'])
    mag_comp = np.array(data['mag_comp'])
    sky = np.array(data['sky'])

    if len(mjd) == 0:
        print("❌ Error: No quedaron datos válidos (todos eran NaN o sin match).")
        return

    # FILTRO NUBES
    med_sky = np.median(sky)
    limit = med_sky + (SKY_SIGMA_THRESHOLD * np.std(sky))
    mask = sky < limit
    
    print(f"Datos válidos: {len(mjd)} | Tras filtro nubes: {np.sum(mask)}")
    
    # Aplicar máscara
    t = mjd[mask]
    m_obj = mag_obj[mask]
    m_comp = mag_comp[mask]
    
    if len(t) == 0: return

    # TIEMPO
    if t[0] > 50000: t_hours = (t - t[0]) * 24.0
    else: t_hours = np.arange(len(t))

    # CÁLCULO DIFERENCIAL
    m_diff = m_obj - m_comp
    
    # Normalizar (restar mediana)
    m_obj_norm = m_obj - np.median(m_obj)
    m_comp_norm = m_comp - np.median(m_comp)
    m_diff_norm = m_diff - np.median(m_diff)

    # RMS ROBUSTO (Ignorando NaNs residuales si los hubiera)
    rms_abs = np.nanstd(m_obj_norm)
    rms_diff = np.nanstd(m_diff_norm)
    mejora = rms_abs / rms_diff if rms_diff > 0 else 0

    print("-" * 30)
    print(f"RMS Absoluta:    {rms_abs*1000:.2f} mmag")
    print(f"RMS Diferencial: {rms_diff*1000:.2f} mmag")
    print(f"Mejora:          {mejora:.1f}x")
    print("-" * 30)

    # GRAFICAR
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(t_hours, m_obj_norm, '.', label=rf'Target (Abs) $\sigma$={rms_abs*1e3:.1f} mmag', alpha=0.6, color='tab:blue')
    ax1.plot(t_hours, m_comp_norm + 0.05, '.', label='Comp (Abs) + Offset', color='gray', alpha=0.4)
    ax1.invert_yaxis()
    ax1.set_ylabel(r"$\Delta$ Mag Instrumental")
    ax1.set_title(f"Fotometría Diferencial: {OBJ_NAME}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_hours, m_diff_norm, 'o-', color='tab:red', label=rf'Diferencial $\sigma$={rms_diff*1e3:.1f} mmag', markersize=4)
    ax2.invert_yaxis()
    ax2.set_ylabel("Mag Diferencial (T - C)")
    ax2.set_xlabel("Tiempo (Horas)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_name = f"diff_photometry_v3_{OBJ_NAME}.png"
    plt.savefig(plot_name)
    print(f"✅ Gráfico guardado: {plot_name}")

    # Guardar FITS limpio para Exoplanet
    t_out = Table()
    t_out['MJD'] = t
    t_out['MAG_DIFF'] = m_diff
    t_out['MAG_ERR'] = np.full(len(t), rms_diff)
    t_out.write(f"lightcurve_diff_{OBJ_NAME}_CLEAN.fits", overwrite=True)

if __name__ == "__main__":
    process_differential_photometry()

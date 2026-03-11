import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.stats import linregress

# --- CONFIGURACIÓN ---
STANDARD_STARS = {
    'LTT4364': {'B': 11.666, 'V': 11.504, 'R': 11.331}, 
    'LTT6248': {'B': 12.288, 'V': 11.797, 'R': 11.478},
    'LTT7987': {'B': 12.276, 'V': 12.230, 'R': 12.292}, 
    'LTT9239': {'B': 12.677, 'V': 12.068, 'R': 11.671}
}

FILTER_MAP = {'ov': 'V', 'b': 'B', 'r': 'R'}

def safe_float(value):
    try:
        return float(value)
    except:
        return np.nan

def calc_bouguer_real():
    # Buscar catálogos
    cats = glob.glob("**/phot_cat_comb_cfbo_*.fits", recursive=True)
    if not cats: cats = glob.glob("LTT*/phot_cat_comb_cfbo_*.fits")
    
    print(f"Procesando {len(cats)} archivos...")
    data = []

    for cat_file in cats:
        try:
            filename = os.path.basename(cat_file)
            parts = filename.split('_')
            target = next((p for p in parts if "LTT" in p), None)
            if not target: continue
            
            idx = parts.index(target)
            img_id = parts[idx+1]
            filt_code = parts[idx+2]
            band = FILTER_MAP.get(filt_code, filt_code)
            
            # Buscar original para Header
            simple_name = f"{target}_{img_id}.fits"
            possible_paths = [
                simple_name,
                os.path.join(target, simple_name),
                os.path.join(os.path.dirname(cat_file), simple_name),
                f"../{simple_name}"
            ]
            
            original_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    original_path = p
                    break
            
            if not original_path: continue

            with fits.open(original_path) as h:
                airmass = safe_float(h[0].header.get('AIRMASS', h[0].header.get('SECZ', np.nan)))
                exptime = safe_float(h[0].header.get('EXPTIME', 1.0))
            
            if np.isnan(airmass) or exptime <= 0: continue

            # Leer Mag
            t = Table.read(cat_file)
            if 'mag' not in t.colnames: continue
            valid_mags = [safe_float(m) for m in t['mag'] if not np.isnan(safe_float(m))]
            if not valid_mags: continue
            
            mag_raw = min(valid_mags)
            
            # --- NORMALIZACIÓN CRÍTICA ---
            mag_norm = mag_raw + 2.5 * np.log10(exptime)
            
            if target not in STANDARD_STARS: continue
            if band not in STANDARD_STARS[target]: continue
            
            mag_cat = STANDARD_STARS[target][band]
            
            # Delta = m_inst - m_cat
            # Teoria: Delta = k*X + ZP
            delta = mag_norm - mag_cat
            
            data.append({
                'Band': band,
                'X': airmass,
                'Delta': delta,
                'File': simple_name
            })
            
        except Exception:
            continue

    df = pd.DataFrame(data)
    if df.empty:
        print("No hay datos.")
        return

    # --- AJUSTE Y GRÁFICO ---
    print("\nResultados Ley de Bouguer (Calculado):")
    print(f"{'Filtro':<6} | {'k (Extinción)':<15} | {'ZP (Intercept)':<15} | {'R²':<6}")
    print("-" * 60)

    for band in ['B', 'V', 'R']:
        subset = df[df['Band'] == band]
        if len(subset) < 3: continue
        
        # Filtro de Outliers Pre-Ajuste (Sigma Clipping simple)
        # Ajustamos una recta preliminar para sacar los puntos muy desviados
        slope_pre, intercept_pre, _, _, _ = linregress(subset['X'], subset['Delta'])
        residuals = np.abs(subset['Delta'] - (slope_pre * subset['X'] + intercept_pre))
        std_res = np.std(residuals)
        mask = residuals < 2.0 * std_res # 2 sigma clipping
        
        clean_subset = subset[mask]
        
        if len(clean_subset) < 3:
            print(f"{band:<6} | Puntos insuficientes tras filtrado")
            continue

        # Ajuste Final
        slope, intercept, r_val, p_val, std_err = linregress(clean_subset['X'], clean_subset['Delta'])
        
        print(f"{band:<6} | {slope:.4f} +/- {std_err:.4f} | {intercept:.4f}          | {r_val**2:.3f}")
        
        # Graficar
        plt.figure(figsize=(8, 6))
        plt.scatter(subset['X'], subset['Delta'], color='gray', alpha=0.5, label='Ignorados')
        plt.scatter(clean_subset['X'], clean_subset['Delta'], color='blue', label='Usados')
        
        x_line = np.linspace(clean_subset['X'].min(), clean_subset['X'].max(), 100)
        y_line = slope * x_line + intercept
        
        plt.plot(x_line, y_line, 'r--', label=f'k={slope:.3f}, ZP={intercept:.3f}')
        
        plt.title(f"Ley de Bouguer Real - Banda {band}")
        plt.xlabel("Masa de Aire (X)")
        plt.ylabel("$m_{inst} (1s) - m_{cat}$")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()
        plt.savefig(f"bouguer_real_{band}.png")
        # plt.show()

if __name__ == "__main__":
    calc_bouguer_real()

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import glob
import os
import pandas as pd
import seaborn as sns

# --- CONFIGURACIÓN ---
PIXEL_SCALE = 0.2546  # arcsec/pixel
INPUT_DIR = "reduced0"
FILE_PATTERN = "comb_cfbo_lightPollution_*_e.fits" # Usamos los archivos en electrones

# Puntos Cero (ZP)
# Nota: El ZP -23.274 es para Banda R. 
# Para B y OV usaremos 0.0 para ver diferencias relativas (gradientes).
ZPS = {
    'r': 23.238,  # Tu resultado Bouguer
    'v': 23.223,  # Tu resultado Bouguer (Excelente R2)
    'ov': 23.223,  # Tu resultado Bouguer (Excelente R2)
    'b': 22.646,  # Tu resultado Bouguer (Mayor error, pero útil)  
}

def parse_filename(filename):
    """
    Extrae metadata del nombre del archivo.
    Ej: comb_cfbo_lightPollution_0012_ov_180s_1_e.fits
    """
    basename = os.path.basename(filename)
    parts = basename.split('_')
    
    # Estructura típica basada en tu ls:
    # 0: comb, 1: cfbo, 2: lightPollution, 3: ID, 4: Filter, 5: ExptimeString
    try:
        img_id = int(parts[3])
        filt = parts[4]
        exptime_str = parts[5] # "180s"
        exptime = float(exptime_str.replace('s', ''))
        return img_id, filt, exptime
    except:
        return None, None, None

def remove_outliers(df):
    """Elimina filas que se desvíen más de 1.5 sigma en su grupo (Filtro+Ubicación)"""
    clean_df = pd.DataFrame()
    for (filt, loc), group in df.groupby(['Filter', 'Location']):
        mean = group['SB_mag_arcsec2'].mean()
        std = group['SB_mag_arcsec2'].std()
        
        # Si hay pocos datos (ej. 3), el std es muy sensible. 
        # Usamos un filtro simple: si se aleja más de 0.5 mag de la mediana, fuera.
        if len(group) < 5:
            median = group['SB_mag_arcsec2'].median()
            mask = np.abs(group['SB_mag_arcsec2'] - median) < 1.0 # Tolerancia de 1 mag
        else:
            mask = np.abs(group['SB_mag_arcsec2'] - mean) < 2 * std
            
        clean_group = group[mask]
        clean_df = pd.concat([clean_df, clean_group])
        
        removed = len(group) - len(clean_group)
        if removed > 0:
            print(f"🧹 Eliminados {removed} outliers en {filt} - {loc}")
            
    return clean_df

def analyze_sky():
    search_path = os.path.join(INPUT_DIR, FILE_PATTERN)
    files = sorted(glob.glob(search_path))
    
    if not files:
        print(f"❌ No encontré archivos en {search_path}")
        return

    print(f"Procesando {len(files)} imágenes reducidas desde {INPUT_DIR}...")
    
    records = []

    for f in files:
        img_id, filt, exptime = parse_filename(f)
        
        if img_id is None:
            continue

        # --- DETERMINAR UBICACIÓN POR ID ---
        # IDs 0-31 = Cénit (según tu estructura de carpetas anterior)
        # IDs 32-40 = La Serena
        if img_id >= 32:
            location = "La Serena (Horizonte)"
        else:
            location = "Cénit"

        try:
            with fits.open(f) as hdul:
                # Usar la primera extensión con datos
                data = hdul[0].data if hdul[0].data is not None else hdul[1].data
                data = data.astype(float)
                
                # --- ESTADÍSTICA DE FONDO ---
                # Sigma clipping robusto para ignorar estrellas
                mean, median, std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
                
                # CÁLCULO DE BRILLO SUPERFICIAL
                # Asumimos que la imagen ya tiene Bias restado (reduced) y está en electrones (_e)
                flux_e_sec = median / exptime
                
                if flux_e_sec <= 0:
                    continue

                # Flujo por arcsec cuadrado
                flux_arcsec2 = flux_e_sec / (PIXEL_SCALE ** 2)
                
                # Magnitud
                zp = ZPS.get(filt, 0.0)
                sb = -2.5 * np.log10(flux_arcsec2) + zp
                
                records.append({
                    'ID': img_id,
                    'Filter': filt.upper(), # R, B, OV
                    'Exptime': exptime,
                    'Location': location,
                    'Sky_Median_e': median,
                    'SB_mag_arcsec2': sb
                })
                
        except Exception as e:
            print(f"Error en {f}: {e}")
            continue

    if not records:
        print("No se generaron datos.")
        return

    df = pd.DataFrame(records)

    # --- LIMPIEZA ---
    print("\n--- APLICANDO FILTRO DE OUTLIERS ---")
    df = remove_outliers(df)

    # --- RESULTADOS POR CONSOLA ---
    print("\n" + "="*50)
    print(" ANÁLISIS DE CONTAMINACIÓN LUMÍNICA (Cénit vs LS)")
    print("="*50)
    
    # Agrupar por Filtro y Ubicación
    summary = df.groupby(['Filter', 'Location'])['SB_mag_arcsec2'].agg(['mean', 'std', 'count'])
    print(summary)
    
    # Calcular Delta (Diferencia de magnitud)
    print("\n--- IMPACTO DE LA CIUDAD (Diferencia de Magnitud) ---")
    for filt in df['Filter'].unique():
        subset = df[df['Filter'] == filt]
        locs = subset['Location'].unique()
        if len(locs) == 2:
            val_cenit = subset[subset['Location'] == 'Cénit']['SB_mag_arcsec2'].mean()
            val_ls = subset[subset['Location'] == 'La Serena (Horizonte)']['SB_mag_arcsec2'].mean()
            diff = val_cenit - val_ls # Cénit (oscuro, mag mayor) - LS (brillante, mag menor)
            ratio = 10**(0.4 * diff)
            
            print(f"Filtro {filt}: Delta = {diff:.2f} mag (LS es {ratio:.1f}x más brillante)")

    # --- GRÁFICOS ---
    plt.style.use('seaborn-v0_8-paper')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Boxplot Comparativo
    sns.boxplot(data=df, x='Filter', y='SB_mag_arcsec2', hue='Location', ax=axes[0], palette="Set2")
    axes[0].invert_yaxis()
    axes[0].set_title('Impacto de la Contaminación Lumínica')
    axes[0].set_ylabel(r'Brillo Superficial ($mag/arcsec^2$)')
    axes[0].grid(True, alpha=0.3)

    # 2. Scatter Plot por Exposición (Verificar linealidad/consistencia)
    sns.scatterplot(data=df, x='ID', y='SB_mag_arcsec2', hue='Filter', style='Location', s=100, ax=axes[1])
    axes[1].invert_yaxis()
    axes[1].set_title('Consistencia de Mediciones por ID')
    axes[1].set_xlabel('ID de Imagen')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('light_pollution_analysis_clean.png')
    print("\n✅ Gráfico guardado: light_pollution_analysis_clean.png")
    plt.show()

if __name__ == "__main__":
    analyze_sky()

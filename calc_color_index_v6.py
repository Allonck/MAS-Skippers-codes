from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
FILE_R = 'lightcurve_data_R.fits'
FILE_V = 'V/lightcurve_data_V.fits'

# Índices de imágenes "malas"
BAD_INDICES_R = [17]             
BAD_INDICES_V = [35, 40, 44, 72] 

# Configuración del Ajuste
POLY_DEGREE = 2       # Grado 2 (Parábola local)
SIGMA_CLIP = 2.0      # Umbral para descartar glints
MAX_ITERS = 5         # Iteraciones de limpieza
N_LAST_POINTS = 25    # <-- NUEVO: Usar solo los últimos N puntos de R

def iterative_polyfit(t, y, degree, sigma=2.0, max_iters=5):
    """Ajuste iterativo con sigma-clipping."""
    mask = np.ones(len(y), dtype=bool)
    print(f"Iniciando ajuste iterativo local (N={len(y)}, G{degree})...")
    
    for i in range(max_iters):
        if np.sum(mask) <= degree + 1:
            print("  Advertencia: Muy pocos puntos para ajustar.")
            break
            
        coeffs = np.polyfit(t[mask], y[mask], degree)
        model = np.poly1d(coeffs)
        residuals = y - model(t)
        std = np.std(residuals[mask])
        
        if std == 0: break # Ajuste perfecto o sin variación
        
        new_mask = np.abs(residuals) < (sigma * std)
        n_rejected = np.sum(~new_mask)
        
        print(f"  Iter {i+1}: std={std:.4f}, rechazados={n_rejected}")
        
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask
        
    return model, mask

def analyze_color_local():
    print("--- ANÁLISIS DE COLOR: AJUSTE LOCAL (ÚLTIMOS PUNTOS) ---")

    # 1. CARGA Y FILTRADO (Igual que antes)
    try:
        tbl_r = Table.read(FILE_R)
        tbl_v = Table.read(FILE_V)
    except FileNotFoundError:
        print("Error: Faltan archivos .fits")
        return

    # Procesar R
    mask_r = np.isfinite(tbl_r['MAG_CAL'])
    idx_r_all = np.arange(len(tbl_r))
    mask_manual_r = ~np.isin(idx_r_all, BAD_INDICES_R)
    final_mask_r = mask_r & mask_manual_r
    
    t_r = tbl_r['MJD'][final_mask_r].value
    mag_r = tbl_r['MAG_CAL'][final_mask_r].value
    
    # Ordenar R
    sort_r = np.argsort(t_r)
    t_r = t_r[sort_r]
    mag_r = mag_r[sort_r]

    # Procesar V
    mask_v = np.isfinite(tbl_v['MAG_CAL'])
    idx_v_all = np.arange(len(tbl_v))
    mask_manual_v = ~np.isin(idx_v_all, BAD_INDICES_V)
    final_mask_v = mask_v & mask_manual_v
    
    t_v = tbl_v['MJD'][final_mask_v].value
    mag_v = tbl_v['MAG_CAL'][final_mask_v].value

    # Sincronización temporal (Horas desde inicio global R)
    t0 = t_r[0]
    t_hours_r = (t_r - t0) * 24.0
    t_hours_v = (t_v - t0) * 24.0

    # 2. SELECCIÓN DE DATOS PARA AJUSTE (LOCAL)
    # Tomamos solo los últimos N puntos
    if len(t_hours_r) > N_LAST_POINTS:
        t_fit = t_hours_r[-N_LAST_POINTS:]
        m_fit = mag_r[-N_LAST_POINTS:]
        start_fit_time = t_fit[0] # Para saber dónde empezar a dibujar la línea roja
    else:
        t_fit = t_hours_r
        m_fit = mag_r
        start_fit_time = t_hours_r[0]

    # 3. AJUSTE ITERATIVO (Solo sobre la selección local)
    model, clean_mask_local = iterative_polyfit(t_fit, m_fit, POLY_DEGREE, SIGMA_CLIP, MAX_ITERS)

    # 4. CÁLCULO
    t_start_v = t_hours_v[0]
    window_v = t_hours_v <= (t_start_v + 0.041) # ~2.5 min
    
    target_time = np.mean(t_hours_v[window_v])
    measured_mag_v = np.mean(mag_v[window_v])
    predicted_mag_r = model(target_time)
    
    color_index = measured_mag_v - predicted_mag_r

    print("-" * 50)
    print(f"RESULTADO FINAL (Local {N_LAST_POINTS} ptos):")
    print(f"  R Predicho: {predicted_mag_r:.3f}")
    print(f"  V Medido:   {measured_mag_v:.3f}")
    print(f"  (V - R):    {color_index:.3f}")
    print("-" * 50)

    # 5. GRÁFICO
    plt.figure(figsize=(10, 6))

    # Todos los datos R (Fondo gris claro)
    plt.plot(t_hours_r*60, mag_r, '.', color='lightgray', label='Histórico R', zorder=1)
    
    # Datos R usados en el ajuste (Gris oscuro / Negro)
    # Identificar cuáles de los puntos de ajuste fueron rechazados
    t_fit_good = t_fit[clean_mask_local]
    m_fit_good = m_fit[clean_mask_local]
    t_fit_bad = t_fit[~clean_mask_local]
    m_fit_bad = m_fit[~clean_mask_local]
    
    plt.plot(t_fit_good*60, m_fit_good, '.', color='black', label=f'R (Últimos {N_LAST_POINTS})', zorder=2)
    plt.plot(t_fit_bad*60, m_fit_bad, 'rx', label='Outliers Locales', zorder=3)
    
    # Datos V
    plt.plot(t_hours_v*60, mag_v, 'g.', label='Datos V', zorder=2)

    # Modelo: Dibujar desde el inicio del ajuste local hasta el límite visual
    limit_time = target_time + (3.0/60.0) 
    t_plot = np.linspace(start_fit_time, limit_time, 100)
    
    plt.plot(t_plot*60, model(t_plot), 'r--', linewidth=2, label='Tendencia Local R')

    # Marcadores
    plt.scatter(target_time*60, predicted_mag_r, color='red', s=100, edgecolors='k', zorder=10)
    plt.scatter(target_time*60, measured_mag_v, color='lime', s=100, edgecolors='k', zorder=10)
    plt.vlines(target_time*60, predicted_mag_r, measured_mag_v, color='k', linestyle='-')
    
    # Límite Visual
    plt.axvline(limit_time*60, color='gray', linestyle=':', alpha=0.7)

    # Etiqueta
    mid_point = (predicted_mag_r + measured_mag_v)/2
    plt.text(target_time*60 + 1.5, mid_point, f"V-R = {color_index:.2f}", 
             fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.xlabel('Tiempo (Minutos desde inicio R)')
    plt.ylabel('Magnitud Calibrada')
    plt.title(f'Color ARSAT-1 (Ajuste Local últimos {N_LAST_POINTS} puntos)')
    plt.gca().invert_yaxis()
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    
    # Zoom inteligente: Mostrar desde un poco antes del ajuste local hasta el final
    plt.xlim(left=(start_fit_time*60 - 5), right=(limit_time*60 + 5))
    
    outfile = 'arsat1_color_local_poly.png'
    plt.savefig(outfile)
    print(f"Gráfico guardado: {outfile}")

if __name__ == "__main__":
    analyze_color_local()

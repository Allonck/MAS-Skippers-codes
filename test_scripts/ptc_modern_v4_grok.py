#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import linregress

# --- CONFIGURACIÓN DE ESTILO ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14


def roi_shifting(roi_base, ext):
    """Calcula el desfase de la ROI para cada extensión."""
    gap = 15
    x_start = roi_base[0] + (gap * (ext - 1))
    x_end = roi_base[1] + (gap * (ext - 1))
    return (slice(roi_base[2], roi_base[3]), slice(x_start, x_end))


#def analizar_ptc_fpn_ext(lista_archivos, roi, ext):
#    """Calcula varianza de diferencia y suma de medias."""
#    varianza_diferencia = []
#    suma_medias = []

#    for i in range(0, len(lista_archivos) - 1, 2):
#        try:
#            data_1 = fits.getdata(lista_archivos[i], ext=ext)[roi].astype(float)
#            data_2 = fits.getdata(lista_archivos[i + 1], ext=ext)[roi].astype(float)

#            sum_mean = np.mean(data_1) + np.mean(data_2)
#            diff_var = np.var(data_1 - data_2, ddof=1)

#            suma_medias.append(sum_mean)
#            varianza_diferencia.append(diff_var)
#        except Exception as e:
#            print(f"Error en par {i}-{i+1}, ext {ext}: {e}")
#
#    return np.array(varianza_diferencia), np.array(suma_medias)

def analizar_ptc_fpn_ext(lista_archivos, roi, ext, sigma_cr=5.0, maxiters=3):
    """
    Calcula suma de medias y varianza de diferencia con rechazo robusto de rayos cósmicos.
    
    Parámetros:
        sigma_cr : float
            Nivel de sigma clipping para detectar rayos cósmicos en la diferencia (recomendado 4.5 - 6.0)
        maxiters : int
            Número de iteraciones de clipping
    """
    varianza_diferencia = []
    suma_medias = []
    n_rechazados_total = 0

    for i in range(0, len(lista_archivos) - 1, 2):
        try:
            data_1 = fits.getdata(lista_archivos[i], ext=ext)[roi].astype(float)
            data_2 = fits.getdata(lista_archivos[i+1], ext=ext)[roi].astype(float)

            # --- Rechazo de rayos cósmicos en la diferencia ---
            diff = data_1 - data_2
            mask_good = np.ones(diff.shape, dtype=bool)

            for _ in range(maxiters):
                median_diff = np.median(diff[mask_good])
                std_diff = np.std(diff[mask_good], ddof=1)
                if std_diff == 0:
                    break
                new_mask = np.abs(diff - median_diff) <= sigma_cr * std_diff
                if np.sum(new_mask) == np.sum(mask_good):
                    break
                mask_good = new_mask

            n_rejected = np.sum(~mask_good)
            n_rechazados_total += n_rejected

            # Estadísticas robustas solo con píxeles buenos
            sum_mean = np.mean(data_1[mask_good]) + np.mean(data_2[mask_good])
            
            # Varianza de la diferencia (solo píxeles buenos)
            diff_good = diff[mask_good]
            diff_var = np.var(diff_good, ddof=1)

            suma_medias.append(sum_mean)
            varianza_diferencia.append(diff_var)

            if n_rejected > 0 and i % 4 == 0:   # imprimir ocasionalmente
                print(f"  Par {i:2d}-{i+1:2d} ext {ext:2d}: rechazados {n_rejected:4d} píxeles ({n_rejected/diff.size*100:.2f}%)")

        except Exception as e:
            print(f"Error procesando par {i}, {i+1} en ext {ext}: {e}")

    print(f"Ext {ext:2d}: Total píxeles rechazados por CR ≈ {n_rechazados_total}")
    return np.array(varianza_diferencia), np.array(suma_medias)

def linregress_sigma_clip(x, y, sigma=3.0, maxiters=10):
    """Ajuste lineal con sigma clipping iterativo."""
    if len(x) < 3:
        res = linregress(x, y)
        return res.slope, res.intercept, res.rvalue, res.stderr, np.ones(len(x), dtype=bool)

    mask = np.ones(len(x), dtype=bool)

    for _ in range(maxiters):
        res = linregress(x[mask], y[mask])
        y_pred = res.slope * x + res.intercept
        residuals = y - y_pred
        std_res = np.std(residuals[mask], ddof=1)

        if std_res == 0:
            break
        new_mask = np.abs(residuals) <= sigma * std_res
        if np.sum(new_mask) == np.sum(mask):
            break
        mask = new_mask

    res = linregress(x[mask], y[mask])
    return res.slope, res.intercept, res.rvalue, res.stderr, mask


def graficar_16_ptcs(datos_ext, output_base, lang='es', sigma_clip=3.0):
    """Genera gráfico y muestra tabla de resultados por extensión."""
    fig, ax = plt.subplots(figsize=(10, 8.5))
    colores = plt.cm.viridis(np.linspace(0, 0.9, 16))
    
    ganancias = []
    r_values = []
    slopes = []
    intercepts = []

    print("\n" + "="*80)
    print(f"{'Ext':>3} | {'Ganancia':>10} | {'Slope':>10} | {'Intercept':>10} | {'r':>8} | {'N pts':>6} | Rechazados")
    print("="*80)

    for ext in range(1, 17):
        x_var, y_mean = datos_ext[ext]
        if len(x_var) < 3:
            print(f"{ext:3d} | {'---':>10} | Muy pocos puntos")
            continue

        slope, intercept, r_value, _, mask_good = linregress_sigma_clip(
            x_var, y_mean, sigma=sigma_clip
        )

        gain = 1.0 / slope if slope > 0 else np.nan
        n_used = np.sum(mask_good)
        n_rejected = len(x_var) - n_used

        # Imprimir tabla
        print(f"{ext:3d} | {gain:10.4f} | {slope:10.6f} | {intercept:10.2f} | {r_value:8.5f} | "
              f"{n_used:6d} | {n_rejected:3d}")

        ganancias.append(gain)
        r_values.append(r_value)
        slopes.append(slope)
        intercepts.append(intercept)

        color = colores[ext - 1]
        y_fit = slope * x_var + intercept

        # Puntos buenos
        ax.scatter(x_var[mask_good], y_mean[mask_good], color=color, s=20, alpha=0.7, zorder=3)
        
        # Puntos rechazados
        if n_rejected > 0:
            ax.scatter(x_var[~mask_good], y_mean[~mask_good], edgecolor='gray', facecolor='none',
                       s=28, linewidth=1.2, alpha=0.8, zorder=2, marker='o')

        # Línea de ajuste
        ax.plot(x_var, y_fit, color=color, linestyle='-', linewidth=1.8, alpha=0.85, zorder=4)

    # Estadísticas globales
    mean_gain = np.nanmean(ganancias)
    std_gain = np.nanstd(ganancias)
    mean_r = np.nanmean(r_values)

    print("="*80)
    print(f"Ganancia promedio: {mean_gain:.4f} ± {std_gain:.4f} ADU/e-")
    print(f"r promedio:        {mean_r:.5f}")
    print("="*80)

    # Texto en el gráfico
    txt_stats = {
        'es': f'Ganancia Promedio (3σ)\n⟨K⟩ = {mean_gain:.3f} ± {std_gain:.3f} ADU/e⁻\n⟨r⟩ = {mean_r:.4f}',
        'en': f'Average Gain (3σ)\n⟨K⟩ = {mean_gain:.3f} ± {std_gain:.3f} ADU/e⁻\n⟨r⟩ = {mean_r:.4f}'
    }

    t = txt_stats.get(lang, txt_stats['es'])

    props = dict(boxstyle='round', facecolor='white', alpha=0.92, edgecolor='gray')
    ax.text(0.05, 0.95, t, transform=ax.transAxes, fontsize=15,
            verticalalignment='top', bbox=props, zorder=5)

    ax.set_xlabel('Varianza de la diferencia [ADU²]' if lang == 'es' else 'Difference Variance [ADU²]')
    ax.set_ylabel('Suma de las medias [ADU]' if lang == 'es' else 'Sum of Means [ADU]')
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        plt.savefig(f"{output_base}_16ch_{lang}.{fmt}", bbox_inches='tight', dpi=300 if fmt == 'png' else 200)
    plt.close()

    print(f"\nGráficos guardados: {output_base}_16ch_{lang}.[pdf/png]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análisis PTC con ajuste lineal 3σ por extensión")
    parser.add_argument("--lang", type=str, choices=['es', 'en'], default='es')
    parser.add_argument("--out", type=str, default="ptc_results")
    parser.add_argument("--sigma", type=float, default=3.0, help="Nivel de sigma clipping")
    
    args = parser.parse_args()

    archivos = [f"ptcVRfix29k_140Kdot9_RbandOK_{i:04d}.fits" for i in range(43)]
    roi_base = [250, 400, 250, 400]

    datos_por_extension = {}
    extensiones_mas = [1, 14, 16, 15, 13, 11, 12, 10, 5, 2, 4, 3, 9, 6, 8, 7]

    print("Iniciando análisis PTC para 16 extensiones...")

    for ext in extensiones_mas:
        print(f"Procesando extensión {ext:2d}...", end='\r')
        roi_desfasado = roi_shifting(roi_base, ext)
        x_var, y_mean = analizar_ptc_fpn_ext(archivos, roi_desfasado, ext)
        datos_por_extension[ext] = (x_var, y_mean)

    print("\nGenerando gráfico y tabla de ajustes...")
    graficar_16_ptcs(datos_por_extension, args.out, args.lang, sigma_clip=args.sigma)
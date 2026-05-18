import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuración estricta de estilo para A&A
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

def analizar_ventaja_mas(csv_path, ron_convencional=4.5):
    """
    Calcula la ganancia en SNR y genera un gráfico clasificando las observaciones
    en las tres zonas de ruido, usando sombreado de fondo para claridad.

    Args:
        csv_path (str): Ruta al archivo CSV.
        ron_convencional (float): Ruido de lectura de un CCD estándar.
    """
    df = pd.read_csv(csv_path)
    
    # Filtrar filas sin datos
    df = df.dropna(subset=['Ruido_Cielo_e', 'Ruido_Weighted_e'])
    
    # Calcular Ruido Total (simplificado: asumiendo Dark Current negligible)
    df['Total_Noise_Conv'] = np.sqrt(df['Ruido_Cielo_e']**2 + ron_convencional**2)
    df['Total_Noise_MAS'] = np.sqrt(df['Ruido_Cielo_e']**2 + df['Ruido_Weighted_e']**2)
    
    # Calcular Ventaja Real (Factor de ganancia en SNR)
    df['SNR_Gain'] = df['Total_Noise_Conv'] / df['Total_Noise_MAS']
    
    # Clasificar en zonas
    cond_rn = df['Ruido_Cielo_e'] <= df['Ruido_Weighted_e']
    cond_trans = (df['Ruido_Cielo_e'] > df['Ruido_Weighted_e']) & (df['Ruido_Cielo_e'] < 3 * ron_convencional)
    cond_sky = df['Ruido_Cielo_e'] >= 3 * ron_convencional
    
    # --- GRAFICACIÓN ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Obtener valores límite
    median_mas_ron = df['Ruido_Weighted_e'].median()
    x_max_plot = max(df['Ruido_Cielo_e'].max() + 2, 20)
    
    # 1. Sombreados de fondo (Background shading) en lugar de líneas
    # zorder=0 asegura que el sombreado quede detrás de los puntos y la grilla
    ax.axvspan(0, median_mas_ron, color='#2ca02c', alpha=0.08, zorder=0)
    ax.axvspan(median_mas_ron, 3 * ron_convencional, color='#1f77b4', alpha=0.08, zorder=0)
    ax.axvspan(3 * ron_convencional, x_max_plot, color='#d62728', alpha=0.08, zorder=0)
    
    # Agregar etiquetas en la parte superior del gráfico para demarcar los límites
    # Usamos una transformación híbrida (x en coordenadas de datos, y en coordenadas de ejes [0 a 1])
    ax.text(median_mas_ron, 1.01, r'$\sigma_{MAS}$', transform=ax.get_xaxis_transform(), 
            ha='center', va='bottom', fontsize=12, color='dimgrey')
    ax.text(3 * ron_convencional, 1.01, r'$3\sigma_{conv}$', transform=ax.get_xaxis_transform(), 
            ha='center', va='bottom', fontsize=12, color='dimgrey')

    # 2. Curva teórica
    x_theory = np.linspace(0.1, x_max_plot, 500)
    y_theory = np.sqrt(x_theory**2 + ron_convencional**2) / np.sqrt(x_theory**2 + median_mas_ron**2)
    ax.plot(x_theory, y_theory, 'k--', alpha=0.6, linewidth=1.5, label='Theoretical Gain')
    
    # 3. Scatter plot por zonas
    ax.scatter(df.loc[cond_rn, 'Ruido_Cielo_e'], df.loc[cond_rn, 'SNR_Gain'], 
               color='#2ca02c', alpha=0.8, s=35, edgecolor='none', label='RON-Dominated')
    
    ax.scatter(df.loc[cond_trans, 'Ruido_Cielo_e'], df.loc[cond_trans, 'SNR_Gain'], 
               color='#1f77b4', alpha=0.8, s=35, edgecolor='none', label='Transition Zone')
               
    ax.scatter(df.loc[cond_sky, 'Ruido_Cielo_e'], df.loc[cond_sky, 'SNR_Gain'], 
               color='#d62728', alpha=0.8, s=35, edgecolor='none', label='Sky-Dominated')

    # Configuración de ejes
    ax.set_xlabel(r'Sky Background Noise ($\sigma_{sky}$) [$e^-$ RMS]')
    ax.set_ylabel('SNR Gain Factor (Total Noise Conv / MAS)')
    ax.set_xlim(0, x_max_plot)
    
    # Ajustar un poco el límite superior de Y para que respire si hay puntos altos
    y_max_plot = max(df['SNR_Gain'].max() * 1.1, 1.5)
    ax.set_ylim(0.9, y_max_plot)
    
    ax.grid(True, linestyle='--', alpha=0.4, zorder=1)
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    
    # Guardar en ambos formatos (PDF para latex, PNG para visualizar fácil)
    plt.savefig('MAS_Advantage_Zones_AANDA.pdf')
    plt.savefig('MAS_Advantage_Zones_AANDA.png', dpi=300)
    plt.show()
    
    # Imprimir estadísticas
    print("--- Análisis de Subconjuntos ---")
    print(f"Archivos en RON-Dominated: {cond_rn.sum()}")
    print(f"Archivos en Transition Zone: {cond_trans.sum()}")
    print(f"Archivos en Sky-Dominated: {cond_sky.sum()}")
    print(f"\nGanancia mediana en Zona de Transición: {df.loc[cond_trans, 'SNR_Gain'].median():.2f}x")

# Ejecución
analizar_ventaja_mas('Ruido_Pipeline_Campana_Completa_with_SkyNoise_NOCALIB_NSAMP1_30kMySeq.csv', ron_convencional=4.5)
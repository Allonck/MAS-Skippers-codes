import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuración de estilo para A&A
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

def analizar_ventaja_mas(csv_path, ron_convencional=4.5):
    """
    Calcula la ganancia en SNR y genera un gráfico clasificando las observaciones
    en las tres zonas de ruido.

    Args:
        csv_path (str): Ruta al archivo CSV con las columnas de ruido de cielo.
        ron_convencional (float): Ruido de lectura de un CCD estándar de comparación.
    """
    df = pd.read_csv(csv_path)
    
    # Filtrar filas sin datos de cielo
    df = df.dropna(subset=['Ruido_Cielo_e', 'Ruido_Weighted_e'])
    
    # Calcular Ruido Total (simplificado: asumiendo Dark Current negligible)
    df['Total_Noise_Conv'] = np.sqrt(df['Ruido_Cielo_e']**2 + ron_convencional**2)
    df['Total_Noise_MAS'] = np.sqrt(df['Ruido_Cielo_e']**2 + df['Ruido_Weighted_e']**2)
    
    # Calcular Ventaja Real (Factor de ganancia en SNR)
    # Un valor de 2 significa el doble de SNR (equivalente a 4x tiempo de exposición en conv)
    df['SNR_Gain'] = df['Total_Noise_Conv'] / df['Total_Noise_MAS']
    
    # Clasificar en zonas
    cond_rn = df['Ruido_Cielo_e'] <= df['Ruido_Weighted_e']
    cond_trans = (df['Ruido_Cielo_e'] > df['Ruido_Weighted_e']) & (df['Ruido_Cielo_e'] < 3 * ron_convencional)
    cond_sky = df['Ruido_Cielo_e'] >= 3 * ron_convencional
    
    # --- GRAFICACIÓN ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Curva teórica
    x_theory = np.linspace(0.1, 20, 500)
    median_mas_ron = df['Ruido_Weighted_e'].median()
    y_theory = np.sqrt(x_theory**2 + ron_convencional**2) / np.sqrt(x_theory**2 + median_mas_ron**2)
    ax.plot(x_theory, y_theory, 'k--', alpha=0.5, label='Theoretical Gain')
    
    # Scatter plot por zonas
    ax.scatter(df.loc[cond_rn, 'Ruido_Cielo_e'], df.loc[cond_rn, 'SNR_Gain'], 
               color='#2ca02c', alpha=0.8, s=30, label='RON-Dominated')
    
    ax.scatter(df.loc[cond_trans, 'Ruido_Cielo_e'], df.loc[cond_trans, 'SNR_Gain'], 
               color='#1f77b4', alpha=0.8, s=30, label='Transition Zone')
               
    ax.scatter(df.loc[cond_sky, 'Ruido_Cielo_e'], df.loc[cond_sky, 'SNR_Gain'], 
               color='#d62728', alpha=0.8, s=30, label='Sky-Dominated')

    # Líneas de demarcación
    ax.axvline(median_mas_ron, color='gray', linestyle=':', label=r'$\sigma_{MAS}$')
    ax.axvline(3 * ron_convencional, color='gray', linestyle='-.', label=r'$3\cdot \sigma_{conv}$')
    
    # Configuración de ejes
    ax.set_xlabel(r'Sky Background Noise ($\sigma_{sky}$) [$e^-$ RMS]')
    ax.set_ylabel('SNR Gain Factor (Total Noise Conv / MAS)')
    #ax.set_title(f'MAS Photometric Advantage (Assumed $\sigma_{{conv}} = {ron_convencional} \, e^-$)')
    ax.set_xlim(0, max(df['Ruido_Cielo_e'].max() + 2, 20))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('MAS_Advantage_Zones.pdf')
    plt.show()
    
    # Imprimir estadísticas
    print("--- Análisis de Subconjuntos ---")
    print(f"Archivos en RON-Dominated: {cond_rn.sum()}")
    print(f"Archivos en Transition Zone: {cond_trans.sum()}")
    print(f"Archivos en Sky-Dominated: {cond_sky.sum()}")
    print(f"\nGanancia mediana en Zona de Transición: {df.loc[cond_trans, 'SNR_Gain'].median():.2f}x")

# Ejecución
#analizar_ventaja_mas('Ruido_Pipeline_Campana_Completa_with_SkyNoise_NOCALIB_NSAMP1.csv', ron_convencional=4.5)
analizar_ventaja_mas('Ruido_Pipeline_Campana_Completa_with_SkyNoise_NOCALIB_NSAMP1_30kMySeq.csv', ron_convencional=4.5)
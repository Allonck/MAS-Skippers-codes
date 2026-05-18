#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE ESTILO PASP/A&A ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 16          # Letra base más grande
plt.rcParams['axes.labelsize'] = 18     # Nombres de los ejes bien grandes
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 14    # Números de los ejes escalados
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14    # Leyendas legibles

def plot_linearity_with_residuals(exposure_times, signals_adu, gain, output_pdf="linearity_amp1.pdf", threshold_pct=1.0):
    
    # 1. Convertir Señal de ADU a electrones
    # Restamos el primer punto (offset) si es negativo para normalizar a 0 en el origen
    if signals_adu[0] < 0:
        signals_adu = signals_adu - signals_adu[0]
        
    signals_e = signals_adu / gain
    
    # 2. Ajuste lineal en el régimen seguro (hasta 30,000 e- o tiempo <= 10s)
    # Seleccionamos tiempos > 0 y < 15 para un ajuste puro
    idx_fit = np.where((exposure_times > 0) & (exposure_times <= 12.5))[0]
    
    if len(idx_fit) < 2:
        print("Error: No hay suficientes puntos para ajustar la zona lineal.")
        return

    m, c = np.polyfit(exposure_times[idx_fit], signals_e[idx_fit], 1)
    
    # Modelo lineal ideal extendido a todos los tiempos
    ideal_signal_e = m * exposure_times + c
    
    # 3. Cálculo de Residuos (%)
    with np.errstate(divide='ignore', invalid='ignore'):
        residuals_pct = ((signals_e - ideal_signal_e) / ideal_signal_e) * 100
        residuals_pct[0] = 0.0 # Evitar div/0 en t=0
        
    # 4. Encontrar FWC (El punto donde los residuos caen bajo el umbral negativo)
    # Ignoramos los primeros puntos por la varianza del obturador/ruido bajo
    fwc_limit_time = None
    fwc_limit_signal = None
    
    for i in range(3, len(residuals_pct)):
        if residuals_pct[i] < -threshold_pct:
            fwc_limit_time = exposure_times[i]
            fwc_limit_signal = signals_e[i]
            break

    # --- CREAR LA FIGURA ---
    # Figura más grande para el paper
    fig, (ax_main, ax_res) = plt.subplots(2, 1, figsize=(10, 8.5), 
                                          gridspec_kw={'height_ratios': [2.5, 1]}, 
                                          sharex=True)
    
    # --- Panel Superior ---
    # Puntos y líneas ligeramente más gruesos
    ax_main.scatter(exposure_times, signals_e, color='black', s=50, label='Dato (Amp 1)', zorder=3)
    ax_main.plot(exposure_times, ideal_signal_e, color='#1f77b4', linestyle='--', linewidth=2, label='Ajuste lineal', zorder=2)
    
    if fwc_limit_signal is not None:
        ax_main.axhline(fwc_limit_signal, color='#d62728', linestyle=':', linewidth=2,
                        label=f'FWC ($<{threshold_pct}\%$ dev) = {fwc_limit_signal:,.0f} $e^-$')
        ax_main.axvline(fwc_limit_time, color='#d62728', linestyle=':', alpha=0.5, linewidth=2)
        
    ax_main.set_ylabel('Señal media ($e^-$)')#ax_main.set_ylabel('Mean Signal ($e^-$)')
    ax_main.grid(True, linestyle='--', alpha=0.5)
    ax_main.legend(loc='upper left')

    # --- Panel Inferior ---
    ax_res.plot(exposure_times, residuals_pct, color='black', marker='o', markersize=5, linestyle='-')
    ax_res.axhline(0, color='#1f77b4', linestyle='--', linewidth=2)
    
    # Banda de tolerancia
    ax_res.axhspan(-threshold_pct, threshold_pct, color='#2ca02c', alpha=0.15, label=f'$\pm {threshold_pct}\%$ Tolerancia')
    
    if fwc_limit_time is not None:
        ax_res.axvline(fwc_limit_time, color='#d62728', linestyle=':', alpha=0.5, linewidth=2)
        
    ax_res.set_xlabel('Tiempo de exposición (s)') #ax_res.set_xlabel('Exposure Time (s)')
    ax_res.set_ylabel('Residuos (%)') #ax_res.set_ylabel('Residuals (%)')
    ax_res.grid(True, linestyle='--', alpha=0.5)
    
    # Filar límites Y del residuo para que se vea claro el corte (ej. entre -3.5% y +1.5%)
    ax_res.set_ylim(-3.5, 1.5) 
    
    # Agregar la leyenda para visualizar el label de axhspan
    ax_res.legend(loc='lower left')
    
    plt.subplots_adjust(hspace=0.05)
    
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.savefig(output_pdf.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"Figura guardada como: {output_pdf}")

if __name__ == "__main__":
    # ¡Tus datos reales extraídos del Jupyter!
    t_data = np.array([0., 2.5, 5., 7.5, 10., 12.5, 15., 17.5, 20., 22.5, 25., 27.5, 30., 32.5, 35., 37.5, 40., 42.5, 45., 47.5, 50.])
    y_data_amp1 = np.array([-1945.97, 119774.3, 218424.52, 316965.84, 414388.56, 510787.03, 606911.6, 703217.4, 799431.25, 896168.1, 992794.0, 1089128.6, 1183008.1, 1272311.1, 1357519.6, 1438375.8, 1516787.8, 1585414.2, 1612739.5, 1598161.6, 1572948.1])
    gain_amp1 = 54.087 
    
    plot_linearity_with_residuals(t_data, y_data_amp1, gain_amp1, output_pdf="ptc_TimevsSignal_corrES.pdf", threshold_pct=1.0)
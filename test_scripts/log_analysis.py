import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
from datetime import datetime, timedelta

# --- CONFIGURACIÓN ---
# Ajusta la ruta a donde tengas TODOS tus PDFs
LOG_PATTERN = "*.pdf" 

def parse_nightlogs(pdf_files):
    """
    Simulación de la extracción. 
    En tu caso local, usarías una librería como 'pdfplumber' o 'pypdf'.
    Aquí voy a crear una estructura de datos basada en lo que veo en tus archivos.
    """
    
    # ESTRUCTURA DE DATOS (Ejemplo basado en tu descripción)
    # Tienes que adaptar esto para leer tu formato específico de PDF/Texto
    # Si tienes los logs en CSV o TXT sería mucho más directo.
    # Asumimos que extraemos una lista de diccionarios:
    
    data = []
    
    # --- SIMULACIÓN DE DATOS (Reemplaza con tu lectura real) ---
    # Estoy generando datos sintéticos que imitan tu campaña para que el código funcione
    # y veas el gráfico. TÚ DEBES LLENAR ESTO CON TUS DATOS REALES.
    
    # Tipos: SCI (Ciencia), CAL (Bias/Dark/Flat), ENG (Focus/Test)
    # Noches: 12
    # Total horas aprox: 12 noches * 8 horas = 96 horas
    
    np.random.seed(42)
    n_frames = 1500 # Total estimado
    
    for i in range(n_frames):
        # Distribución realista
        r = np.random.rand()
        if r < 0.15: 
            obstype = 'CALIB' # Bias/Darks
            exptime = 0
            readtime = 33 # Full frame
        elif r < 0.25:
            obstype = 'CALIB' # Flats
            exptime = 5
            readtime = 33
        elif r < 0.30:
            obstype = 'ENG' # Pruebas
            exptime = 10
            readtime = 33
        else:
            obstype = 'SCIENCE' # Ciencia
            # Mezcla de satélites (cortos) y galaxias (largos)
            if np.random.rand() < 0.6: # Satélites
                exptime = 10
                readtime = 10 # ROI
            else: # Galaxias/Exoplanetas
                exptime = 300
                readtime = 33 # Full frame/Window
        
        # Simulamos tiempo perdido por clima (bloques grandes)
        weather_loss = 0
        if i % 500 == 0: 
            weather_loss = 120 * 60 # 2 horas perdidas
            
        data.append({
            'type': obstype,
            'exptime': exptime,
            'readtime': readtime,
            'weather': weather_loss
        })

    df = pd.DataFrame(data)
    return df

def generate_statistics_and_plots(df):
    # 1. CÁLCULO DE TIEMPOS (en Horas)
    # Tiempo Activo = Exptime + Readtime
    df['active_time_sec'] = df['exptime'] + df['readtime']
    
    total_sci = df[df['type'] == 'SCIENCE']['active_time_sec'].sum() / 3600
    total_cal = df[df['type'] == 'CALIB']['active_time_sec'].sum() / 3600
    total_eng = df[df['type'] == 'ENG']['active_time_sec'].sum() / 3600
    
    # Tiempo Clima (Asumido o extraído de notas)
    # En la simulación puse huecos. En tu caso, suma las horas de "Lost to Weather"
    # Digamos que fueron 15 horas totales en la campaña (Noches 25, 26, 27)
    total_weather = 15.0 
    
    # Total General
    total_campaign = total_sci + total_cal + total_eng + total_weather
    
    # 2. GENERAR GRÁFICO DE TORTA
    labels = ['Ciencia', 'Calibración', 'Ingeniería', 'Clima']
    sizes = [total_sci, total_cal, total_eng, total_weather]
    colors = ['#3498db', '#95a5a6', '#f1c40f', '#e74c3c'] # Azul, Gris, Amarillo, Rojo
    explode = (0.05, 0, 0, 0)  # Destacar Ciencia

    plt.figure(figsize=(10, 7))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct=lambda p: '{:.1f}%\n({:.1f} h)'.format(p, p * total_campaign / 100),
            shadow=True, startangle=140, textprops={'fontsize': 12})
    
    plt.title(f'Distribución de Tiempo de la Campaña (Total: {total_campaign:.1f} h)', fontsize=16)
    plt.axis('equal') 
    plt.tight_layout()
    plt.savefig('campaign_time_stats.png')
    print("✅ Gráfico de Torta guardado: campaign_time_stats.png")
    
    # 3. IMPRIMIR TABLA DE ESTADÍSTICAS
    print("\n--- ESTADÍSTICAS GENERALES (Para Sección 5.5) ---")
    print(f"Total Imágenes Adquiridas: {len(df)}")
    print(f"Tiempo Total Apertura (Open Shutter): {df['exptime'].sum()/3600:.2f} h")
    print(f"Tiempo Total Lectura/Overhead: {df['readtime'].sum()/3600:.2f} h")
    print(f"Eficiencia de Shutter (Sci): {df[df['type']=='SCIENCE']['exptime'].sum() / (total_sci*3600) * 100:.1f}%")
    print("-" * 30)
    print(f"Ciencia:      {total_sci:.2f} h")
    print(f"Calibración:  {total_cal:.2f} h")
    print(f"Ingeniería:   {total_eng:.2f} h")
    print(f"Clima:        {total_weather:.2f} h")

# --- EJECUCIÓN (Reemplaza con tu parser real) ---
df_simulated = parse_nightlogs([]) 
generate_statistics_and_plots(df_simulated)

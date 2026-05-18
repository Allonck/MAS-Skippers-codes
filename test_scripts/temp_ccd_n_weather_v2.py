import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import glob
import os
import numpy as np
import warnings
import datetime

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN ---
PDF_FOLDER = "."  
WEATHER_FILE = "/home/allon/Descargas/seeing_weather_ctio_/weather_July17-July28-2025.csv"
SAVE_PLOT = "ccd_thermal_stability_full.png"

# Estilo profesional
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 10)})

def add_night_shading(ax, start_date, end_date):
    """Sombrea las regiones nocturnas."""
    current_date = start_date.replace(hour=19, minute=0, second=0, microsecond=0)
    if current_date > start_date:
        current_date -= datetime.timedelta(days=1)
    final_date = end_date + datetime.timedelta(days=1)
    while current_date < final_date:
        next_morning = current_date + datetime.timedelta(hours=12)
        ax.axvspan(current_date, next_morning, color='gray', alpha=0.15, lw=0, zorder=-1)
        current_date += datetime.timedelta(days=1)

def parse_pdf_thermal(pdf_files):
    all_rows = []
    print(f"🌡️ Extrayendo datos térmicos de {len(pdf_files)} logs...")

    for pdf_file in pdf_files:
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if not row or row[0] == "Filename": continue
                            try:
                                # Indices: 3:CCDTEMP, 4:CCDSETP, 5:HEATERSP, 8:DATE-OBS
                                ccd_temp_raw = row[3]
                                ccd_setp_raw = row[4]
                                heater_raw = row[5]
                                date_obs = row[8]
                                
                                try: temp = float(ccd_temp_raw)
                                except: temp = np.nan
                                try: setp = float(ccd_setp_raw)
                                except: setp = np.nan
                                try: heater = float(heater_raw)
                                except: heater = np.nan
                                
                                # Filtro suave: solo basura extrema (<50K o >300K)
                                if pd.isna(temp) or temp < 50 or temp > 300: continue

                                all_rows.append({
                                    'dt': date_obs,
                                    'ccd_temp': temp,
                                    'ccd_setp': setp,
                                    'heater': heater
                                })
                            except: continue
        except: continue

    df = pd.DataFrame(all_rows)
    df['dt'] = pd.to_datetime(df['dt'], errors='coerce')
    df = df.dropna(subset=['dt']).sort_values('dt')
    return df

def load_weather_data():
    if not os.path.exists(WEATHER_FILE):
        print(f"❌ No encuentro {WEATHER_FILE}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(WEATHER_FILE, sep=';')
        df['time'] = pd.to_datetime(df['time'])
        return df.sort_values('time')
    except Exception as e:
        print(f"Error leyendo clima: {e}")
        return pd.DataFrame()

def plot_thermal_comparison(df_ccd, df_weather):
    t_start = df_ccd['dt'].min() - datetime.timedelta(hours=2)
    t_end = df_ccd['dt'].max() + datetime.timedelta(hours=2)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 12))
    
    # --- PANEL 1: TEMPERATURA CCD ---
    # NO FILTRAMOS NADA, MOSTRAR TODO EL RANGO
    
    # Graficamos el Setpoint
    ax1.plot(df_ccd['dt'], df_ccd['ccd_setp'], color='red', linestyle='--', linewidth=1.5, label='Setpoint', alpha=0.8)
    # Graficamos la Temperatura
    ax1.plot(df_ccd['dt'], df_ccd['ccd_temp'], color='blue', marker='o', linestyle='-', 
             linewidth=0.5, markersize=2, label='Temp. CCD', alpha=0.9)
    
    # Calcular estabilidad global (Residuales)
    residuals = df_ccd['ccd_temp'] - df_ccd['ccd_setp']
    rms = np.std(residuals)
    
    ax1.set_ylabel('Temp. CCD (K)')
    ax1.legend(loc='upper left')
    ax1.set_title(f'Historial Térmico Completo (Estabilidad Global RMS: {rms:.3f} K)')
    ax1.grid(True, alpha=0.3)
    add_night_shading(ax1, t_start, t_end)
    
    # Anotaciones de los cambios de régimen (Opcional, se ve visualmente)
    # ax1.text(df_ccd['dt'].iloc[0], 145, "Pruebas Iniciales (141K)", fontsize=9, rotation=0)

    # --- PANEL 2: HEATER POWER ---
    ax2.plot(df_ccd['dt'], df_ccd['heater'], color='#ff7f0e', label='Heater Output')
    ax2.set_ylabel('Potencia Heater (%)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 105) 
    add_night_shading(ax2, t_start, t_end)

    # --- PANEL 3: TEMPERATURA AMBIENTE ---
    if not df_weather.empty:
        mask = (df_weather['time'] >= t_start) & (df_weather['time'] <= t_end)
        df_w_cut = df_weather[mask]
        ax3.plot(df_w_cut['time'], df_w_cut['temp'], color='black', label='Temp. Ambiente')
        ax3.set_ylabel('Temp. Ambiente ($^{\circ}$C)')
        ax3.legend(loc='upper right')
    
    ax3.grid(True, alpha=0.3)
    add_night_shading(ax3, t_start, t_end)
    
    # Formato Fechas
    ax3.set_xlim(t_start, t_end)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45)
    ax3.set_xlabel('Fecha (UTC)')

    plt.tight_layout()
    plt.savefig(SAVE_PLOT, dpi=300)
    print(f"✅ Gráfico generado: {SAVE_PLOT}")
    plt.show()

if __name__ == "__main__":
    pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
    if pdf_files:
        df_ccd = parse_pdf_thermal(pdf_files)
        df_weather = load_weather_data()
        if not df_ccd.empty:
            plot_thermal_comparison(df_ccd, df_weather)
        else:
            print("❌ No data.")
    else:
        print("❌ No PDFs.")

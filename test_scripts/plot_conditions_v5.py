import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime

# Estilo profesional para tesis
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 8)})

def parse_data():
    # --- Cargar Datos ---
    df_weather = pd.read_csv('weather_July17-July28-2025.csv', sep=';')
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_weather = df_weather.sort_values('time')
    
    df_ringss = pd.read_csv('ringss_July17-July28-2025.csv', sep=';')
    df_ringss['time'] = pd.to_datetime(df_ringss['time'])
    df_ringss = df_ringss.sort_values('time')
    df_ringss = df_ringss[(df_ringss['see'] > 0.3) & (df_ringss['see'] < 3.0)]

    # No filtramos datos finales aquí para aprovechar todo lo que haya hasta el 29
    return df_weather, df_ringss

def add_night_shading(ax, start_date, end_date):
    """Sombrea las regiones nocturnas (19:00 a 07:00)."""
    # Empezar el shading desde el mediodía del día de inicio para asegurar que agarra la primera noche
    current_date = start_date.replace(hour=19, minute=0, second=0, microsecond=0)
    # Si la fecha de inicio del plot es posterior a las 19:00, retroceder un día para sombrear correctamente
    if current_date > start_date:
        current_date -= datetime.timedelta(days=1)
        
    final_date = end_date + datetime.timedelta(days=1)
    
    while current_date < final_date:
        next_morning = current_date + datetime.timedelta(hours=12)
        ax.axvspan(current_date, next_morning, color='gray', alpha=0.15, lw=0, zorder=-1)
        current_date += datetime.timedelta(days=1)

def plot_weather(df):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
    
    # Definir límites exactos para cuadrar noches completas
    # Inicio: Mediodía del 17 (para ver la tarde previa a la primera noche)
    # Fin: Mediodía del 29 (para cerrar la mañana tras la última noche del 28)
    plot_start = pd.Timestamp('2025-07-17 12:00:00')
    plot_end = pd.Timestamp('2025-07-29 12:00:00')

    # Temperatura
    ax1.plot(df['time'], df['temp'], color='#d62728', lw=1)
    add_night_shading(ax1, plot_start, plot_end)
    ax1.set_ylabel('Temperatura ($^{\circ}$C)')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Condiciones Ambientales (CTIO) - Julio 2025\n(Zonas sombreadas: Noche Astronómica)')
    
    # Humedad
    ax2.plot(df['time'], df['hum'], color='#1f77b4', lw=1)
    add_night_shading(ax2, plot_start, plot_end)
    ax2.set_ylabel('Humedad Relativa (%)')
    ax2.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Límite Operativo')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    
    # Viento
    ax3.plot(df['time'], df['wspeed'], color='#2ca02c', lw=1)
    add_night_shading(ax3, plot_start, plot_end)
    ax3.set_ylabel('Velocidad Viento (m/s)')
    ax3.axhline(y=15, color='orange', linestyle='--', alpha=0.5, label='Alerta Viento')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=10)
    
    ax3.set_xlim(left=plot_start, right=plot_end)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('weather_conditions_night_v4.png', dpi=300)
    print("Gráfico de clima generado.")

def plot_seeing(df):
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(1, 4)
    
    # Mismos límites para el seeing
    plot_start = pd.Timestamp('2025-07-17 12:00:00')
    plot_end = pd.Timestamp('2025-07-29 12:00:00')

    # Panel Principal
    ax_main = fig.add_subplot(gs[0, 0:3])
    add_night_shading(ax_main, plot_start, plot_end)
    
    ax_main.scatter(df['time'], df['see'], s=5, alpha=0.4, color='black', label='Mediciones DIMM')
    df_resampled = df.set_index('time').resample('1h').mean()
    ax_main.plot(df_resampled.index, df_resampled['see'], color='red', lw=2, label='Media Horaria')
    
    ax_main.set_ylabel('Seeing (arcsec) @ 0.5 $\mu$m')
    ax_main.set_ylim(0.4, 2.5)
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()
    ax_main.set_title('Calidad de Cielo (Seeing)')
    
    ax_main.set_xlim(left=plot_start, right=plot_end)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45)
    
    # Histograma con Estadística
    ax_hist = fig.add_subplot(gs[0, 3], sharey=ax_main)
    ax_hist.hist(df['see'], bins=30, orientation='horizontal', color='gray', alpha=0.7, density=True)
    
    median_see = df['see'].median()
    mean_see = df['see'].mean()
    std_see = df['see'].std()
    
    # Mediana
    ax_hist.axhline(median_see, color='red', linestyle='--')
    ax_hist.text(0.1, median_see + 0.05, f'Med: {median_see:.2f}"', color='red', fontweight='bold', fontsize=10)
    
    # Media con formato LaTeX solicitado
    text_y_pos = 2.3
    # Usamos string raw r'' para evitar problemas con backslashes en latex
    label_text = r'$\bar{x}$ = ' + f'{mean_see:.2f}"\n' + r'$\pm$ ' + f'{std_see:.2f}"'
    
    ax_hist.text(0.1, text_y_pos, label_text, color='black', fontsize=10, 
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax_hist.set_xlabel('Frecuencia')
    plt.setp(ax_hist.get_yticklabels(), visible=False)
    
    plt.tight_layout()
    plt.savefig('seeing_statistics_night_v4.png', dpi=300)
    print("Gráfico de seeing generado.")

if __name__ == "__main__":
    weather, ringss = parse_data()
    plot_weather(weather)
    plot_seeing(ringss)

import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np
import re
import warnings
from datetime import timedelta

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN ---
PDF_FOLDER = "." 
READ_TIME_FULL = 33.0 
READ_TIME_ROI = 10.0 
OVERHEAD_FILE = 5.0    
NIGHT_LENGTH_STD = 10.0 
# Hora local límite: Si la última foto es antes de las 03:00 AM Local, es pérdida por clima
CRITICAL_END_HOUR_LOCAL = 3 
UTC_OFFSET = -4 # Chile Invierno

# Mapeo de filtros
FILTER_MAP = {
    'ov': 'V', 'v': 'V', 'b': 'B', 'r': 'R', 'i': 'I',
    'g_sdss': 'g (SDSS)', 'r_sdss': 'r (SDSS)', 'i_sdss': 'i (SDSS)',
    'z_sdss': 'z (SDSS)', 'q_sdss': 'Clear/Q', 'open': 'Clear',
    'gm_cont': 'Guide/Cont'
}

def parse_filter_logic(filter_str):
    if not isinstance(filter_str, str): return np.nan, False
    tokens = filter_str.lower().replace("\n", " ").strip().split()
    is_calib_wheel = False
    real_filter = "Unknown"
    
    if 'cb' in tokens:
        is_calib_wheel = True
        other_tokens = [t for t in tokens if 'cb' not in t]
        real_filter = other_tokens[0] if other_tokens else "Block"
    else:
        other_tokens = [t for t in tokens if 'dia' not in t and 'open' not in t and 'cb' not in t]
        real_filter = other_tokens[0] if other_tokens else "Clear"
    
    clean_key = real_filter.replace("_", "")
    return FILTER_MAP.get(clean_key, clean_key.upper()), is_calib_wheel

def is_calibration_filename(filename):
    f = filename.lower()
    if f.startswith("pan_"): return True
    if "bias" in f or "dark" in f or "flat" in f or "init" in f or "focus" in f: return True
    return False

def extract_file_meta(filename):
    base = os.path.basename(filename)
    match = re.match(r"^(\d{2})(.+)_nightlog", base)
    if match:
        return match.group(1), match.group(2)
    return "Unk", "Unk"

def parse_pdf_logs(pdf_files):
    all_rows = []
    print(f"📂 Procesando {len(pdf_files)} nightlogs...")

    for pdf_file in pdf_files:
        day_id, config_id = extract_file_meta(pdf_file)
        
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if not row or row[0] == "Filename" or row[0] is None: continue
                            try:
                                filename = row[0]
                                raw_filter = row[2]
                                nsamp_raw = row[6]
                                skiprow_raw = row[7]
                                date_obs = row[8]
                                exptime_raw = row[9]
                                
                                try: exptime = float(exptime_raw)
                                except: exptime = 0.0
                                try: nsamp = int(float(nsamp_raw))
                                except: nsamp = 1
                                try: skiprow = float(skiprow_raw)
                                except: skiprow = 0
                                
                                if raw_filter == "N/A" or raw_filter is None: raw_filter = np.nan
                                
                                all_rows.append({
                                    'filename': filename,
                                    'raw_filter': raw_filter,
                                    'exptime': exptime,
                                    'nsamp': nsamp,
                                    'skiprow': skiprow,
                                    'date_obs': date_obs,
                                    'night_day': day_id,
                                    'config': config_id
                                })
                            except: continue
        except: continue

    df = pd.DataFrame(all_rows)
    
    # 1. Imputación
    df['raw_filter'] = df['raw_filter'].fillna(method='ffill').fillna(method='bfill')
    
    # 2. Conversión a Hora Local
    # Convertimos strings a datetime UTC, coaccionando errores
    df['dt_utc'] = pd.to_datetime(df['date_obs'], errors='coerce')
    # Sumamos offset (Chile UTC-4)
    df['dt_local'] = df['dt_utc'] + timedelta(hours=UTC_OFFSET)
    
    final_filters, final_types = [], []
    for idx, row in df.iterrows():
        filt_name, is_calib_wheel = parse_filter_logic(str(row['raw_filter']))
        if is_calibration_filename(row['filename']) or is_calib_wheel:
            otype = 'CALIB'
        else:
            otype = 'SCIENCE'
        final_filters.append(filt_name)
        final_types.append(otype)
        
    df['filter'] = final_filters
    df['type'] = final_types
    
    is_roi = df['skiprow'] > 0
    readtime_unit = np.where(is_roi, READ_TIME_ROI, READ_TIME_FULL)
    df['readtime_total'] = readtime_unit * df['nsamp']
    df['overhead'] = df['readtime_total'] + OVERHEAD_FILE
    
    return df

def analyze_campaign(df):
    if df.empty:
        print("❌ No hay datos.")
        return

    # ==========================================
    # 1. ANÁLISIS POR NOCHE (Consolidado)
    # ==========================================
    print("\n" + "="*90)
    print("🌙 TABLA 1: RESUMEN OPERATIVO POR NOCHE (Hora Local UTC-4)")
    print("="*90)
    print(f"{'Día':<4} {'Imgs':<6} {'Sci(h)':<8} {'Cal(h)':<8} {'Span(h)':<8} {'Point(h)':<8} {'Lost(h)':<8} {'Fin(Local)'}")
    
    nights = df.groupby('night_day')
    
    # Acumuladores Globales
    acc = {'sci':0, 'cal':0, 'point':0, 'weather':0}
    
    for day, group in nights:
        t_sci_exp = group[group['type']=='SCIENCE']['exptime'].sum() / 3600
        t_cal_tot = (group[group['type']=='CALIB']['exptime'].sum()) / 3600
        t_active_sum = (group['exptime'].sum() + group['overhead'].sum()) / 3600
        
        weather_h, pointing_h, span_h = 0, 0, 0
        last_hour_local = -1
        
        # Calcular SPAN usando Hora Local
        valid_dates = group['dt_local'].dropna()
        if not valid_dates.empty:
            start_t = valid_dates.min()
            end_t = valid_dates.max()
            
            # Span real + duración última foto
            span_h = (end_t - start_t).total_seconds() / 3600
            span_h += (group.iloc[-1]['exptime'] + group.iloc[-1]['overhead'])/3600
            
            pointing_h = max(0, span_h - t_active_sum)
            
            # Lógica Clima (Hora Local)
            # Si terminamos antes de las 03:00 AM y trabajamos poco (<5h)
            last_hour_local = end_t.hour
            if last_hour_local < CRITICAL_END_HOUR_LOCAL and span_h < 5.0:
                weather_h = max(0, NIGHT_LENGTH_STD - span_h)
                status_char = "⚠️"
            else:
                status_char = "✅"
        else:
            span_h = t_active_sum * 1.1 # Fallback
            pointing_h = span_h - t_active_sum
        
        # Acumular
        acc['sci'] += t_sci_exp
        acc['cal'] += t_cal_tot
        acc['point'] += (pointing_h + (group['overhead'].sum()/3600)) # Pointing + Readout
        acc['weather'] += weather_h
        
        print(f"{day:<4} {len(group):<6} {t_sci_exp:.2f}     {t_cal_tot:.2f}     {span_h:.2f}     {pointing_h:.2f}     {weather_h:.2f}     {last_hour_local:02d}:xx {status_char}")

    # ==========================================
    # 2. ANÁLISIS POR CONFIGURACIÓN (GLOBAL)
    # ==========================================
    print("\n" + "="*80)
    print("⚡ TABLA 2: RENDIMIENTO POR CONFIGURACIÓN (GLOBAL)")
    print("="*80)
    print(f"{'Config':<10} {'Desc':<10} {'Imgs':<6} {'Sci(h)':<8} {'% Sci'}")
    
    configs = df.groupby('config')
    for conf_name, group in configs:
        desc = "Unk"
        if "hh3" in conf_name: desc = "30k e-"
        elif "d7" in conf_name: desc = "60k e-"
        elif "M2" in conf_name: desc = "5k e-"
        
        t_sci = group[group['type']=='SCIENCE']['exptime'].sum() / 3600
        pct = (t_sci / acc['sci'] * 100) if acc['sci'] > 0 else 0
        
        print(f"{conf_name:<10} {desc:<10} {len(group):<6} {t_sci:.2f}     {pct:.1f}%")

    # ==========================================
    # 3. ANÁLISIS POR NOCHE + CONFIGURACIÓN
    # ==========================================
    print("\n" + "="*80)
    print("📅 TABLA 3: DETALLE NOCHE x CONFIGURACIÓN")
    print("="*80)
    print(f"{'Noche-Conf':<15} {'Imgs':<6} {'Sci(h)':<8} {'Filtros Top'}")
    
    # Agrupar por ambas columnas
    night_configs = df.groupby(['night_day', 'config'])
    
    for (day, conf), group in night_configs:
        label = f"{day}-{conf}"
        t_sci = group[group['type']=='SCIENCE']['exptime'].sum() / 3600
        
        # Filtros usados
        filts = group[group['type']=='SCIENCE']['filter'].value_counts().head(2).index.tolist()
        filt_str = ", ".join(filts) if filts else "-"
        
        print(f"{label:<15} {len(group):<6} {t_sci:.2f}     {filt_str}")

    # ==========================================
    # 4. ESTADÍSTICAS DETALLADAS (EXTRAS)
    # ==========================================
    print("\n" + "="*80)
    print("🔬 ESTADÍSTICAS DETALLADAS DE CIENCIA")
    print("="*80)
    
    df_sci = df[df['type'] == 'SCIENCE']
    
    if not df_sci.empty:
        # A. CONTEO FILTROS
        print("\n[A] IMÁGENES POR FILTRO:")
        print(df_sci['filter'].value_counts())
        
        # B. TIEMPO POR FILTRO
        print("\n[B] TIEMPO EXPOSICIÓN (HORAS) POR FILTRO:")
        print((df_sci.groupby('filter')['exptime'].sum() / 3600).sort_values(ascending=False))
        
        # C. MODOS NSAMP (SKIPPER)
        print("\n[C] USO DE MODO SKIPPER (NSAMP):")
        # Mostrar conteo y porcentaje
        nsamp_counts = df['nsamp'].value_counts().sort_index()
        for ns, count in nsamp_counts.items():
            print(f"   N={ns:<3}: {count} imágenes")

        # D. USO DE ROI
        roi_counts = df_sci['skiprow'].apply(lambda x: 'ROI/Window' if x > 0 else 'FullFrame').value_counts()
        print("\n[D] GEOMETRÍA DE LECTURA (CIENCIA):")
        print(roi_counts)

        # GRÁFICO DE FILTROS (Extra Solicitado)
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(y='filter', data=df_sci, order=df_sci['filter'].value_counts().index, palette='viridis')
        ax.set_title("Distribución de Imágenes Científicas por Filtro")
        ax.set_xlabel("Cantidad de Imágenes")
        ax.bar_label(ax.containers[0])
        plt.tight_layout()
        plt.savefig("stats_filters_detailed.png")
        print("\n✅ Gráfico guardado: stats_filters_detailed.png")

    # ==========================================
    # 5. TOTALES FINALES
    # ==========================================
    total_campaign = acc['sci'] + acc['cal'] + acc['point'] + acc['weather']
    
    print("\n" + "="*40)
    print("📊 TOTALES FINALES (Sección 5.5)")
    print("="*40)
    print(f"Total Imágenes:      {len(df)}")
    print(f"Ciencia (Shutter):   {acc['sci']:.2f} h ({acc['sci']/total_campaign*100:.1f}%)")
    print(f"Calibración:         {acc['cal']:.2f} h ({acc['cal']/total_campaign*100:.1f}%)")
    print(f"Soporte (Point+Read):{acc['point']:.2f} h ({acc['point']/total_campaign*100:.1f}%)")
    print(f"Clima (Pérdida):     {acc['weather']:.2f} h ({acc['weather']/total_campaign*100:.1f}%)")
    
    # GRÁFICO TORTA
    sns.set_theme(style="whitegrid")
    labels = ['Ciencia', 'Soporte', 'Calibración', 'Clima']
    sizes = [acc['sci'], acc['point'], acc['cal'], acc['weather']]
    colors = ['#3498db', '#f1c40f', '#95a5a6', '#e74c3c']
    
    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title(f"Tiempo Total de Campaña: {total_campaign:.1f} h")
    plt.tight_layout()
    plt.savefig("stats_pie_final_v6.png")
    print("✅ Gráfico guardado: stats_pie_final_v6.png")

if __name__ == "__main__":
    pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
    if pdf_files:
        df = parse_pdf_logs(pdf_files)
        df.to_csv("campaign_data_master.csv", index=False)
        analyze_campaign(df)
    else:
        print("❌ No encontré PDFs.")

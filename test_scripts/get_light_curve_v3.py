from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import matplotlib.pyplot as plt
import glob
import numpy as np
import os

# --- CONFIGURACIÓN ---
# Coordenadas iniciales de los dos objetos a seguir (X_IMAGE, Y_IMAGE)
obj1_pos = (413.0, 44.0)  # Primer target
#obj2_pos = (266.0, 153.0) # Segundo target
match_radius = 15.0  # Radio de coincidencia en píxeles
ellipticity_threshold = 20# 3.5  # Umbral para excluir objetos con alta elipticidad
edge_buffer = 5.0  # Píxeles desde el borde para considerar un objeto "dentro del campo"
catalog_pattern = "comb_*_cat.txt"  # Patrón de los archivos de catálogo

# Estilo para PASP
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (16/1.5, 9/1.5),
    'axes.linewidth': 0.8,
    'xtick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.major.size': 4,
    'ytick.minor.size': 2,
})

# --- LECTURA Y PROCESO ---
lightcurves = {
    0: {"time": [], "mag": [], "err": [], "snr": [], "exptime": [], "basename": [], "awin": [], "bwin": [], "x_image": [], "y_image": []},
    1: {"time": [], "mag": [], "err": [], "snr": [], "exptime": [], "basename": [], "awin": [], "bwin": [], "x_image": [], "y_image": []}
}
obj_positions = {0: list(obj1_pos)}#, 1: list(obj2_pos)}  # Convertir a lista para permitir actualización

# Verificar si hay archivos que coincidan con el patrón
catfiles = sorted(glob.glob(catalog_pattern))
if not catfiles:
    raise FileNotFoundError(f"No se encontraron archivos que coincidan con el patrón '{catalog_pattern}' en el directorio actual.")

for catfile in catfiles:
    # Nombre del fits asociado
    fitsfile = catfile.replace("_cat.txt", ".fits")
    basename = os.path.splitext(os.path.basename(fitsfile))[0]
    
    # Verificar si el archivo FITS existe
    if not os.path.exists(fitsfile):
        print(f"Advertencia: No se encontró el archivo FITS '{fitsfile}'. Saltando este catálogo.")
        continue
    
    # Leer el header para obtener el tiempo, EXPTIME y dimensiones de la imagen
    try:
        with fits.open(fitsfile) as hdul:
            uts = hdul[0].header.get("UTSHUT")
            exptime = hdul[0].header.get("EXPTIME", 1.0)
            naxis1 = hdul[0].header.get("NAXIS1", 1000)  # Ancho de la imagen
            naxis2 = hdul[0].header.get("NAXIS2", 1000)  # Alto de la imagen
    except Exception as e:
        print(f"Advertencia: Error al leer el header de '{fitsfile}': {e}. Saltando este catálogo.")
        continue
    
    if uts is None:
        print(f"Advertencia: No se encontró la keyword 'UTSHUT' en el header de '{fitsfile}'. Saltando este catálogo.")
        continue
    
    # Convertir UTSHUT → formato ISO
    try:
        date_part, time_part = uts.split("T")
        time_part = time_part.replace("-", ":")
        uts_iso = date_part + "T" + time_part
        obstime = Time(uts_iso, format="isot", scale="utc")
        mjd = obstime.mjd
    except ValueError as e:
        print(f"Advertencia: Error al procesar UTSHUT en '{fitsfile}': {e}. Saltando este catálogo.")
        continue
    
    # Leer el catálogo
    try:
        tab = Table.read(catfile, format="ascii")
    except Exception as e:
        print(f"Advertencia: Error al leer el catálogo '{catfile}': {e}. Saltando este catálogo.")
        continue
    
    # Verificar columnas necesarias
    required_columns = ["EXT_NUMBER", "X_IMAGE", "Y_IMAGE", "MAG_WIN", "MAGERR_WIN", "SNR_WIN", "AWIN_IMAGE", "BWIN_IMAGE"]
    if not all(col in tab.colnames for col in required_columns):
        print(f"Advertencia: El catálogo '{catfile}' no contiene todas las columnas necesarias {required_columns}. Saltando este catálogo.")
        continue
    
    # Filtrar objetos con EXT_NUMBER == 1
    ext_mask = tab["EXT_NUMBER"] == 1
    ext_table = tab[ext_mask]
    if len(ext_table) == 0:
        print(f"Advertencia: No hay objetos en EXT_NUMBER 1 para el catálogo '{catfile}'. Saltando este catálogo.")
        continue
    
    # Filtrar por baja elipticidad
    ellipticity_mask = (ext_table["AWIN_IMAGE"] <= ellipticity_threshold) & (ext_table["BWIN_IMAGE"] <= ellipticity_threshold)
    ext_table = ext_table[ellipticity_mask]
    if len(ext_table) == 0:
        print(f"Advertencia: No hay objetos con baja elipticidad en '{catfile}'. Saltando este catálogo.")
        continue
    
    # Filtrar objetos dentro de los límites de la imagen
    valid_mask = (ext_table["X_IMAGE"] > edge_buffer) & \
                 (ext_table["X_IMAGE"] < naxis1 - edge_buffer) & \
                 (ext_table["Y_IMAGE"] > edge_buffer) & \
                 (ext_table["Y_IMAGE"] < naxis2 - edge_buffer)
    ext_table = ext_table[valid_mask]
    if len(ext_table) == 0:
        print(f"Advertencia: No hay objetos válidos dentro de los límites de la imagen en '{catfile}' (bordes: {edge_buffer} píxeles). Saltando este catálogo.")
        continue
    
    # Encontrar los objetos basados en sus posiciones
    for obj_idx, pos in obj_positions.items():
        x_ref, y_ref = pos
        distances = np.sqrt((ext_table["X_IMAGE"] - x_ref)**2 + (ext_table["Y_IMAGE"] - y_ref)**2)
        if len(distances) == 0 or np.min(distances) > match_radius:
            print(f"Advertencia: No se encontró el objeto {obj_idx} en '{catfile}' dentro del radio de {match_radius} píxeles. "
                  f"Coordenadas de referencia: ({x_ref:.2f}, {y_ref:.2f}).")
            continue
        match_idx = np.argmin(distances)
        matched_obj = ext_table[match_idx]
        distance = distances[match_idx]
        
        # Registrar coordenadas y distancia para diagnóstico
        print(f"Objeto {obj_idx} encontrado en '{catfile}': X_IMAGE={matched_obj['X_IMAGE']:.4f}, "
              f"Y_IMAGE={matched_obj['Y_IMAGE']:.4f}, distancia={distance:.4f} píxeles, SNR_WIN={matched_obj['SNR_WIN']:.4f}")
        
        # Almacenar datos
        lightcurves[obj_idx]["time"].append(mjd)
        lightcurves[obj_idx]["mag"].append(matched_obj["MAG_WIN"])
        lightcurves[obj_idx]["err"].append(matched_obj["MAGERR_WIN"])
        lightcurves[obj_idx]["snr"].append(matched_obj["SNR_WIN"])
        lightcurves[obj_idx]["exptime"].append(exptime)
        lightcurves[obj_idx]["basename"].append(basename)
        lightcurves[obj_idx]["awin"].append(matched_obj["AWIN_IMAGE"])
        lightcurves[obj_idx]["bwin"].append(matched_obj["BWIN_IMAGE"])
        lightcurves[obj_idx]["x_image"].append(matched_obj["X_IMAGE"])
        lightcurves[obj_idx]["y_image"].append(matched_obj["Y_IMAGE"])
        
        # Actualizar posición para el próximo catálogo
        obj_positions[obj_idx] = [matched_obj["X_IMAGE"], matched_obj["Y_IMAGE"]]

# --- GUARDAR TABLAS DE CURVAS DE LUZ ---
for obj in lightcurves:
    data = lightcurves[obj]
    if not data["time"]:
        print(f"Error: No se encontraron datos válidos para el objeto {obj} en EXT_NUMBER=1. No se generará tabla.")
        continue
    
    # Crear tabla con MJD, mag, err, snr, x_image, y_image, awin, bwin
    table_data = Table([data["time"], data["mag"], data["err"], data["snr"], 
                        data["x_image"], data["y_image"], data["awin"], data["bwin"], data["basename"]],
                       names=["MJD", "MAG_WIN", "MAGERR_WIN", "SNR_WIN", "X_IMAGE", "Y_IMAGE", "AWIN_IMAGE", "BWIN_IMAGE", "BASENAME"])
    table_data.sort("MJD")
    
    # Usar el primer basename disponible
    basename = table_data["BASENAME"][0]
    output_filename = f"light_curve_{basename}_obj{obj}.txt"
    
    # Seleccionar columnas para la tabla de salida
    filtered_table = table_data[["MJD", "MAG_WIN", "MAGERR_WIN", "SNR_WIN", "X_IMAGE", "Y_IMAGE", "AWIN_IMAGE", "BWIN_IMAGE"]]
    
    # Guardar tabla
    filtered_table.write(output_filename, format="ascii.fixed_width", 
                        formats={"MJD": "%.6f", "MAG_WIN": "%.4f", "MAGERR_WIN": "%.4f", "SNR_WIN": "%.4f", 
                                 "X_IMAGE": "%.4f", "Y_IMAGE": "%.4f", "AWIN_IMAGE": "%.4f", "BWIN_IMAGE": "%.4f"},
                        overwrite=True)

# --- GRAFICADO ---
if any(data["time"] for data in lightcurves.values()):
    fig, ax = plt.subplots()

    markers = ['o', 's']  # Círculo y cuadrado
    colors = ['black', 'darkgray']

    for obj, marker, color in zip(lightcurves.keys(), markers, colors):
        data = lightcurves[obj]
        if not data["time"]:
            continue
        ax.scatter(data["time"], data["mag"], marker=marker, color=color, s=30, label=f'Objeto {obj}')
        for t, m, e in zip(data["time"], data["mag"], data["err"]):
            ax.plot([t, t], [m - e, m + e], color=color, linewidth=0.8)

    ax.invert_yaxis()
    ax.set_xlabel(r'MJD', fontsize=14)
    ax.set_ylabel(r'Magnitude (MAG\_WIN)', fontsize=14)
    ax.set_title('Curva de luz', fontsize=16)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(f'light_curve_{basename}.png', format='png', bbox_inches='tight')
    plt.show()
else:
    print("No se generó gráfico porque no hay datos válidos para los objetos.")

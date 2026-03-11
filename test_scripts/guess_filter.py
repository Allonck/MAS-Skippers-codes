import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import glob
import os
import warnings

# Ignorar advertencias molestas de FITS
warnings.filterwarnings('ignore')

# Buscar archivos en la carpeta actual
files = sorted(glob.glob("reduced/o_AR*.fits"))

print(f"Archivos encontrados: {len(files)}")

if len(files) == 0:
    print("¡ERROR! No hay archivos .fits en esta carpeta.")
    exit()

results = []
file_labels = []

print(f"{'Archivo':<25} | {'Ext':<3} | {'Exp':<5} | {'Fondo (ADU/s)':<10}")
print("-" * 65)

for f in files:
    try:
        with fits.open(f) as hdul:
            # BUSCAR DATOS: Normalmente en Ext 1 para MEF
            target_ext = 1
            data = None
            
            # Verificar si Ext 1 existe y tiene datos
            if len(hdul) > 1 and hdul[1].data is not None:
                data = hdul[1].data
                target_ext = 1
            # Si no, buscar la primera extensión con datos válida
            else:
                for i, hdu in enumerate(hdul):
                    if hdu.data is not None and hdu.data.ndim == 2: # Solo imágenes 2D
                        data = hdu.data
                        target_ext = i
                        break
            
            if data is None:
                print(f"{os.path.basename(f):<25} | ??? | Error: FITS vacío")
                continue

            # LEER EXPTIME (A veces está en Header 0, a veces en Header de Extensión)
            # Prioridad: Header de la extensión de datos, luego Header 0
            header = hdul[target_ext].header
            exptime = header.get('EXPTIME', None)
            
            if exptime is None:
                # Intentar en el primario
                exptime = hdul[0].header.get('EXPTIME', 1.0)
            
            exptime = float(exptime)
            if exptime <= 0: exptime = 1.0 # Evitar división por cero

            # ESTADÍSTICA DEL CENTRO (Recorte)
            h, w = data.shape
            cy, cx = int(h/2), int(w/2)
            # Recorte pequeño de 200x200 para velocidad
            cutout = data[cy-100:cy+100, cx-100:cx+100]
            
            # Sigma clipping para sacar estrellas
            mean, median, std = sigma_clipped_stats(cutout, sigma=3.0)
            
            # Tasa en ADU/s
            rate = median / exptime
            
            print(f"{os.path.basename(f):<25} | {target_ext:<3} | {exptime:<5.1f} | {rate:.2f}")
            results.append(rate)
            file_labels.append(os.path.basename(f))

    except Exception as e:
        print(f"{os.path.basename(f):<25} | ERR | {str(e)}")

# --- GRAFICAR ---
if len(results) > 0:
    plt.figure(figsize=(12, 6))
    x = range(len(results))
    
    # Graficar puntos
    plt.scatter(x, results, c='blue', marker='o', s=50, alpha=0.7)
    
    # Dibujar líneas entre puntos para ver secuencia
    plt.plot(x, results, 'b-', alpha=0.3)
    
    # Etiquetas en ejes
    plt.xlabel('Secuencia de Imagen')
    plt.ylabel('Fondo de Cielo (ADU/s)')
    plt.title('Identificación de Filtros por Nivel de Fondo')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Mostrar nombres de algunos archivos en el eje X para guiarse
    # (Solo mostramos 1 de cada 5 para no saturar)
    if len(file_labels) > 20:
        step = int(len(file_labels)/10)
        plt.xticks(x[::step], file_labels[::step], rotation=45, ha='right', fontsize=8)
    else:
        plt.xticks(x, file_labels, rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    plt.savefig("filter_guess_mef.png")
    print("\nGráfico guardado como 'filter_guess_mef.png'")
    # plt.show() # Descomenta si tienes pantalla gráfica
else:
    print("\nNo se pudieron extraer datos.")

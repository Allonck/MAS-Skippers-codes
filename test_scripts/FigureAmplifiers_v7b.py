import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.colors import LogNorm  # <--- IMPORTANTE

#Ahora no sólo muestra solo el primer canal en las 16 subimágenes!
# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
FITS_FILE = '/home/allon/Escritorio/new_ptcs/bsw/Lab/07_07_2025/16ch_test_0002_sort.fits'

PIXEL_SCALE_ARCSEC = 0.2546
SCALEBAR_VALUE = 10 

SATURN_CENTER_Y = 316
SATURN_CENTER_X = 132

# Geometría "Tira Vertical" (16:1 aprox)
CROP_WIDTH = 20
CROP_HEIGHT = 95

CMAP = 'viridis' #"gray"

# ==========================================
# 2. PROCESAMIENTO
# ==========================================

OFFSET_PER_CHANNEL = 15

try:
    hdul = fits.open(FITS_FILE)
    print(f"Archivo cargado. Extensiones encontradas: {len(hdul)}")
except Exception as e:
    print(f"Error al abrir FITS: {e}")
    exit()

# Creamos una lista para guardar los recortes de cada canal
crops = []

for i in range(1, 17):  # Canales 1 al 16 (Extensiones 1-16)
    try:
        data_ch = hdul[i].data.astype(float)
        
        # Calculamos el desplazamiento: 0 para el primer canal, 15 para el segundo...
        # Esto asume que el rayo "avanza" 15 píxeles en X por cada extensión
        current_offset = (i - 1) * OFFSET_PER_CHANNEL
        
        # Ajustamos el centro de recorte en X (o Y si el sensor es vertical)
        center_x_shifted = SATURN_CENTER_X + current_offset
        
        y1 = int(max(0, SATURN_CENTER_Y - CROP_HEIGHT // 2))
        y2 = int(min(data_ch.shape[0], SATURN_CENTER_Y + CROP_HEIGHT // 2))
        x1 = int(max(0, center_x_shifted - CROP_WIDTH // 2))
        x2 = int(min(data_ch.shape[1], center_x_shifted + CROP_WIDTH // 2))
        
        # Validación de límites para evitar recortes vacíos
        if x1 >= data_ch.shape[1] or x2 <= 0:
            # Si el offset saca el cuadro del sensor, ponemos ceros o el borde
            crops.append(np.zeros((CROP_HEIGHT, CROP_WIDTH)))
        else:
            crops.append(data_ch[y1:y2, x1:x2])
            
    except IndexError:
        print(f"Extensión {i} no encontrada.")

if not crops:
    print("Error: No se pudieron procesar las extensiones.")
    exit()

# Estadísticas globales (usando todos los canales para escala lineal)
all_data = np.concatenate([c.flatten() for c in crops])
vmin = np.percentile(all_data, 2.0)
vmax = np.percentile(all_data, 99.0) # Subí a 99 para no perder el brillo del rayo

# --- LÓGICA LOGARÍTMICA SEGURA ---
# 1. Filtramos solo valores > 0 para calcular estadísticas
#    (Evitamos que los píxeles muertos de -1e8 rompan el logaritmo)
#valid_pixels = saturn_crop[saturn_crop > 0]

#if len(valid_pixels) == 0:
#    print("Error: No hay píxeles positivos en el recorte.")
#    exit()

# 2. Calculamos vmin/vmax sobre los datos válidos
#    vmin: percentil 5 para levantar el fondo y que el negro sea negro.
#    vmax: percentil 99.9 para no saturar demasiado el núcleo.
#vmin = np.percentile(valid_pixels, 5.0)
#vmax = np.percentile(valid_pixels, 99.9)

#print(f"Escala Logarítmica: vmin={vmin:.2f}, vmax={vmax:.2f}")

# ------------------------------------------------------------------
# Calculamos vmin/vmax para escala lineal
# Usamos percentiles para evitar que hot pixels saturen la imagen
#vmin = np.percentile(saturn_crop, 2.0)
#vmax = np.percentile(saturn_crop, 98.0)


# ==========================================
# 3. GRAFICADO
# ==========================================
fig = plt.figure(figsize=(24, 7)) 

gs = gridspec.GridSpec(2, 16, height_ratios=[35, 1], width_ratios=[1]*16)

gs.update(wspace=0.08, hspace=0.02, bottom=0.08, top=0.98, left=0.02, right=0.98)

main_axes = []

for i in range(16):
    ax = plt.subplot(gs[0, i])
    
    # --- CAMBIO AQUÍ: norm=LogNorm(...) ---
    #im = ax.imshow(saturn_crop, 
    #               origin='lower', 
    #               cmap=CMAP, 
    #               norm=LogNorm(vmin=vmin, vmax=vmax), # Escala Log
    #               aspect='equal')

    # CAMBIO: Se elimina LogNorm y se usan vmin/vmax directos
    #im = ax.imshow(saturn_crop, 
    #               origin='lower', 
    #               cmap=CMAP, 
    #               vmin=vmin, 
    #               vmax=vmax,
    #               aspect='equal')
    
    # IMPORTANTE: Usamos crops[i] para mostrar el canal correspondiente
    im = ax.imshow(crops[i], 
                   origin='lower', 
                   cmap=CMAP, 
                   vmin=vmin, 
                   vmax=vmax,
                   aspect='equal')

    ax.set_xticks([])
    ax.set_yticks([])
    
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(1.0)
        
    main_axes.append(ax)

hdul.close() #Aquí?

# Barra de Escala
#pixel_length = SCALEBAR_VALUE / PIXEL_SCALE_ARCSEC
#fontprops = fm.FontProperties(size=11, weight='bold')
#scalebar = AnchoredSizeBar(
#    main_axes[-1].transData, pixel_length, f'{SCALEBAR_VALUE}"', 
#    loc='lower right', pad=0.4, color='red', frameon=False, 
#    size_vertical=2, fontproperties=fontprops
#)
#main_axes[-1].add_artist(scalebar)

# Definimos el tamaño de la barra directamente en píxeles
SCALEBAR_PIXELS = 5

# Barra de Escala en Píxeles
fontprops = fm.FontProperties(size=10, weight='bold')
scalebar = AnchoredSizeBar(
    main_axes[-1].transData, 
    SCALEBAR_PIXELS,           # Tamaño en unidades de los datos (píxeles)
    f'{SCALEBAR_PIXELS} px',   # Etiqueta de laboratorio
    loc='lower right', 
    pad=0.4, 
    color='white', 
    frameon=False, 
    size_vertical=1,           # Grosor de la barra en píxeles
    fontproperties=fontprops
)
main_axes[-1].add_artist(scalebar)

# ==========================================
# 4. BARRA DE COLOR LOGARÍTMICA
# ==========================================

#----------------------------------------
#cax = plt.subplot(gs[1, :])
# format='%.0e' pone notación científica bonita (e.g., 1e5) en la barra
#cbar = plt.colorbar(im, cax=cax, orientation='horizontal', format='%.0e')
#cbar.set_label('Intensidad (ADU) [Escala Log]', color='black', fontsize=10)
#cbar.ax.tick_params(labelsize=9, color='black', labelcolor='black')
# Forzamos minerticks para que se vea la graduación logarítmica
#cbar.ax.minorticks_on()
# -----------------------------------------

# ==========================================
# 4. BARRA DE COLOR LINEAL
# ==========================================
cax = plt.subplot(gs[1, :])
# Se elimina el formato científico '%.0e' para lectura natural, o usa '%.1f'
cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label('Cuentas (ADU) [Escala Lineal]', color='white', fontsize=10)
cbar.ax.tick_params(labelsize=9, color='black', labelcolor='black')
# En escala lineal, minorticks suelen ser automáticos o innecesarios

output_file = 'Bias_16ch_v7'
plt.savefig(output_file + ".png", dpi=300, bbox_inches='tight', facecolor='white',format="png")
plt.savefig(output_file + ".pdf", dpi=300, bbox_inches='tight',format="pdf")
print(f"✅ Imagen generada: {output_file}")
plt.show()

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

# --- CONFIGURACIÓN ---
FILENAME = "DeepLAE.fits"
OUTPUT_FILENAME = "DeepLAE_WCS_Structured.fits"

# DATOS DE REFERENCIA
REF_PIXEL = [104, 486]            # [X, Y]
REF_COORD = [334.29312, 0.11317]  # [RA, DEC]
SCALE_DEG = 0.2546 / 3600.0       # grados/pixel
ROTATION_DEG = 0.0

def create_simple_wcs(ref_pix, ref_coord, scale, rot_angle=0):
    w = WCS(naxis=2)
    w.wcs.crpix = ref_pix
    w.wcs.crval = ref_coord
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
    # Matriz de rotación/escala lineal
    cos_rot = np.cos(np.deg2rad(rot_angle))
    sin_rot = np.sin(np.deg2rad(rot_angle))
    w.wcs.pc = [[cos_rot, -sin_rot], 
                [sin_rot, cos_rot]]
    w.wcs.cdelt = [scale, scale]
    return w

# --- PROCESO ---
print(f"Procesando {FILENAME} manteniendo estructura MEF...")

with fits.open(FILENAME) as hdul:
    # 1. COPIA LITERAL DE HEADERS (Como pediste)
    primary_hdr = hdul[0].header.copy()
    image_hdr = hdul[1].header.copy()
    
    # Extraemos la data de la extensión 1
    image_data = hdul[1].data

    # 2. LIMPIEZA DEL HEADER DE LA IMAGEN (HDU 1)
    # Solo borramos las WCS keywords del header de la imagen para evitar conflictos
    wcs_keywords = [
        'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 
        'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
        'CDELT1', 'CDELT2', 'CRVAL1', 'CRVAL2', 
        'CRPIX1', 'CRPIX2', 'CTYPE1', 'CTYPE2',
        'RADESYS', 'LONPOLE', 'LATPOLE', 'WCSAXES'
    ]
    
    for key in list(image_hdr.keys()):
        # Borrar distorsiones SIP/PV y WCS antiguo
        if any(key.startswith(k) for k in ['A_', 'B_', 'AP_', 'BP_', 'PV']):
            del image_hdr[key]
        elif key in wcs_keywords:
            del image_hdr[key]

    # 3. GENERAR E INYECTAR NUEVO WCS
    new_wcs = create_simple_wcs(REF_PIXEL, REF_COORD, SCALE_DEG, ROTATION_DEG)
    # Actualizamos image_hdr con el nuevo WCS
    image_hdr.update(new_wcs.to_header())

    # 4. RECONSTRUIR ESTRUCTURA FITS (HDU 0 + HDU 1)
    
    # HDU 0: PrimaryHDU con header original y sin datos
    hdu0 = fits.PrimaryHDU(header=primary_hdr)
    
    # HDU 1: ImageHDU con datos y header modificado
    hdu1 = fits.ImageHDU(data=image_data, header=image_hdr)
    
    # Lista de HDUs
    hdul_new = fits.HDUList([hdu0, hdu1])
    
    # 5. GUARDAR
    hdul_new.writeto(OUTPUT_FILENAME, overwrite=True)

print(f"✅ Archivo guardado: {OUTPUT_FILENAME}")
print("   - Estructura HDU 0 / HDU 1 mantenida.")
print(f"   - WCS actualizado en HDU 1 (Pixel {REF_PIXEL}).")

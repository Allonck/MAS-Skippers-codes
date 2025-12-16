import glob
from astropy.io import fits

# --- CONFIGURACIÓN ---
# Patrón de búsqueda de tus imágenes (ej: todas las fits en la carpeta actual)
INPUT_PATTERN = "DeepLAE_copy.fits"

# OFFSETS DE CALIBRACIÓN (Diferencia = Real - Header)
# Pon aquí los valores exactos que calculaste con la imagen DeepLAE
D_RA  = -0.028832 # Grados
D_DEC = 0.003956  # Grados

def batch_correction():
    files = sorted(glob.glob(INPUT_PATTERN))
    print(f"🔍 Encontrados {len(files)} archivos para corregir.")
    print(f"   Aplicando Shift: dRA={D_RA}, dDec={D_DEC}")

    for filename in files:
        try:
            # Abrimos en modo 'update' para modificar in-situ
            with fits.open(filename, mode='update') as hdul:
                # Iterar sobre extensiones que tengan imagen (usualmente 1 en tus MEF)
                for ext in range(len(hdul)):
                    if hdul[ext].data is None: continue
                    
                    header = hdul[ext].header
                    
                    # Verificar si tiene WCS para corregir
                    if 'CRVAL1' in header and 'CRVAL2' in header:
                        old_ra = header['CRVAL1']
                        old_dec = header['CRVAL2']
                        
                        # APLICAR OFFSET
                        new_ra = old_ra + D_RA
                        new_dec = old_dec + D_DEC
                        
                        header['CRVAL1'] = new_ra
                        header['CRVAL2'] = new_dec
                        
                        # Opcional: Si necesitas corregir el "espejo" (Flip X) en el WCS
                        # Descomenta esto si tu RA aumenta hacia la DERECHA en la imagen
                        # header['PC1_1'] = 1.0  # Matriz identidad (Espejo en Astronomía)
                        # header['PC2_2'] = 1.0
                        
                        header['HISTORY'] = f"WCS Shifted: dRA={D_RA:.5f}, dDec={D_DEC:.5f}"
                
                # El 'with' se encarga de guardar y cerrar (flush)
                print(f"✅ Corregido: {filename}")
                
        except Exception as e:
            print(f"❌ Error en {filename}: {e}")

if __name__ == "__main__":
    batch_correction()

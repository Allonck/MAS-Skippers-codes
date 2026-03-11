from astropy.io import fits
import glob
import os

# --- CONFIGURACIÓN MANUAL (¡EDITA ESTO CADA VEZ!) ---

# 1. Patrón de archivos a modificar (Usa * como comodín)
# Ejemplos: "testSAT_02*.fits", "testSAT_020[0-4].fits"
FILE_PATTERN = "ARSAT1_0*.fits" 

# 2. Filtro a asignar (r, ov, b, etc.)
# FILTER1 siempre será 'dia' según tu instrucción.
TARGET_FILTER = "ov" 

# ----------------------------------------------------

def update_fits_headers():
    # Buscar archivos
    files = sorted(glob.glob(FILE_PATTERN))
    
    if not files:
        print(f"ERROR: No encontré archivos con el patrón '{FILE_PATTERN}'")
        return

    print(f"--- Actualizando {len(files)} archivos al filtro '{TARGET_FILTER}' ---")
    
    for f in files:
        try:
            # Abrir en modo 'update' para guardar cambios
            with fits.open(f, mode='update') as hdul:
                
                # Modificamos el Header Primario (0) y si es MEF, también el de Ciencia (1)
                # para asegurar compatibilidad con cualquier software.
                for ext in [0, 1]:
                    if ext < len(hdul):
                        header = hdul[ext].header
                        
                        # Asignar valores
                        header['FILTER1'] = 'dia'
                        header['FILTER2'] = TARGET_FILTER
                        header['FILTERS'] = f"dia {TARGET_FILTER}"
                        
                        # Opcional: Agregar comentario para trazabilidad
                        header['HISTORY'] = f"Filtro recuperado forensemente: {TARGET_FILTER}"

                # Guardar cambios (flush automático al cerrar con 'update')
                print(f"OK: {os.path.basename(f)} -> dia {TARGET_FILTER}")
                
        except Exception as e:
            print(f"ERROR en {os.path.basename(f)}: {e}")

if __name__ == "__main__":
    # Confirmación de seguridad
    print(f"Vas a marcar los archivos '{FILE_PATTERN}' como '{TARGET_FILTER}'.")
    confirm = input("¿Continuar? (s/n): ")
    if confirm.lower() == 's':
        update_fits_headers()
    else:
        print("Cancelado.")

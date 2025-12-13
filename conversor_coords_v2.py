import sys
import readline # <--- ESTO ARREGLA LAS FLECHAS Y EL HISTORIAL EN LINUX
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
import warnings

warnings.filterwarnings("ignore")

def identificar_y_convertir(texto_input):
    texto_input = texto_input.strip()
    
    # ------------------------------------------------------
    # CASO 1: Intentar detectar un PAR de coordenadas (RA y DEC)
    # ------------------------------------------------------
    try:
        # Asumimos el estándar astronómico: RA en horas, DEC en grados
        c = SkyCoord(texto_input, unit=(u.hourangle, u.deg))
        imprimir_resultado_skycoord(c, "Par detectado: RA(h) DEC(deg)")
        return
    except (ValueError, u.UnitsError):
        pass

    try:
        # Intento alternativo: Ambos en grados
        c = SkyCoord(texto_input, unit=(u.deg, u.deg))
        imprimir_resultado_skycoord(c, "Par detectado: RA(deg) DEC(deg)")
        return
    except (ValueError, u.UnitsError):
        pass
    
    # ------------------------------------------------------
    # CASO 2: Intentar detectar UN SOLO VALOR
    # ------------------------------------------------------
    
    # Intento A: El string YA TIENE unidades (ej: "10h30m" o "45d")
    try:
        a = Angle(texto_input)
        imprimir_resultado_angulo(a, "Unidad explícita en input")
        return
    except (ValueError, u.UnitsError):
        pass

    # Intento B: No tiene unidades, ASUMIR HORAS (Formato HMS: 14 40 55)
    # Esto soluciona tu error con "14 40 55"
    try:
        a = Angle(texto_input, unit=u.hour)
        imprimir_resultado_angulo(a, "Interpretado como HORAS (HMS)")
        return
    except (ValueError, u.UnitsError):
        pass

    # Intento C: No tiene unidades, ASUMIR GRADOS (Decimal o DMS)
    try:
        a = Angle(texto_input, unit=u.deg)
        imprimir_resultado_angulo(a, "Interpretado como GRADOS")
        return
    except (ValueError, u.UnitsError):
        pass

    # Si nada funciona:
    print(f"\n[ERROR] No se pudo interpretar '{texto_input}'.")
    print("Intenta ser explícito, ej: '14h40m55s' o '14:40:55'")

def imprimir_resultado_skycoord(coord, nota):
    print(f"\n✅ ÉXITO ({nota})")
    print(f"{'TIPO':<10} | {'DECIMAL (deg)':<20} | {'SEXAGESIMAL':<25}")
    print("-" * 65)
    
    ra_hms = coord.ra.to_string(unit=u.hour, sep=':', precision=3, pad=True)
    print(f"{'RA':<10} | {coord.ra.degree:<20.5f} | {ra_hms} (h:m:s)")
    
    dec_dms = coord.dec.to_string(unit=u.deg, sep=':', precision=2, alwayssign=True, pad=True)
    print(f"{'DEC':<10} | {coord.dec.degree:<20.5f} | {dec_dms} (d:m:s)")

def imprimir_resultado_angulo(angle, nota):
    print(f"\n✅ ÉXITO ({nota})")
    print("-" * 60)
    
    # Mostramos ambas interpretaciones posibles para un solo valor
    print(f"Si es RA (Horas):    {angle.to_string(unit=u.hour, sep=':', precision=3)}  (h:m:s)")
    print(f"Si es DEC (Grados):  {angle.to_string(unit=u.deg, sep=':', precision=2)}  (d:m:s)")
    print(f"Valor Decimal:       {angle.degree:.6f} deg")
    print(f"Radianes:            {angle.radian:.6f} rad")

def main():
    print("=== CONVERSOR ASTROPY MEJORADO ===")
    print("Soporta historial (flechas) y formatos ambigüos.")
    print("Escribe 'q' o 'exit' para salir.\n")

    while True:
        try:
            # El input ahora usa readline automáticamente
            inp = input("\n>> Ingresa coordenadas: ")
            
            if inp.strip().lower() in ['q', 'exit', 'salir', 'quit', 'exit()']:
                print("Cerrando...")
                break
            if not inp.strip():
                continue
            
            identificar_y_convertir(inp)
            
        except KeyboardInterrupt:
            print("\nSaliendo...")
            break
        except Exception as e:
            print(f"Error de sistema: {e}")

if __name__ == "__main__":
    main()

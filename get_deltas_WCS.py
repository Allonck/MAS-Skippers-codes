from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

# Cargar WCS Malo (Original)
# OJO: Como tu original no tenía PC matrix (-1), asumía RA creciendo a la derecha.
# Eso también es parte del error que vamos a corregir con el Delta.
hdr_bad = fits.getheader("DeepLAE.fits", ext=1) # O ext 0 si moviste cosas
w_bad = WCS(hdr_bad)

# Cargar WCS Bueno (Corregido)
hdr_good = fits.getheader("DeepLAE_WCS_Structured.fits", ext=1)
w_good = WCS(hdr_good)

# Calcular coordenadas en el CENTRO FÍSICO de la imagen
# Usamos el centro del chip (CRPIX original) para comparar
center_pix = [hdr_bad['NAXIS1']/2, hdr_bad['NAXIS2']/2] # [255.5, 512]

# ¿Qué coordenadas ve cada WCS en ese pixel?
coords_bad = w_bad.pixel_to_world(*center_pix)
coords_good = w_good.pixel_to_world(*center_pix)

# Calcular Delta (Bueno - Malo)
d_ra = coords_good.ra.deg - coords_bad.ra.deg
d_dec = coords_good.dec.deg - coords_bad.dec.deg

print(f"--- CONSTANTES DE CALIBRACIÓN ---")
print(f"DELTA_RA  = {d_ra:.6f}")
print(f"DELTA_DEC = {d_dec:.6f}")
print("---------------------------------")

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import kurtosis, skew
from scipy import fftpack

# --- CONFIGURACIÓN ---
files = {
    2: "/home/allon/Descargas/lae/hhM2/reduced/deep/electron/DeepLAE.fits",
    3: "/home/allon/Descargas/lae/hhM2/redChebDeg3n3/deep/electron/DeepLAE3.fits",
    4: "/home/allon/Descargas/lae/hhM2/redChebDeg4n4/deep/electron/DeepLAE4.fits",
    5: "/home/allon/Descargas/lae/hhM2/redChebDeg5n5/deep/electron/DeepLAEbad.fits"
}

REF_DEG = 2

def get_data(filename):
    with fits.open(filename) as hdul:
        if hdul[0].data is not None: return hdul[0].data.astype(float)
        if len(hdul) > 1 and hdul[1].data is not None: return hdul[1].data.astype(float)
    return None

def get_radial_psd(image):
    """Calcula el Espectro de Potencia Promediado Azimutalmente"""
    F1 = fftpack.fft2(image)
    F2 = fftpack.fftshift(F1)
    psd2D = np.abs(F2)**2
    
    h, w = image.shape
    y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
    r = np.hypot(x, y).astype(int)
    
    # Promedio radial
    tbin = np.bincount(r.ravel(), psd2D.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def run_analysis():
    print(f"{'Deg':<4} | {'RMS (%)':<8} | {'P-V (%)':<8} | {'Kurtosis':<8} | {'Skewness':<8}")
    print("-" * 50)

    try:
        ref_data = get_data(files[REF_DEG])
        ref_data[ref_data == 0] = np.nan
        
        # Guardamos perfiles para plotear
        psd_dict = {}
        hist_dict = {}

        plt.figure(figsize=(12, 10))

        for deg in [3, 4, 5]:
            data = get_data(files[deg])
            if data is None: continue

            # 1. Calcular Residuo Relativo (Ratio - 1.0)
            # Restamos 1.0 para centrar en 0 (ej: 1.05 -> 0.05 o 5%)
            ratio = (data / ref_data) - 1.0
            
            # Limpiar NaNs
            valid_pixels = ratio[~np.isnan(ratio)]
            
            # --- MÉTRICAS ESCALARES ---
            rms = np.std(valid_pixels) * 100
            pv = (np.max(valid_pixels) - np.min(valid_pixels)) * 100
            kurt = kurtosis(valid_pixels)
            skw = skew(valid_pixels)
            
            print(f"{deg:<4} | {rms:<8.3f} | {pv:<8.1f} | {kurt:<8.2f} | {skw:<8.2f}")

            # --- PREPARAR PLOTS ---
            
            # A) Histograma
            plt.subplot(2, 2, 1)
            plt.hist(valid_pixels.ravel(), bins=100, alpha=0.5, label=f'Deg {deg}', density=True, range=(-0.2, 0.2))
            
            # B) Espectro de Potencia (PSD)
            # Usamos el residuo puro para ver la frecuencia de la "dona"
            psd = get_radial_psd(np.nan_to_num(ratio))
            psd_dict[deg] = psd

        # Plot A: Histograma
        plt.subplot(2, 2, 1)
        plt.title("Distribución de Errores (Histograma)")
        plt.xlabel("Error Relativo (Fracción)")
        plt.ylabel("Densidad")
        plt.legend()
        plt.grid(alpha=0.3)

        # Plot B: Espectro de Potencia
        plt.subplot(2, 1, 2)
        for deg, psd in psd_dict.items():
            plt.loglog(psd, label=f'Deg {deg}', linewidth=2)
        
        plt.title("Espectro de Potencia Espacial (Firma de la Oscilación)")
        plt.xlabel("Frecuencia Espacial (k)")
        plt.ylabel("Potencia (Log)")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_analysis()

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.visualization import ZScaleInterval, LinearStretch, AsinhStretch, ImageNormalize
from astropy.table import Table

from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.utils import calc_total_error
from photutils.background import Background2D, MedianBackground

# Set Matplotlib style for publication (PASP-like)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['font.size'] = 12 / 2
plt.rcParams['axes.labelsize'] = 14 / 2
plt.rcParams['axes.titlesize'] = 14 / 2
plt.rcParams['xtick.labelsize'] = 12 / 2
plt.rcParams['ytick.labelsize'] = 12 / 2
plt.rcParams['legend.fontsize'] = 12 / 2
plt.rcParams['figure.dpi'] = 300 / 2
plt.rcParams['savefig.bbox'] = 'tight'

def perform_photometry(file_path, aperture_radius=15.0, threshold=5.0, fwhm=3.0, box_size=(50, 50), zeropoint=0.0):
    """
    Detecta sources y realiza photometry en todas las extensiones 2D de un archivo FITS MEF.

    Args:
        file_path (str): Ruta al archivo FITS MEF.
        aperture_radius (float): Radio de apertura en píxeles.
        threshold (float): Umbral para detección (x bkg_std).
        fwhm (float): FWHM en píxeles.
        box_size (tuple): Tamaño de caja para background 2D.
        zeropoint (float): Zeropoint for magnitudes.

    Returns:
        list of Table: Lista de catálogos (uno por extensión HDU con data 2D).
    """
    all_tables = []  # Lista para tablas por HDU

    # Load FITS
    with fits.open(file_path) as hdul:
        for i in range(1, len(hdul)):  # Loop por HDUs 1 a N
            if hdul[i].data is None or len(hdul[i].data.shape) != 2:
                print(f"HDU {i}: Skipping - no 2D data.")
                continue

            data = hdul[i].data.astype(float)
            header = hdul[i].header
            print(f"HDU {i}: Processing shape: {data.shape}")

            # Background subtraction
            bkg = Background2D(data, box_size=box_size, filter_size=(3, 3), sigma_clip=SigmaClip(sigma=3.0))
            bkg_data = bkg.background
            data_sub = data - bkg_data

            # Noise estimation
            _, bkg_median, bkg_std = sigma_clipped_stats(bkg_data, sigma=3.0)
            print(f"HDU {i}: Background median: {bkg_median:.2f}, STD: {bkg_std:.2f}")

            # Source detection
            daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * bkg_std)
            sources = daofind(data_sub)
            if sources is None or len(sources) == 0:
                print(f"HDU {i}: No sources detected; skipping HDU.")
                continue

            positions = np.transpose((sources['xcentroid'], sources['ycentroid']))

            # Aperture photometry with annuli for local sky
            apertures = CircularAperture(positions, r=aperture_radius)
            annuli = CircularAnnulus(positions, r_in=aperture_radius * 1.5,
                                     r_out=aperture_radius * 2.5)  # Annulus for sky
            hdu_table = aperture_photometry(data_sub, apertures, error=None)  # First without error
            sky_table = aperture_photometry(data_sub, annuli, error=None)  # Sky annuli

            # Add xcentroid, ycentroid from sources
            hdu_table['xcentroid'] = sources['xcentroid']
            hdu_table['ycentroid'] = sources['ycentroid']

            # Add sky columns
            hdu_table['sky_mean'] = sky_table['aperture_sum'] / annuli.area  # Sky per pixel
            hdu_table['sky_std'] = np.sqrt(
                np.maximum(sky_table['aperture_sum'], 0)) / annuli.area  # Sky noise per pixel

            # Sky-subtracted flux
            area_aperture = np.pi * aperture_radius ** 2
            hdu_table['flux_sky_sub'] = hdu_table['aperture_sum'] - hdu_table['sky_mean'] * area_aperture

            # Magnitudes from sky-subtracted flux
            valid_mask = hdu_table['flux_sky_sub'] > 0
            hdu_table['mag'] = np.full(len(hdu_table), np.nan)
            hdu_table['mag'][valid_mask] = -2.5 * np.log10(hdu_table['flux_sky_sub'][valid_mask]) + zeropoint

            # Errors and SNR
            gain = 1.0 if file_path.endswith("_e.fits") else header.get('GAIN', 1.0)
            ron = bkg_std
            bkg_error = np.full_like(data, bkg_std)
            ron_term = ron * gain
            bkg_error_with_ron = np.sqrt(bkg_error ** 2 + ron_term ** 2)
            total_error = calc_total_error(data_sub, bkg_error_with_ron, gain)

            hdu_table_err = aperture_photometry(data_sub, apertures, error=total_error)
            hdu_table['aperture_sum_err'] = hdu_table_err['aperture_sum_err']

            # Breakdown of errors
            hdu_table['flux_err_poisson'] = np.sqrt(np.maximum(hdu_table['flux_sky_sub'], 0)) * gain  # Poisson error
            area_annuli = np.pi * (aperture_radius * 2.5 ** 2 - aperture_radius * 1.5 ** 2)
            hdu_table['flux_err_bg'] = hdu_table['sky_std'] * np.sqrt(area_annuli)  # Background error
            hdu_table['flux_err_rdnoise'] = ron * np.sqrt(area_aperture) * gain  # Readout noise error
            hdu_table['aperture_sum_err'] = np.sqrt(
                hdu_table['flux_err_poisson'] ** 2 + hdu_table['flux_err_bg'] ** 2 + hdu_table[
                    'flux_err_rdnoise'] ** 2)  # Total error

            # Magnitudes error (after aperture_sum_err exists)
            hdu_table['mag_err'] = np.full(len(hdu_table), np.nan)
            hdu_table['mag_err'][valid_mask] = 1.0857 * hdu_table['aperture_sum_err'][valid_mask] / \
                                               hdu_table['aperture_sum'][valid_mask]

            # SNR from sky-subtracted flux
            hdu_table['snr'] = np.full(len(hdu_table), np.nan)
            hdu_table['snr'][valid_mask] = hdu_table['flux_sky_sub'][valid_mask] / hdu_table['aperture_sum_err'][
                valid_mask]

            # FWHM estimate from sources (if available)
            if 'fwhm' in sources.colnames:
                hdu_table['fwhm_est'] = sources['fwhm']
            else:
                hdu_table['fwhm_est'] = fwhm  # Global default

            # Add HDU ID
            hdu_table['hdu_id'] = i

            all_tables.append(hdu_table)
            print(f"HDU {i}: Detected {len(hdu_table)} sources.")

    if not all_tables:
        print("No 2D HDUs found; returning empty list.")
        return []

    return all_tables  # Lista de tablas (una por HDU)

def visualize_photometry(file_path, phot_table, zoom_size=100):
    """
    Visualiza detección y fotometría.
    """
    with fits.open(file_path) as hdul:
        data = hdul[1].data  # Asume HDU 1
        header = hdul[1].header

    # Chequeo robusto: longitud y columnas clave
    if len(phot_table) == 0 or 'xcentroid' not in phot_table.colnames or 'ycentroid' not in phot_table.colnames:
        print(
            f"No sources detected (len={len(phot_table)}, columns={list(phot_table.colnames) if len(phot_table) > 0 else 'empty'}); skipping visualization.")
        return

    # Filtrar a top 50 fuentes por SNR para evitar sobrecarga
    top_mask = phot_table['snr'] > 5  # Ajusta >5 para ~50 fuentes
    filtered_table = phot_table[top_mask]
    if len(filtered_table) == 0:
        print("No high-SNR sources; skipping visualization.")
        return

    positions = np.transpose(
        (filtered_table['xcentroid'], filtered_table['ycentroid']))  # Corregido: 'xcentroid' (con 'i')

    # Brightest source for zoom (de tabla completa)
    brightest_idx = np.argmax(phot_table['aperture_sum'])
    bright_pos = np.array([phot_table['xcentroid'][brightest_idx],
                           phot_table['ycentroid'][brightest_idx]])  # Corregido: 'xcentroid' (con 'i')

    x_min, x_max = max(0, bright_pos[0] - zoom_size), min(data.shape[1], bright_pos[0] + zoom_size)
    y_min, y_max = max(0, bright_pos[1] - zoom_size), min(data.shape[0], bright_pos[1] + zoom_size)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())

    # Rename PNG to "phot + input.png"
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    plot_name = f"phot_{base_name}.png"

    # Full FOV subplot
    ax1.imshow(data, norm=norm, cmap='gray', origin='lower')
    for pos in np.transpose((filtered_table['xcentroid'], filtered_table['ycentroid'])):
        circle1 = plt.Circle(pos, 15.0, color='cyan', fill=False, lw=2)
        ax1.add_patch(circle1)
    ax1.set_xlim(0, data.shape[1])
    ax1.set_ylim(0, data.shape[0])
    ax1.set_title(f'Full FOV (top {len(filtered_table)} sources)')

    # Zoom subplot
    ax2.imshow(data, norm=norm, cmap='gray', origin='lower')
    for pos in np.transpose((filtered_table['xcentroid'], filtered_table['ycentroid'])):
        circle2 = plt.Circle(pos, 15.0, color='red', fill=False, lw=2)
        ax2.add_patch(circle2)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_title('Zoom on Brightest Source')

    plt.tight_layout()
    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Gráfico guardado: {plot_name}")
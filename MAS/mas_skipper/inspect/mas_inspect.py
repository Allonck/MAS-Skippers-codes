#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
import importlib.metadata

# Intentar obtener versión, fallback si no está instalado
try:
    __version__ = importlib.metadata.version('mas_skipper_pipeline')
except importlib.metadata.PackageNotFoundError:
    __version__ = "dev"


def main():
    parser = argparse.ArgumentParser(description="MAS-INSPECT: Interactive Flat Field Inspector")
    parser.add_argument("input", type=str, help="Ruta al archivo FITS")
    parser.add_argument("--ext", type=int, default=1, help="Extensión FITS (Default: 1)")
    args = parser.parse_args()

    # Cargar datos
    try:
        with fits.open(args.input) as hdul:
            data = hdul[args.ext].data.astype(float)
            med = np.nanmedian(data)
            if med > 0: data = data / med  # Normalizar
            print(f"📂 Cargado: {args.input} (Norm por {med:.1f})")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    data = np.nan_to_num(data, nan=1.0)
    ny, nx = data.shape
    init_row = ny // 2
    init_deg = 5

    # --- GUI SETUP ---
    fig = plt.figure(figsize=(12, 10))
    plt.subplots_adjust(left=0.1, bottom=0.20, hspace=0.3)

    # 1. Imagen 2D (Arriba)
    ax_img = plt.axes([0.1, 0.65, 0.8, 0.3])
    ax_img.imshow(data, origin='lower', cmap='gray', vmin=0.9, vmax=1.1, aspect='auto')
    ax_img.set_title("Vista Previa (Master Flat)")
    line_cut = ax_img.axhline(init_row, color='r', linewidth=1)

    # 2. Corte y Ajuste (Medio)
    ax_plot = plt.axes([0.1, 0.40, 0.8, 0.18])
    x_axis = np.arange(nx)
    y_data = data[init_row, :]

    data_line, = ax_plot.plot(x_axis, y_data, 'k-', markersize=2, alpha=0.3, label='Raw Data')
    fit_line, = ax_plot.plot(x_axis, y_data, 'r-', linewidth=2, label='Legendre Fit')
    ax_plot.set_ylabel("Intensidad")
    ax_plot.legend(loc='upper right')
    ax_plot.grid(True, alpha=0.3)

    # 3. Residuos (Abajo - ¡LO IMPORTANTE!)
    ax_res = plt.axes([0.1, 0.15, 0.8, 0.18])
    res_line, = ax_res.plot(x_axis, np.ones_like(x_axis), 'b-', linewidth=1)
    ax_res.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax_res.set_ylabel("Residuos (Data / Fit)")
    ax_res.set_ylim(0.95, 1.05)  # Zoom en los errores
    ax_res.grid(True, alpha=0.3)
    ax_res.set_title("Calidad del Ajuste (Debe ser plano en 1.0)")

    # --- WIDGETS ---
    ax_slice = plt.axes([0.15, 0.05, 0.65, 0.03])
    s_slice = Slider(ax_slice, 'Slice', 0, ny - 1, valinit=init_row, valstep=1)

    ax_degree = plt.axes([0.15, 0.02, 0.65, 0.03])
    s_degree = Slider(ax_degree, 'Degree', 1, 20, valinit=init_deg, valstep=1)

    ax_radio = plt.axes([0.85, 0.02, 0.10, 0.08])
    radio = RadioButtons(ax_radio, ('X (Rows)', 'Y (Cols)'), active=0)

    def update_fit(val=None):
        idx = int(s_slice.val)
        deg = int(s_degree.val)
        mode = radio.value_selected

        if mode == 'X (Rows)':
            y_raw = data[idx, :]
            line_cut.set_ydata([idx, idx])
            line_cut.set_xdata([0, nx - 1])
        else:
            y_raw = data[:, idx]
            line_cut.set_xdata([idx, idx])
            line_cut.set_ydata([0, ny - 1])

        x_ax = np.arange(len(y_raw))

        # Ajuste Robusto (Sigma Clipping)
        fitter = fitting.LinearLSQFitter()
        model = models.Legendre1D(degree=deg)
        or_fit = fitting.FittingWithOutlierRemoval(fitter, sigma_clip, niter=3, sigma=3.0)
        fitted_model, mask = or_fit(model, x_ax, y_raw)
        y_fit = fitted_model(x_ax)

        # Residuos: División (Data / Fit) porque el Flat divide
        # Evitar división por cero
        with np.errstate(divide='ignore', invalid='ignore'):
            residuals = y_raw / y_fit

        # Actualizar gráficos
        data_line.set_data(x_ax, y_raw)
        fit_line.set_data(x_ax, y_fit)
        res_line.set_data(x_ax, residuals)

        # Auto-escala Y del plot principal para ver bien el vignetting
        ax_plot.set_ylim(np.nanmin(y_fit) * 0.95, np.nanmax(y_fit) * 1.05)
        ax_plot.set_xlim(0, len(y_raw))
        ax_res.set_xlim(0, len(y_raw))

        fig.canvas.draw_idle()

    s_slice.on_changed(update_fit)
    s_degree.on_changed(update_fit)
    radio.on_clicked(update_fit)

    plt.show()

if __name__ == "__main__":
    main()
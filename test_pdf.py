#!/usr/bin/env python3
import argparse
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile
import os
import datetime
import mas_skipper_utils as msu
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from io import BytesIO

FONT_REGULAR = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"
FONT_ITALIC = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf"
FONT_BOLDITALIC = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold_Italic.ttf"

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("TimesNewRoman", "", FONT_REGULAR)
        self.add_font("TimesNewRoman", "B", FONT_BOLD)
        self.add_font("TimesNewRoman", "I", FONT_ITALIC)
        self.add_font("TimesNewRoman", "BI", FONT_BOLDITALIC)

from astropy.io import fits

def info_header(path_files, ext=0):
    """
    Extracts information from the header of a FITS file.

    Args:
        path_files (list): A list containing the path to the FITS file. 
                           It's assumed that the relevant file is the first one (path_files[0]).
        ext (int, optional): The extension number of the FITS file to access. 
                            0 for the primary header, 1 for the second extension. Defaults to 0.

    Returns:
        dict: A dictionary containing the extracted header information.
              Returns an empty dictionary if there's an error.
    """

    try:
        with fits.open(path_files[0]) as hdul:  # Use context manager for safe file handling
            hdr = hdul[ext].header
            if ext == 0:
                sequencer = {"Sequencer path": hdr.get("DHEFIRM", None)}
                ext = ext +1
                hdr = hdul[ext].header
                sequencer['DelayHOverlap'] = hdr.get("DELAY_H_OVERLAP", None)
                sequencer['DelayIntegPed'] = hdr.get("DELAY_INTEG_PED", None)
                sequencer['DelayIntegSig'] = hdr.get("DELAY_INTEG_SIG", None)
                sequencer['CCDNCOL'] = hdr.get("CCDNCOL", None)
                sequencer['CCDNROW'] = hdr.get("CCDNROW", None)
                
                return sequencer# Use .get() to avoid KeyError
            elif ext == 1:
                voltages = {
                    "V1AH": hdr.get("V1AH", None),
                    "V1AL": hdr.get("V1AL", None),
                    "H1AH": hdr.get("H1AH", None),
                    "H1AL": hdr.get("H1AL", None),
                    "SWBH": hdr.get("SWBH", None),
                    "SWBL": hdr.get("SWBL", None),
                    "SWAH": hdr.get("SWAH", None),
                    "SWAL": hdr.get("SWAL", None),
                    "RGBH": hdr.get("RGBH", None),
                    "RGBL": hdr.get("RGBL", None),
                    "OGBH": hdr.get("OGBH", None),
                    "OGBL": hdr.get("OGBL", None),
                    "DGAH": hdr.get("DGAH", None),
                    "DGAL": hdr.get("DGAL", None),
                    "DGBH": hdr.get("DGBH", None),
                    "DGBL": hdr.get("DGBL", None),
                    "TGBH": hdr.get("TGBH", None),
                    "TGBL": hdr.get("TGBL", None),
                    "VDRAIN": hdr.get("VDRAIN", None),
                    "VDD": hdr.get("VDD", None),
                    "VR": hdr.get("VR", None),
                    "VSUB": hdr.get("VSUB", None),
                }
                return voltages
            else:
                print(f"Error: Extension {ext} not supported.")
                return {}  # Return an empty dictionary for unsupported extensions

    except FileNotFoundError:
        print(f"Error: File '{path_files[0]}' not found.")
        return {}
    except KeyError as e:
        print(f"Error: Keyword '{e}' not found in header[{ext}].")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}

def limit_gain(path_files):
    gain_limit = int(np.round(len(path_files) / 2))
    if gain_limit % 2 == 0:
        print(f"Image pairs were not load: {gain_limit}")
        gain_limit = gain_limit + 1
        print(f"Fixed: {gain_limit}")
    else:
        print("Image pairs loaded succesfully")
    return gain_limit

def gain_all(pdf, path_files, gain_limit, roi_gain, n_points):
    g_all = []
    gain_figs = []
    roi_figs = []
    for i in range(1, 16 + 1):
        print(f"Processing extension {i}")
        g, _, fig_gain, fig_roi = msu.new_gain_dynamic_ROI_single_extension_fast(
            path_files[0:gain_limit],
            roi_gain[i - 1],
            extension_number=i,
            n_points=n_points,
            savefigs="return",
            rowcolsROI = [4,4]
        )
        g_all.append(g)
        gain_figs.append(fig_gain)
        roi_figs.append(fig_roi)
    return g_all, gain_figs, roi_figs

def fw_all(pdf, path_files, g_all, roifw, n_points):
    # x_data = np.arange(0,11000,1000) / 1000
    ext = 1
    # y_data = y_all[ext-1]

    roifw = msu.roi_shifting([75,175,5,105])
    y_data,x_data = msu.obtain_fw_data(path_files=path_files, roi=roifw[ext-1], extension_number=ext)
    y_data = np.asarray(y_data, dtype=np.float32)
    x_data = np.asarray(x_data, dtype=np.float32)

    gain_by_hand = g_all[ext-1][0] #Change first
    linear_indices, errors = msu.find_linear_subset(x_data, y_data, window_size_initial=6, max_iter_refine=5)

    if linear_indices is not None:
        print("Índices del subconjunto lineal encontrado:", linear_indices)
        print("Errores relativos:", errors)
        print(f"FW: {y_data[linear_indices][-1] / gain_by_hand}")

        #plt.figure(figsize=(10, 6))
        plt.scatter(x_data, y_data, label='Datos originales')
        plt.scatter(x_data[linear_indices], y_data[linear_indices], color='red', label='Subconjunto lineal')
        plt.plot(x_data[linear_indices], LinearRegression().fit(x_data[linear_indices].reshape(-1, 1), 
                                                            y_data[linear_indices]).predict(x_data[linear_indices].reshape(-1, 1))
             , color='green', label=f'Ajuste lineal | FW = {y_data[linear_indices][-1] / gain_by_hand}[e-] | Assuming {gain_by_hand} of gain')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Detección de subconjunto lineal')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No se encontró un subconjunto lineal con el error especificado.")
    return 

    
def create_gain_plots_grid(pdf, figs):
    """
    Inserta los gráficos de ganancia en una cuadrícula de 4x4 en el PDF.
    """
    pdf.add_page()  # Nueva página para los gráficos

    grid_x_start = 10  # Margen izquierdo
    grid_y_start = 20  # Margen superior
    plot_width = 40    # Ancho de cada gráfico
    plot_height = 40   # Alto de cada gráfico
    x_space = 10       # Espacio horizontal entre gráficos
    y_space = 10       # Espacio vertical entre gráficos

    # Orden para las extensiones (esquina inferior izquierda -> arriba, izquierda -> derecha)
    extension_order = np.arange(1,16+1,1)

    for i, ext_index in enumerate(extension_order):
        fig = figs[ext_index - 1]  # Obtener la figura correcta (ajustar el índice)
        if fig is not None:
            img_buffer = BytesIO()
            fig.savefig(img_buffer, format='png')
            plt.close(fig)
            img_buffer.seek(0)

            # Calcular la posición del gráfico en la cuadrícula
            col = i % 4
            row = 3 - (i // 4)  # Invertir el orden de las filas para empezar desde abajo

            x = grid_x_start + col * (plot_width + x_space)
            y = grid_y_start + row * (plot_height + y_space)
    
            pdf.image(img_buffer, x=x, y=y, w=plot_width, h=plot_height)
            img_buffer.close()
        else:
            print(f"Warning: No plot for extension {ext_index}")

def create_roi_plots_grid_1st(pdf, roi_figs):
    """
    Inserta los gráficos de ROI en una cuadrícula de 4x2 en el PDF.
    """
    pdf.add_page()  # Nueva página para los gráficos

    grid_x_start = 10  # Margen izquierdo
    grid_y_start = 5  # Margen superior
    plot_width = 80    # Ancho de cada gráfico
    plot_height = 80   # Alto de cada gráfico
    x_space = 10       # Espacio horizontal entre gráficos
    y_space = 0.1      # Espacio vertical entre gráficos

    extension_order = [1, 2, 3, 4, 5, 6, 7, 8,] # Orden secuencial

    for i, ext_index in enumerate(extension_order):
        fig = roi_figs[ext_index - 1]
        if fig is not None:
            img_buffer = BytesIO()
            fig.savefig(img_buffer, format='png')
            plt.close(fig)
            img_buffer.seek(0)

            col = i % 2
            row = i // 2

            x = grid_x_start + col * (plot_width + x_space)
            y = grid_y_start + row * (plot_height + y_space)

            pdf.image(img_buffer, x=x, y=y, w=plot_width, h=plot_height)
            img_buffer.close()
        else:
            print(f"Warning: No ROI plot for extension {ext_index}")

def create_roi_plots_grid_2nd(pdf, roi_figs):
    """
    Inserta los gráficos de ROI en una cuadrícula de 4x2 en el PDF.
    """
    pdf.add_page()  # Nueva página para los gráficos

    grid_x_start = 10  # Margen izquierdo
    grid_y_start = 5  # Margen superior
    plot_width = 80    # Ancho de cada gráfico
    plot_height = 80   # Alto de cada gráfico
    x_space = 10       # Espacio horizontal entre gráficos
    y_space = 0.1       # Espacio vertical entre gráficos

    extension_order = [9, 10, 11, 12, 13, 14, 15, 16] # Orden secuencial

    for i, ext_index in enumerate(extension_order):
        fig = roi_figs[ext_index - 1]
        if fig is not None:
            img_buffer = BytesIO()
            fig.savefig(img_buffer, format='png')
            plt.close(fig)
            img_buffer.seek(0)

            col = i % 2
            row = i // 2

            x = grid_x_start + col * (plot_width + x_space)
            y = grid_y_start + row * (plot_height + y_space)

            pdf.image(img_buffer, x=x, y=y, w=plot_width, h=plot_height)
            img_buffer.close()
        else:
            print(f"Warning: No ROI plot for extension {ext_index}")
            
def create_readout_noise_plots_grid(pdf, noise_figs):
    """
    Inserta los gráficos de Readout Noise en una cuadrícula de 4x4 en el PDF.
    """
    pdf.add_page()  # Nueva página para los gráficos

    grid_x_start = 10  # Margen izquierdo
    grid_y_start = 20  # Margen superior
    plot_width = 40    # Ancho de cada gráfico
    plot_height = 40   # Alto de cada gráfico
    x_space = 10       # Espacio horizontal entre gráficos
    y_space = 10       # Espacio vertical entre gráficos

    # Orden para las extensiones
    extension_order = np.arange(1, 16 + 1, 1)

    for i, ext_index in enumerate(extension_order):
        fig = noise_figs[ext_index - 1]  # Obtener la figura correcta
        if fig is not None:
            img_buffer = BytesIO()
            fig.savefig(img_buffer, format='png')
            plt.close(fig)
            img_buffer.seek(0)

            # Calcular la posición del gráfico en la cuadrícula (mismo diseño que ganancia)
            col = i % 4
            row = 3 - (i // 4)

            x = grid_x_start + col * (plot_width + x_space)
            y = grid_y_start + row * (plot_height + y_space)

            pdf.image(img_buffer, x=x, y=y, w=plot_width, h=plot_height)
            img_buffer.close()
        else:
            print(f"Warning: No readout noise plot for extension {ext_index}")

def create_report_pdf(filename="my_document.pdf", path=".", roigain=[225,325,845,945], roird=[545, 635, 600, 700], roifw=[75,175,5,105],n_points=4):
    #previous ROIgain: [405, 505, 770, 870] 
    pdf = PDF()
    pdf.add_page()

    path_files = msu.obtain_path_files(path, ends_with=True, filtering=".fits", NOT=False)
    print(f"Se encontraron {len(path_files)} en {path}")
    gain_limit = limit_gain(path_files)
    print(f"Para el cálculo de ganancia se usará hasta el {gain_limit} punto")
    roi_gain = msu.roi_shifting(roigain)  # [340,370,760,840]
    g_all, gain_figs, roi_figs = gain_all(pdf, path_files, gain_limit, roi_gain, n_points)  # Obtener también las figuras de ROI

    title = "Informe de Análisis de Datos MAS-Skipper"
    now = datetime.datetime.now()
    date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

    pdf.set_font("TimesNewRoman", "B", 16)
    pdf.cell(0, 10, title, new_x=XPos.RIGHT, new_y=YPos.TOP, align='L')

    pdf.set_font("TimesNewRoman", "", 10)
    pdf.cell(0, 10, date_time_str, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='R')

    pdf.ln(10)

    pdf.set_font("TimesNewRoman", "", 12)

    preview_files = "\n".join([os.path.basename(f) for f in path_files[:5]])
    if len(path_files) > 5:
        preview_files += "\n..."

    pdf.multi_cell(
        0, 8, f"Found {len(path_files)} FITS files in:\n{path}\n\nPreview:\n{preview_files}", new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )
    # La línea anterior ahora mueve la posición a la siguiente línea después de la multi_cell

    pdf.ln(2) # Agregar un pequeño espacio después de la vista previa

    pdf.cell(0, 8, f"Último archivo: {os.path.basename(path_files[-1])}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(5)
    pdf.ln(5)

# --- Extraer Sequencer y Voltajes ---
    sequencer_info = info_header(path_files, ext=0)
    #####sequencer = sequencer_info.get("sequencer", "N/A")  # Obtener sequencer, "N/A" si no existe

    # Formatear los voltajes para el PDF (puedes ajustar el formato según tus necesidades)
    sequencer_text = "Sequencer:\n"
    for keyh, valueh in sequencer_info.items():
        sequencer_text += f"  {keyh}: {valueh}\n"
    # --- Fin de extracción ---
    
    voltages_info = info_header(path_files, ext=1)

    # Formatear los voltajes para el PDF (puedes ajustar el formato según tus necesidades)
    voltages_text = "Voltajes:\n"
    for key, value in voltages_info.items():
        voltages_text += f"  {key}: {value}\n"
    # --- Fin de extracción ---

    ########pdf.cell(0, 10, f"Sequencer: {sequencer}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.multi_cell(0, 10, sequencer_text)  # Insertar todos los voltajes

    pdf.multi_cell(0, 10, voltages_text)  # Insertar todos los voltajes

    pdf.ln(1)

    for line in [
        "Plot de ROI de ruido x16.",
        "Tabla de ruido. (Plotear ruido de bias y ruido de última exp. Saturada)",
        "Plot de FW x16.",
    ]:
        pdf.cell(0, 10, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.cell(0, 10, "Plot de ganancia x 16.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    create_gain_plots_grid(pdf, gain_figs)  # Usar las figuras de ganancia

    pdf.add_page() # Nueva página para los plots de ROI
    pdf.cell(0, 10, "Plot de ROI de ganancia x 16.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    create_roi_plots_grid_1st(pdf, roi_figs) # Nueva función para los plots de ROI
    create_roi_plots_grid_2nd(pdf, roi_figs) 

    print("Calcularing Full well capacity...")
    # ------- FW CALCULATION -------------
    pdf.add_page() # Nueva página para los plots de Readout Noise
    pdf.cell(0, 10, "Plot de FW para cada extensión.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    fw_all(pdf, path_files, g_all, roifw, n_points)

    pdf.add_page()
    # --- Calculate Readout Noise and Get Plots ---
    print("Calculating Readout Noise...")
    readout_noise_data, _, readout_noise_figs, _ = msu.save_readout_noise(
        path_files, gain=g_all, roi_base= roird, return_mean=False, save_txt=False, saveplots="return"
    )
    
    pdf.add_page() # Nueva página para los plots de Readout Noise
    pdf.cell(0, 10, "Plot de Readout Noise x 16.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    create_readout_noise_plots_grid(pdf, readout_noise_figs)

    pdf.add_page() # Nueva página para los plots de Readout Noise

    pdf.cell(0, 10, "Readout Noise Values (e-):", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    if readout_noise_data is not None:
        for noise_values in readout_noise_data:
            noise_line = " ".join(f"{float(x):.2f}" for x in noise_values)
            pdf.cell(0, 8, noise_line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
    else:
        pdf.cell(0, 8, "Readout noise data could not be calculated.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(5)
    
    pdf.output(filename)
    print(f"PDF '{filename}' creado exitosamente.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar un reporte PDF.")
    parser.add_argument("--filename", type=str, default="my_document.pdf", help="Nombre del archivo PDF de salida.")
    parser.add_argument("--path", type=str, default=".", help="Path donde están los .fits")
    parser.add_argument("--roigain", type=int, nargs=4, default=[370, 420, 800, 850], help="ROI para calcular ganancia.")
    parser.add_argument("--roird", type=int, nargs=4, default=[545, 635, 40, 140], help="ROI para calcular ruido de lectura.")
    parser.add_argument("--roifw", type=int, nargs=4, default=[75, 175, 5, 105], help="ROI para calcular full well.")
    parser.add_argument("--npoints", type=int, default=4, help="Puntos a considerar para cálculo de ganancia.")

    args = parser.parse_args()

    create_report_pdf(filename=args.filename, path=args.path, roigain=args.roigain, n_points=args.npoints)

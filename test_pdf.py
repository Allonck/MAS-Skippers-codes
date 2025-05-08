#!/usr/bin/env python3
import argparse
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile
import os
import io
import datetime

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

def create_report_pdf(filename="my_document.pdf"):
    pdf = PDF()
    pdf.add_page()

    title = "Informe de Análisis de Datos MAS-Skipper"
    now = datetime.datetime.now()
    date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

    pdf.set_font("TimesNewRoman", "B", 16)
    pdf.cell(0, 10, title, new_x=XPos.RIGHT, new_y=YPos.TOP, align='L')

    pdf.set_font("TimesNewRoman", "", 10)
    pdf.cell(0, 10, date_time_str, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='R')

    pdf.ln(10)

    pdf.set_font("TimesNewRoman", "", 12)
    for line in [
        "Contenido:",
        "Fecha:",
        "Hora:",
        "Filenames:",
        "Full path:",
        "Sequencer:",
        "Voltajes:",
        "",
        "Plot de ganancia x 16.",
        "Plot de ROI de ganancia x 16.",
        "Plot de ROI de ruido x16.",
        "Tabla de ruido. (Plotear ruido de bias y ruido de última exp. Saturada)",
        "Plot de FW x16.",
    ]:
        pdf.cell(0, 10, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(filename)
    print(f"PDF '{filename}' creado exitosamente.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar un reporte PDF.")
    parser.add_argument("--filename", type=str, default="my_document.pdf", help="Nombre del archivo PDF de salida.")
    args = parser.parse_args()

    create_report_pdf(args.filename)
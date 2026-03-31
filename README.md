# MASSKIP: Multi-Amplifier Sensing Skipper CCD Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/968102861.svg)](https://doi.org/10.5281/zenodo.19260714)
<a href="https://ascl.net/2603.022"><img src="https://img.shields.io/badge/ascl-2603.022-blue.svg?colorB=262255" alt="ascl:2603.022" /></a>

**MASSKIP** is a modular Python pipeline designed for the reduction, calibration, and photometric analysis of images acquired with **Multi-Amplifier Sensing (MAS) Skipper CCDs**. 

Originally developed and tested for the 16-channel MAS prototype at the SMARTS 0.9m telescope (CTIO), this pipeline is optimized to handle the specific challenges of Skipper CCDs, such as amplifier heterogeneity and deep sub-electron read noise regimes. It supports Multi-Extension FITS (MEF) files and follows standard reduction procedures enhanced by dynamic noise-weighted algorithms.

---

## 🧰 Core Features

* **Advanced Calibration:**
  * Overscan correction per extension (with independent ROI shifting for crosstalk mitigation).
  * Pixel-by-pixel bias subtraction.
  * Normalized master flat creation (pixel-wise division and polynomial fitting).
* **Noise-Optimized Combination:**
  * Multiple image combination methods, including **Variance/SNR-weighted combination** to dynamically suppress noisy amplifiers and achieve theoretical sub-electron read noise limits.
* **Astrometry & Cosmic Rays:**
  * Enhanced cosmic ray rejection and dynamic WCS injection.
* **Ready-to-Use Science Outputs:**
  * Extension-preserving outputs compatible with `ds9 -mosaicimage iraf`.
  * Physical unit conversion (ADUs to $e^-$) preserved in the final combined headers.
* **Integrated Analysis Tools:**
  * Built-in photometry module (`mas-phot`) utilizing `photutils` (DAOFIND/DAOPHOT).
  * Fast stacking and RGB composing (`mas-stack`).

---

## 📦 Installation with venv

```bash
git clone [https://github.com/Allonck/MAS-Skippers-codes.git](https://github.com/Allonck/MAS-Skippers-codes.git)
cd MAS
python3 -m venv .maspipeline
source .maspipeline/bin/activate
pip install -e .
```
To verify the install: 
```bash
mas-ccd -h
```

## 📦 Configure Jupyter notebook with venv

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name=.venv --display-name "MASSKIP-env"
``` 

In Jupyter, select `Kernel` -> `Change Kernel` -> `MASSKIP-env`.

---

## ⚙️ Usage Examples (CLI)

MASSKIP provides Command Line Interfaces (CLI) also.

### 1. Full Data Reduction (`mas-ccd`)

```bash
mas-ccd \
  --full-reduction \
  --sci-pattern "science_filename_*.fits"
```

* Workflow executed: `Overscan correction` -> `Sigma-clipped Master Bias generation` -> `Bias subtraction` -> `Master Flat generation` -> `Flat-fielding` -> `Cosmic ray rejection` -> `WCS addition` -> `Multi-extension combination`.

### 2. Photometric Analysis (`mas-phot`)

```bash
mas-phot \
  --input "science_filename_*.fits" \
  --aperture-radius 10 \
  --do-visualize \
  --zeropoint 25
```

* Workflow executed: `Source detection (DAOFIND)` -> `Aperture Photometry` -> `Interactive visual plotting` -> `Photometric catalog generation.`.

---

## General management
To enter the venv:
```bash
source ~/.maspipeline/bin/activate
```
To exit the venv:
```bash
deactivate
```

---

## Architecture & Important Notes

* Sensor geometry: The current build assumes a 16-extension MEF format (excluding primary HDU).

* Overscan and Extension order: Fixed by default but can be changed using the `roi_shifting()` module.

* Header Preservation: FITS headers are strictly preserved in all intermediate outputs to support DS9 visualization (`-mosaic iraf`). The pipeline only modifies the DATASEC keyword when needed.

---

## Citation

If you use MASSKIP in your research, please cite the software using the following BibTeX entry:

```bibtex
@software{masskip_2026,
  author       = {Montalbán, C. K.},
  title        = {{MASSKIP: Multi-Amplifier Sensing Skipper CCD Pipeline}},
  month        = mar,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v2.3.3},
  doi          = {10.5281/zenodo.19260715},
  url          = {[https://github.com/Allonck/MAS-Skippers-codes/tree/refactor](https://github.com/Allonck/MAS-Skippers-codes/tree/refactor)}
}
```
---
## Acknowledgments & Credits

MASSKIP was developed by @Allonck at the Universidad de La Serena.

The pipeline was originally designed to process data from the SMARTS 0.9m telescope at the Cerro Tololo Inter-American Observatory (CTIO, part of NSF-NOIRLab) using a 16-channel MAS Skipper CCD prototype developed by LBNL & Fermilab, characterized and tested at NOIRLab.

License: This project is licensed under the MIT License - see the LICENSE file for details.

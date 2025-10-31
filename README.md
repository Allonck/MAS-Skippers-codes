# MASSKIP – Multi-Amplifier Sensing Skipper CCD Pipeline

**MASSKIP** is a prototype Python pipeline designed for the basic reduction of images acquired with a **16-channel MAS Skipper CCD**. It supports MEF (Multi-Extension FITS) files and follows a standard overscan + bias + flat calibration procedure, inspired by data reduction pipelines such as SOAR/Goodman.

---

## 🧰 Features (At the moment of 2.0.0)

- ✅ Overscan correction per extension (with independent ROI shifting considering crosstalk).
- ✅ Pixel-by-pixel bias subtraction.
- ✅ Normalized master flat creation.
- ✅ Flat-fielding normalization with pixel-wise division and polynomial fitting.
- ✅ Extension-preserving output compatible with `ds9 -mosaicimage iraf`.
- ✅ Multiple simple and graphical SNR-weighted based combination of images using mean.
- ✅ Cosmic rays rejection available and enhanced.
- ✅ Add WCS.
- ✅ Add full ROI support.
- ✅ ADUs to e- values available in final combined images.
- ✅ Header history in final combined images.
- ⚙️ CLI tool: `mas-ccd` for batch reduction.

---

## 📦 Installation with venv

```bash
git clone https://github.com/Allonck/MAS-Skippers-codes.git
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
Then you can create an .ipynb file, go to "Kernel" -> "Change Kernel" -> "Select 'MASSKIP-env'"

## ⚙️ Example Use of CLI

mas-ccd \
  --full-reduction \
  --sci-pattern "science_filename_*.fits" 
  
This will:

   * Apply overscan correction to all biases.

   * Generate a master bias with sigma clipping.

   * Apply overscan + bias subtraction to science images.

   * Generate a normalized master flat.

   * Apply flat-fielding to all bias-corrected science images.

   * Add WCS

   * Annihilate cosmic rays.

   * Combine all the extensions into a Photometry / Astrometry ready Science image.

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

## IMPORTANT NOTES

* Assumes 16 extensions (excluding primary HDU).

* Overscan ROI is fixed but can be shifted per extension using roi_shifting().

* Headers are preserved in all outputs to support DS9 visualization (-mosaic iraf) and only modifies the keyword DATASEC.

* The pipeline is modular and intended for further extension in the near future (e.g. decorrelation, photometry, astrometry).

---
## Credits

Developed by @Allonck, originally for internal use with the [SMARTS 0.9m telescope@CTIO with a 16ch-MAS belonging to LBNL & Fermilab, characterized and tested at NOIRLab].

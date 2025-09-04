# MASSKIP – Multi-Amplifier Sensing Skipper CCD Pipeline

**MASSKIP** is a prototype Python pipeline designed for the basic reduction of images acquired with a **16-channel MAS Skipper CCD**. It supports MEF (Multi-Extension FITS) files and follows a standard overscan + bias + flat calibration procedure, inspired by professional pipelines such as SOAR/Goodman.

---

## 🧰 Features (At the moment of 0.3.0)

- ✅ Overscan correction per extension (with independent ROI shifting considering crosstalk).
- ✅ Pixel-by-pixel bias subtraction.
- ✅ Normalized master flat creation.
- ✅ Flat-fielding normalization with pixel-wise division.
- ✅ Extension-preserving output compatible with `ds9 -mosaicimage iraf`.
- ✅ Mutiple simple-combination of images using mean.
- ✅ Cosmic rays rejection available.
- ⚙️ CLI tool: `mas-ccd` for batch reduction.

---

## 📦 Installation

```bash
git clone https://github.com/youruser/masskip.git
cd masskip
python3 -m venv .maspipeline
source .maspipeline/bin/activate
pip install -e .
```

---

## ⚙️ Example Use of CLI

mas-ccd \
  --raw ./raw_data \
  --reduction
  --sci-pattern "sci*.fits" \
  --output ./reduced 
  
This will:

   * Apply overscan correction to all biases.

   * Generate a master bias with sigma clipping.

   * Apply overscan + bias subtraction to science images.

   * Generate a normalized master flat.

   * Apply flat-fielding to all bias-corrected science images.

---

## IMPORTANT NOTES

* Assumes 16 extensions (excluding primary HDU).

* Overscan ROI is fixed but can be shifted per extension using roi_shifting().

* Headers are preserved in all outputs to support DS9 visualization (-mosaic iraf) and only modifies the keyword DATASEC.

* The pipeline is modular and intended for further extension in the near future (e.g., cosmic ray rejection, gain correction).

---
## Credits

Developed by @Allonck, originally for internal use with the [SMARTS 0.9m telescope@Noirlab with the MAS-16ch belonging to LBNL & Fermilab].

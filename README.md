# AstroPhotoScripts

A high-performance CLI pipeline for multi-exposure nightscapes, RAW/DNG bracket sequences, star stacking, and landscape exposure blending.

Designed to solve the classic nightscape dilemma:
- **Overexposed Mountains/Landscape**: Bright headlights, moonlight, or long exposures blow out terrain textures. The pipeline automatically analyzes clipping to either select the optimal non-overexposed exposure or fuse well-exposed shadow/highlight frames using Mertens exposure fusion.
- **Star Tracking & Noise Reduction**: Because Earth rotates, stars drift across exposures. The pipeline uses frequency-domain phase correlation to detect and correct sub-pixel star drift, stacks exposures with SNR-optimal weighting, levels background vignetting, and applies an Asinh astrophotography stretch to reveal faint stars, constellations (e.g. Orion), and nebular details (M42) without noise elevation.
- **Multithreaded Performance**: Explicitly configures OpenCV to utilize all available CPU cores with SIMD acceleration.
- **RAW & DNG Support**: Native support for camera RAW formats (`.dng`, `.cr2`, `.nef`, `.arw`) alongside standard formats (`.jpg`, `.png`, `.tiff`).

---

## Installation

```bash
git clone https://github.com/T-Dynamos/AstroPhotoScripts.git
cd AstroPhotoScripts

# Install dependencies
pip install -r requirements.txt

# (Optional) Install CLI globally or into your environment
pip install -e .
```

---

## Quick Start (CLI)

### 1. Process an Exposure Sequence
Run the complete dual-stack pipeline on a folder of bracketed photos:
```bash
astrophoto run /path/to/exposures --output ./output
# Or directly with python:
python main.py run /path/to/exposures --output ./output
```

### 2. Inspect Image Sequence
Inspect shutter speeds, ISOs, formats, and highlight/shadow clipping without running the full stack:
```bash
astrophoto inspect /path/to/exposures
```

### 3. Local Constellation & DSO Auto-Identification (Plate Solving)
Locally auto-identify constellations, stars, and deep-sky objects (e.g., Orion, M42 Nebula, Pleiades) and draw glowing celestial overlays:
```bash
astrophoto identify output/astro_composite.jpg
```

---

## CLI Options

```
usage: astrophoto run [-h] [--output OUTPUT] [--ground-mode {fuse,select}]
                      [--max-clip MAX_CLIP] [--sky-method {weighted,median}]
                      [--stretch STRETCH] [--no-denoise] [--no-layers]
                      [input]

positional arguments:
  input                 Input directory containing exposures (default: current directory).

options:
  --output, -o OUTPUT   Directory to save final master TIFF, previews, and layers (default: 'output').
  --ground-mode {fuse,select}
                        'fuse': blends non-overexposed exposures via Mertens fusion (default).
                        'select': picks the single best non-overexposed shot.
  --max-clip MAX_CLIP   Maximum allowable highlight clipping percentage for the landscape (default: 0.50%).
  --sky-method {weighted,median}
                        Sky stacking method: 'weighted' (optimal SNR, default) or 'median' (satellite rejection).
  --stretch STRETCH     Astrophotography Asinh contrast stretch factor for faint stars and nebulae (default: 28.0).
  --no-denoise          Disable chroma noise suppression on sky background.
  --no-layers           Skip saving individual intermediate layers.
```

---

## Pipeline Architecture

```
                            Input Sequence (RAW/DNG or JPEG/TIFF)
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
             [Ground Registration]                           [Celestial Star Drift]
          Sub-pixel ECC minimization                     Frequency-domain phase correlation
                      │                                               │
                      ▼                                               ▼
             [Landscape Layer]                               [Star-Aligned Sky]
     Mode 'fuse' (Mertens on safe frames)              SNR-weighted or median stacking
     Mode 'select' (safest non-blown frame)            Background leveling & vignetting correction
                      │                                Asinh non-linear contrast stretch
                      │                                Selective chroma denoising
                      └───────────────────────┬───────────────────────┘
                                              ▼
                                   [Horizon Segmentation]
                             Multi-cue bottom-anchored Otsu
                             Edge-preserving Guided Filter
                                              ▼
                                      [Final Master]
                            16-bit uncompressed master TIFF
                            8-bit JPEG preview & summary card
                            Offline Constellation Plate Solve (Orion / M42)
```

---

## Output Deliverables

All deliverables are exported to the designated output folder:
- `astro_composite.tiff`: **16-bit uncompressed TIFF master** preserving full dynamic range.
- `astro_composite.jpg`: High-quality 8-bit preview.
- `comparison_summary.jpg`: Side-by-side comparison card (Raw Blown Shot vs Landscape Layer vs Final Composite).
- `ground_layer.jpg`: Selected or fused landscape layer.
- `sky_layer.jpg`: Star-aligned, contrast-stretched sky layer.
- `horizon_mask.jpg`: Edge-feathered terrain ridge mask.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

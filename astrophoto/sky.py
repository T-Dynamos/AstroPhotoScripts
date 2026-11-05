"""
sky.py - Astrophotography sky stacking, background leveling, and contrast enhancement.
"""

from typing import List, Optional
import cv2
import numpy as np


def stack_sky_exposures(
    star_aligned_images: List[np.ndarray],
    exposure_times: List[Optional[float]],
    method: str = "weighted",
) -> np.ndarray:
    """Stack star-aligned images to suppress sensor thermal and read noise."""
    if method == "median" and len(star_aligned_images) >= 3:
        stack = np.stack(star_aligned_images, axis=0)
        return np.median(stack, axis=0).astype(np.float32)

    valid_times = [t if (t is not None and t > 0) else 1.0 for t in exposure_times]
    weights = np.sqrt(np.array(valid_times, dtype=np.float32))
    weights /= np.sum(weights)

    stacked = np.zeros_like(star_aligned_images[0], dtype=np.float32)
    for img, w in zip(star_aligned_images, weights):
        stacked += img * w

    return np.clip(stacked, 0.0, 1.0)


def flatten_sky_background(
    sky_bgr: np.ndarray,
    sky_mask: np.ndarray,
    filter_kernel_size: int = 9,
    blur_sigma: int = 151,
) -> np.ndarray:
    """Remove background vignetting and light pollution without bleeding bright ground into sky.

    Crucial fix for horizon dark lines: The ground is masked out and filled with the median
    sky level prior to computing the background gradient. This prevents the bright terrestrial
    landscape from artificially inflating the background estimate and carving out a dark trench.
    """
    gray = cv2.cvtColor(sky_bgr, cv2.COLOR_BGR2GRAY)
    pure_sky_pixels = gray[sky_mask[:, :, 0] > 0.8]
    sky_median = float(np.median(pure_sky_pixels)) if len(pure_sky_pixels) > 0 else 0.04

    # Fill ground with sky_median so ground does NOT bleed into the background blur
    gray_sky_only = gray.copy()
    gray_sky_only[sky_mask[:, :, 0] <= 0.5] = sky_median

    gray_u8 = np.clip(gray_sky_only * 255.0, 0, 255).astype(np.uint8)

    # Morphological star removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filter_kernel_size, filter_kernel_size))
    bg_no_stars = cv2.morphologyEx(gray_u8, cv2.MORPH_OPEN, kernel).astype(np.float32) / 255.0

    # Smooth 2D illumination profile
    bg_gradient = cv2.GaussianBlur(bg_no_stars, (blur_sigma, blur_sigma), 0)

    # Target baseline: 15th percentile of background gradient in pure sky
    bg_sky_pixels = bg_gradient[sky_mask[:, :, 0] > 0.8]
    target_bg = float(np.percentile(bg_sky_pixels, 15)) if len(bg_sky_pixels) > 0 else 0.04

    # Subtract only true sky vignetting excess
    excess = np.maximum(bg_gradient - target_bg, 0.0)
    excess_bgr = np.expand_dims(excess, axis=2)

    leveled = np.maximum(sky_bgr - excess_bgr * 0.80, 0.0)
    return np.clip(leveled, 0.0, 1.0)


def stretch_sky_asinh(
    sky_bgr: np.ndarray,
    sky_mask: np.ndarray,
    beta: float = 28.0,
    target_baseline: float = 0.045,
    boost_factor: float = 1.25,
) -> np.ndarray:
    """Apply non-linear Asinh astrophotography stretch with a preserved dark sky floor.

    Anchors the background sky to a clean, velvety floor (~0.045 / ~11 RGB) so that
    near-horizon pixels never clip to zero or produce dark fringe lines.
    """
    gray = cv2.cvtColor(sky_bgr, cv2.COLOR_BGR2GRAY)
    pure_sky_pixels = gray[sky_mask[:, :, 0] > 0.8]

    if len(pure_sky_pixels) > 0:
        black_pt = float(np.percentile(pure_sky_pixels, 12))
    else:
        black_pt = float(np.percentile(gray, 10))

    # Extract positive signal above black point
    signal = np.maximum(sky_bgr - black_pt, 0.0)
    denom = max(1e-5, float(np.arcsinh(beta * (1.0 - black_pt))))

    # Asinh curve preserves star color and reveals faint nebulae/Milky Way dust
    stretched_signal = np.arcsinh(beta * signal) / denom

    # Reconstruct sitting on the guaranteed velvety dark baseline
    result = target_baseline + (1.0 - target_baseline) * stretched_signal * boost_factor
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def denoise_sky_chroma(
    sky_bgr: np.ndarray,
    h_lum: float = 4.0,
    h_color: float = 6.0,
) -> np.ndarray:
    """Apply selective chroma and background denoising without softening star cores."""
    u8 = np.clip(sky_bgr * 255.0, 0, 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoisingColored(
        u8,
        None,
        h=h_lum,
        hColor=h_color,
        templateWindowSize=7,
        searchWindowSize=17,
    )
    return denoised.astype(np.float32) / 255.0

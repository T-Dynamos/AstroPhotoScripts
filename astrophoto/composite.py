"""
composite.py - Generalized horizon segmentation, edge-guided blending, and comparison cards.

Supports any terrestrial landscape:
- Mountains, cliffs, and hills
- Trees, forests, and fine foliage
- Architecture, rooflines, towers, and city skylines
- Plains, desert dunes, beaches, and water horizons
- Both illuminated foregrounds and dark silhouette horizons
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 15,
    eps: float = 1e-3,
) -> np.ndarray:
    """Fast edge-preserving Guided Image Filter (He, Sun, Tang 2010).

    Transfers fine geometric structural edges (leaves, rooflines, antennas, terrain ridges)
    from the guide luminance image to the alpha segmentation mask without halo artifacts.
    Runs in linear time O(N) using multi-threaded box filters.
    """
    mean_I = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
    mean_p = cv2.boxFilter(src, cv2.CV_32F, (radius, radius))
    mean_Ip = cv2.boxFilter(guide * src, cv2.CV_32F, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))

    q = mean_a * guide + mean_b
    return np.clip(q, 0.0, 1.0)


def segment_horizon_mask(
    reference_image_bgr: np.ndarray,
    foreground_type: str = "auto",
    guided_radius: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment any terrestrial foreground (trees, buildings, landscape) from celestial sky."""
    if reference_image_bgr.dtype != np.float32:
        img_float = reference_image_bgr.astype(np.float32) / 255.0
    else:
        img_float = reference_image_bgr

    h, w = img_float.shape[:2]
    gray = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)

    top_half = gray[:int(h * 0.45), :]
    bottom_half = gray[int(h * 0.55):, :]
    top_mean = float(np.mean(top_half))
    bottom_mean = float(np.mean(bottom_half))

    if foreground_type == "auto":
        is_illuminated = bottom_mean >= top_mean
    else:
        is_illuminated = foreground_type.lower() == "illuminated"

    blur = cv2.GaussianBlur(gray, (15, 15), 0)

    if is_illuminated:
        sky_baseline = float(np.percentile(top_half, 40))
        ground_baseline = float(np.percentile(bottom_half, 40))
        thresh_val = sky_baseline + (ground_baseline - sky_baseline) * 0.30
        thresh_val = max(0.04, min(thresh_val, 0.35))
        coarse_mask = (blur > thresh_val).astype(np.uint8) * 255
    else:
        sky_baseline = float(np.percentile(top_half, 60))
        ground_baseline = float(np.percentile(bottom_half, 40))
        thresh_val = ground_baseline + (sky_baseline - ground_baseline) * 0.50
        coarse_mask = (blur < thresh_val).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    coarse_mask = cv2.morphologyEx(coarse_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(coarse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(coarse_mask)
    min_area = int(w * h * 0.015)
    for c in contours:
        if cv2.contourArea(c) > min_area:
            cv2.drawContours(filled, [c], -1, 255, -1)

    filled[int(h * 0.95):, :] = 255

    coarse_float = filled.astype(np.float32) / 255.0
    refined_mask = guided_filter(gray, coarse_float, radius=guided_radius, eps=1e-3)

    mask_fg = np.expand_dims(refined_mask, axis=2)
    mask_sky = 1.0 - mask_fg
    return mask_fg, mask_sky


generate_ridge_mask = segment_horizon_mask


def blend_composite(
    foreground_layer: np.ndarray,
    sky_layer: np.ndarray,
    mask_foreground: np.ndarray,
    mask_sky: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Seamlessly composite terrestrial landscape and celestial sky using edge-guided alpha masks."""
    if mask_sky is None:
        mask_sky = 1.0 - mask_foreground

    composite = (foreground_layer * mask_foreground) + (sky_layer * mask_sky)
    return np.clip(composite, 0.0, 1.0).astype(np.float32)


def create_comparison_card(
    longest_shot: np.ndarray,
    foreground_layer: Optional[np.ndarray] = None,
    final_composite: Optional[np.ndarray] = None,
    ground_layer: Optional[np.ndarray] = None,
    shutter_label: str = "Longest Exposure",
    thumb_w: int = 533,
    thumb_h: int = 400,
) -> np.ndarray:
    """Create a 3-panel comparison card (Raw Blown Shot vs Landscape Layer vs Final Composite)."""
    # Accept ground_layer as alias for foreground_layer
    fg = foreground_layer if foreground_layer is not None else ground_layer

    def to_u8(img: np.ndarray) -> np.ndarray:
        if img.dtype != np.uint8:
            return np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img

    t1 = cv2.resize(to_u8(longest_shot), (thumb_w, thumb_h))
    t2 = cv2.resize(to_u8(fg), (thumb_w, thumb_h))
    t3 = cv2.resize(to_u8(final_composite), (thumb_w, thumb_h))

    def add_banner(thumb: np.ndarray, text: str) -> np.ndarray:
        bar = np.zeros((40, thumb.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, text, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([bar, thumb])

    card = np.hstack([
        add_banner(t1, f"Single Shot ({shutter_label}) - Overexposed"),
        add_banner(t2, "Selected/Fused Landscape Layer"),
        add_banner(t3, "AstroPhotoScripts Dual-Stack Output"),
    ])
    return card

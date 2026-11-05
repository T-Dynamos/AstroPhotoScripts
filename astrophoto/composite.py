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
    guided_radius: int = 25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment any terrestrial foreground (mountains, trees, buildings, railings, balconies) from celestial sky.

    Uses normalized multi-cue Otsu segmentation anchored to the bottom terrestrial edge,
    eliminating false mountain blobs or domes in the night sky.
    Refines boundaries with an edge-preserving guided filter to ensure seamless alpha blending.
    """
    if reference_image_bgr.dtype != np.float32:
        img_float = reference_image_bgr.astype(np.float32)
        if img_float.max() > 1.5:
            img_float /= 255.0
    else:
        img_float = reference_image_bgr

    h, w = img_float.shape[:2]
    gray = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)

    # Robust percentile normalization (works across RAW DNG linear values and gamma JPEGs)
    p2 = float(np.percentile(gray, 2))
    p98 = float(np.percentile(gray, 98))
    norm_gray = np.clip((gray - p2) / max(1e-4, p98 - p2), 0.0, 1.0)

    # Morphological star removal so star point sources do not trigger false foreground edges
    star_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    smooth_gray = cv2.morphologyEx(norm_gray, cv2.MORPH_OPEN, star_kernel)
    smooth_gray = cv2.GaussianBlur(smooth_gray, (15, 15), 0)

    # Sample zenith sky (top 25%) vs terrestrial ground (bottom 20%)
    sky_sample = smooth_gray[:int(h * 0.25), :]
    ground_sample = smooth_gray[int(h * 0.80):, :]
    sky_mean = float(np.mean(sky_sample))
    ground_mean = float(np.mean(ground_sample))

    # Otsu thresholding on lower 75% of the frame
    roi = smooth_gray[int(h * 0.20):, :]
    roi_u8 = (roi * 255).astype(np.uint8)
    otsu_thresh, _ = cv2.threshold(roi_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_val = otsu_thresh / 255.0

    if foreground_type == "auto":
        is_illuminated = ground_mean >= sky_mean
    else:
        is_illuminated = foreground_type.lower() == "illuminated"

    if is_illuminated:
        thresh = (otsu_val + sky_mean) / 2.0 if otsu_val > sky_mean else (sky_mean + ground_mean) / 2.0
        raw_mask = (smooth_gray > thresh).astype(np.float32)
    else:
        thresh = (otsu_val + ground_mean) / 2.0 if otsu_val < sky_mean else (sky_mean + ground_mean) / 2.0
        raw_mask = (smooth_gray < thresh).astype(np.float32)

    # Guaranteed topological priors:
    # 1. Top 18% is guaranteed to be sky
    raw_mask[:int(h * 0.18), :] = 0.0
    # 2. Bottom 5% is guaranteed to be ground
    raw_mask[int(h * 0.95):, :] = 1.0

    # Morphological closing to seal small wire/railing/mesh gaps
    mask_u8 = (raw_mask * 255).astype(np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask_closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)

    # Connected component analysis: keep ONLY components touching the bottom edge (ground)
    # This completely eliminates floating false mountain blobs/domes in the sky
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_closed)
    ground_mask = np.zeros((h, w), dtype=np.uint8)
    bottom_labels = set(labels[-1, :]) | set(labels[-3, :]) | set(labels[-10, :])
    if 0 in bottom_labels:
        bottom_labels.remove(0)

    for lbl in bottom_labels:
        if stats[lbl, cv2.CC_STAT_AREA] > (w * h * 0.005):
            ground_mask[labels == lbl] = 255

    # Fill downward per column from the topmost detected terrestrial point
    # Ensures solid terrain/mountains/balconies without false sky holes
    step = 2
    for c in range(0, w, step):
        fg_rows = np.where(ground_mask[:, c] == 255)[0]
        if len(fg_rows) > 0:
            first_fg = fg_rows[0]
            ground_mask[first_fg:, max(0, c - step // 2):min(w, c + step // 2 + 1)] = 255

    # Edge-preserving guided filter refinement on normalized grayscale
    coarse_float = ground_mask.astype(np.float32) / 255.0
    refined_mask = guided_filter(norm_gray, coarse_float, radius=guided_radius, eps=1e-3)

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
    input_images: List[np.ndarray],
    input_labels: List[str],
    landscape_layer: np.ndarray,
    sky_layer: np.ndarray,
    final_composite: np.ndarray,
    target_width: int = 1800,
    **kwargs,
) -> np.ndarray:
    """Create a comprehensive multi-row comparison summary card.

    Row 1: All raw input exposures labeled with shutter speeds and metadata.
    Divider: Section separator.
    Row 2: Processed deliverables:
           - Highlight-safe landscape layer
           - Celestial star-aligned & stretched sky layer
           - Final AstroPhotoScripts composite master
    """
    def to_u8(img: np.ndarray) -> np.ndarray:
        if img.dtype != np.uint8:
            return np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img

    def make_card(img: np.ndarray, title: str, w: int, h: int) -> np.ndarray:
        u8 = to_u8(img)
        thumb = cv2.resize(u8, (w, h))
        banner = np.zeros((36, w, 3), dtype=np.uint8)
        cv2.putText(
            banner, title, (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA
        )
        return np.vstack([banner, thumb])

    # Backward compatibility if called with old 3-arg signature
    if not isinstance(input_images, list):
        longest_shot = input_images
        fg = input_labels if isinstance(input_labels, np.ndarray) else landscape_layer
        comp = landscape_layer if isinstance(landscape_layer, np.ndarray) else final_composite
        t_w, t_h = target_width // 3, int((target_width // 3) * 0.75)
        c1 = make_card(longest_shot, "Single Long Exposure (Overexposed)", t_w, t_h)
        c2 = make_card(fg, "Landscape Layer (Highlight-Safe)", t_w, t_h)
        c3 = make_card(comp, "AstroPhotoScripts Final Master", target_width - 2 * t_w, t_h)
        return np.hstack([c1, c2, c3])

    num_inputs = max(1, len(input_images))
    col_w = max(200, target_width // num_inputs)
    col_h = int(col_w * 0.75)

    # Row 1: All input exposures
    row1_cards = [
        make_card(img, label, col_w, col_h)
        for img, label in zip(input_images, input_labels)
    ]
    row1 = np.hstack(row1_cards)
    actual_w = row1.shape[1]

    # Row 2: 3 key processed deliverables
    r2_col_w = actual_w // 3
    r2_col_h = int(r2_col_w * 0.70)
    row2_cards = [
        make_card(landscape_layer, "Landscape Layer (Highlight-Safe)", r2_col_w, r2_col_h),
        make_card(sky_layer, "Celestial Sky (Star-Stacked)", r2_col_w, r2_col_h),
        make_card(final_composite, "AstroPhotoScripts Final Master", actual_w - 2 * r2_col_w, r2_col_h),
    ]
    row2 = np.hstack(row2_cards)

    # Divider bar
    divider = np.zeros((26, actual_w, 3), dtype=np.uint8)
    div_text = "--- PROCESSED DELIVERABLES ---"
    text_size = cv2.getTextSize(div_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
    text_x = max(10, (actual_w - text_size[0]) // 2)
    cv2.putText(
        divider, div_text, (text_x, 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA
    )

    card = np.vstack([row1, divider, row2])
    return card

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
    """Segment any terrestrial foreground (mountains, trees, buildings, railings, balconies, roofs) from celestial sky.

    Uses adaptive night-sky luminance profiling with topological hole-filling and morphological
    structure reconstruction. Handles overhead structures (roofs, eaves, tree branches) and complex
    foregrounds without cutting dark lines or false chasms into shadowed ground.
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
    no_stars = cv2.morphologyEx(norm_gray, cv2.MORPH_OPEN, star_kernel)
    smooth = cv2.GaussianBlur(no_stars, (25, 25), 0)

    # Adaptive celestial sky baseline: the lowest 15-20% smooth luminance represents the open night sky
    sky_baseline = float(np.percentile(smooth, 15))
    sky_std = float(np.std(smooth[smooth < np.percentile(smooth, 30)]))

    if foreground_type == "silhouette":
        # Terrain is silhouetted against a brighter twilight sky
        thresh = sky_baseline + max(0.04, sky_std * 2.0)
        coarse = (smooth < thresh).astype(np.uint8) * 255
    else:
        # Default / Illuminated: celestial sky is the dark floor; terrestrial elements are above threshold
        thresh = sky_baseline + max(0.04, sky_std * 3.5)
        coarse = (smooth > thresh).astype(np.uint8) * 255

    # Morphological closing to bridge railings, window frames, cables, and terrain crevices
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
    closed = cv2.morphologyEx(coarse, cv2.MORPH_CLOSE, kernel_close)

    # Topological hole-filling: fill dark shadow holes completely enclosed within terrestrial foreground
    contours, hier = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled = closed.copy()
    if hier is not None:
        for cnt, h_info in zip(contours, hier[0]):
            # If contour is a child hole (enclosed within an outer foreground boundary)
            if h_info[3] >= 0:
                area = cv2.contourArea(cnt)
                # Fill holes up to 15% of the entire frame area (shadows on floors, dark walls, etc.)
                if area < (w * h * 0.15):
                    cv2.drawContours(filled, [cnt], -1, 255, -1)

    # Topological ground prior: ensure the bottom 8% is solid terrestrial floor
    filled[int(h * 0.92):, :] = 255

    # Edge-preserving guided filter refinement on normalized grayscale
    coarse_float = filled.astype(np.float32) / 255.0
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

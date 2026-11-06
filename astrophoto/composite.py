"""
composite.py - Generalized horizon segmentation, edge-guided blending, and comparison cards.

Supports any terrestrial landscape:
- Mountains, cliffs, and hills
- Trees, forests, and fine foliage
- Architecture, rooflines, towers, and city skylines
- Plains, desert dunes, beaches, and water horizons
- Both illuminated foregrounds and dark silhouette horizons
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
import urllib.request
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


def get_segformer_model_path() -> Path:
    """Return local path to cached SegFormer B0 ONNX model, auto-downloading if missing."""
    cache_dir = Path.home() / ".cache" / "astrophoto" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "segformer_b0.onnx"
    if not model_path.exists() or model_path.stat().st_size < 10000000:
        print("  [Mask Engine] Downloading lightweight SegFormer ONNX model (~15MB)...")
        url = "https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/main/onnx/model.onnx"
        urllib.request.urlretrieve(url, str(model_path))
    return model_path


def predict_segformer_sky(img_bgr: np.ndarray, min_sky_coverage: float = 0.05) -> Optional[np.ndarray]:
    """Run SegFormer B0 ADE20K semantic segmentation to produce a high-confidence celestial sky probability map.

    SegFormer distinguishes sky (Class 2) from ceilings (Class 5), walls (Class 0), buildings (Class 1),
    floors (Class 3), and fences/railings (Class 32), eliminating false sky holes in shadowed ground
    and preserving overhanging roofs/eaves without hand-crafted heuristics.
    """
    try:
        import onnxruntime as ort
        model_path = get_segformer_model_path()
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

        h_orig, w_orig = img_bgr.shape[:2]
        u8 = np.clip(img_bgr * 255.0, 0, 255).astype(np.uint8) if img_bgr.dtype != np.uint8 else img_bgr
        rgb = cv2.cvtColor(u8, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        batch = np.expand_dims(((resized / 255.0 - mean) / std).transpose(2, 0, 1), axis=0)

        logits = session.run(["logits"], {"pixel_values": batch})[0]
        exp_logits = np.exp(logits[0] - np.max(logits[0], axis=0, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
        sky_prob_512 = probs[2]  # Class 2 = sky in ADE20K

        coverage = float(np.mean(sky_prob_512 > 0.40))
        if coverage >= min_sky_coverage:
            print(f"  [Mask Engine] SegFormer AI semantic segmentation active ({coverage*100:.1f}% sky coverage detected).")
            return cv2.resize(sky_prob_512, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        return None
    except Exception as e:
        print(f"  [Mask Engine] AI segmentation unavailable ({e}). Falling back to energy segmentation.")
        return None


def segment_horizon_mask(
    reference_image_bgr: np.ndarray,
    foreground_type: str = "auto",
    mask_engine: str = "auto",
    guided_radius: int = 21,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment any terrestrial foreground (mountains, trees, buildings, railings, balconies, roofs) from celestial sky.

    Supports:
    - 'auto': Evaluates SegFormer AI semantic segmentation. If structural sky is detected with high confidence,
              uses the AI semantic map. Otherwise falls back to adaptive landscape energy profiling.
    - 'ai': Enforces SegFormer AI semantic segmentation.
    - 'classic': Enforces classical adaptive energy/Otsu segmentation.
    Both modes are edge-refined at native camera resolution using a fast edge-preserving guided filter.
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

    sky_prob_full = None
    if mask_engine in ("auto", "ai"):
        sky_prob_full = predict_segformer_sky(img_float)

    if sky_prob_full is None:
        # Classical landscape energy segmentation (ideal for wilderness dark-sky mountains)
        print("  [Mask Engine] Using high-precision edge-preserving landscape energy segmentation.")
        smooth = cv2.GaussianBlur(gray, (15, 15), 0)
        otsu_th, _ = cv2.threshold((smooth[int(h * 0.20):, :] * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_val = otsu_th / 255.0

        if foreground_type == "silhouette":
            coarse_gnd = (smooth < (otsu_val * 1.5)).astype(np.float32)
        else:
            is_illuminated = float(smooth[int(h * 0.80):, :].mean()) >= float(smooth[:int(h * 0.25), :].mean())
            if is_illuminated:
                coarse_gnd = (smooth > (otsu_val * 0.5)).astype(np.float32)
            else:
                coarse_gnd = (smooth < (otsu_val * 1.5)).astype(np.float32)

        # Morphological closing
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        closed_gnd = cv2.morphologyEx((coarse_gnd * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)

        # Hole filling for shadow crevices
        contours, hier = cv2.findContours(closed_gnd, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        filled = closed_gnd.copy()
        if hier is not None:
            for cnt, h_info in zip(contours, hier[0]):
                if h_info[3] >= 0:
                    if cv2.contourArea(cnt) < (w * h * 0.15):
                        cv2.drawContours(filled, [cnt], -1, 255, -1)

        filled[int(h * 0.95):, :] = 255
        sky_prob_full = 1.0 - (filled.astype(np.float32) / 255.0)

    # Edge-preserving guided filter refinement at native resolution
    refined_sky = guided_filter(norm_gray, sky_prob_full, radius=guided_radius, eps=1e-3)
    mask_sky = np.expand_dims(refined_sky, axis=2)
    mask_fg = 1.0 - mask_sky
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

"""
landscape.py - Landscape selection and highlight-protected terrain fusion.

Implements:
1. Automatic selection of the brightest landscape exposure that is NOT overexposed.
2. Highlight-protected Mertens exposure fusion across well-exposed ground frames.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np
from .io import ImageMetadata


def analyze_ground_exposures(
    ground_aligned_images: List[np.ndarray],
    ground_mask: np.ndarray,
    metadata_list: List[ImageMetadata],
    highlight_thresh: float = 0.92,
) -> List[Tuple[int, float, float]]:
    """Analyze highlight clipping and average brightness in the ground region.

    Returns:
        List of tuples: (image_index, highlight_clip_percentage, mean_ground_luminance).
    """
    results = []
    g_mask_2d = ground_mask[:, :, 0] > 0.5
    total_ground_px = float(np.sum(g_mask_2d))

    for i, img in enumerate(ground_aligned_images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g_px = gray[g_mask_2d]
        if total_ground_px > 0:
            clip_pct = float(np.sum(g_px >= highlight_thresh)) / total_ground_px * 100.0
            mean_lum = float(np.mean(g_px))
        else:
            clip_pct = 0.0
            mean_lum = float(np.mean(gray))
        results.append((i, clip_pct, mean_lum))

    return results


def select_best_ground_exposure(
    ground_aligned_images: List[np.ndarray],
    ground_mask: np.ndarray,
    metadata_list: List[ImageMetadata],
    max_clip_allowed_pct: float = 0.50,
) -> Tuple[int, np.ndarray]:
    """Select the single brightest landscape exposure that does NOT exceed highlight clipping limits.

    Args:
        ground_aligned_images: List of float32 BGR images.
        ground_mask: Mask of terrain (1.0 = ground).
        metadata_list: List of image metadata.
        max_clip_allowed_pct: Maximum allowable percentage of blown pixels in ground.

    Returns:
        Tuple of (selected_index, selected_ground_image).
    """
    stats = analyze_ground_exposures(ground_aligned_images, ground_mask, metadata_list)

    print("\n  [Landscape Exposure Analysis]")
    print(f"  {'Idx':<4} {'Filename':<28} {'Shutter':<14} {'Highlight Clip%':<18} {'Mean Lum':<10} {'Status'}")
    print("  " + "-" * 82)

    qualifying_indices = []
    for idx, clip_pct, mean_lum in stats:
        meta = metadata_list[idx]
        is_safe = clip_pct <= max_clip_allowed_pct
        status = "PASSED (Safe)" if is_safe else f"OVEREXPOSED ({clip_pct:.2f}% clipped)"
        print(f"  {idx+1:<4} {meta.filename:<28} {meta.shutter_str:<14} {clip_pct:<18.2f} {mean_lum:<10.3f} {status}")
        if is_safe:
            qualifying_indices.append(idx)

    # Pick the brightest qualifying exposure (or the least clipped if none qualify)
    if qualifying_indices:
        # Sort qualifying by mean luminance descending
        best_idx = max(qualifying_indices, key=lambda i: stats[i][2])
    else:
        print("  [Warning] All exposures exceeded threshold. Choosing the exposure with lowest clipping.")
        best_idx = min(range(len(stats)), key=lambda i: stats[i][1])

    best_meta = metadata_list[best_idx]
    print(f"  -> Selected optimal landscape exposure #{best_idx+1}: {best_meta.filename} ({best_meta.shutter_str})\n")
    return best_idx, ground_aligned_images[best_idx]


def fuse_ground_mertens(
    ground_aligned_images: List[np.ndarray],
    ground_mask: np.ndarray,
    metadata_list: List[ImageMetadata],
    max_clip_allowed_pct: float = 0.50,
    contrast_weight: float = 1.2,
    saturation_weight: float = 1.0,
    exposure_weight: float = 1.0,
) -> np.ndarray:
    """Fuse low-to-medium exposures for the ground using Mertens fusion, excluding overexposed frames."""
    stats = analyze_ground_exposures(ground_aligned_images, ground_mask, metadata_list)

    # Keep all frames that do not have severe highlight clipping
    usable_images = []
    usable_names = []
    for idx, clip_pct, _ in stats:
        if clip_pct <= max_clip_allowed_pct:
            u8 = np.clip(ground_aligned_images[idx] * 255.0, 0, 255).astype(np.uint8)
            usable_images.append(u8)
            usable_names.append(metadata_list[idx].filename)

    # Fallback to first 2-3 lowest exposures if filtering was too aggressive
    if len(usable_images) < 2:
        cutoff = max(2, len(ground_aligned_images) - 1)
        usable_images = [
            np.clip(ground_aligned_images[i] * 255.0, 0, 255).astype(np.uint8)
            for i in range(cutoff)
        ]
        usable_names = [metadata_list[i].filename for i in range(cutoff)]

    print(f"  [Ground Fusion] Blending {len(usable_images)} non-overexposed exposures: {', '.join(usable_names)}")
    fusion = cv2.createMergeMertens(
        contrast_weight=contrast_weight,
        saturation_weight=saturation_weight,
        exposure_weight=exposure_weight,
    )
    result = fusion.process(usable_images)
    return np.clip(result, 0.0, 1.0).astype(np.float32)

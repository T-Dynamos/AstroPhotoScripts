"""
Unit tests for AstroPhotoScripts pipeline components:
- Sky cropping and bounding box calculation
- Cropped star alignment
- Sky enhancement (black/white balancing and contrast boost)
- Ground highlight protection and Mertens fusion
- Seamless edge-guided blending
"""

import numpy as np
import pytest
from astrophoto.alignment import get_sky_crop_bbox, extract_and_align_sky_crops
from astrophoto.composite import blend_composite, guided_filter
from astrophoto.sky import enhance_sky_crop, stack_sky_exposures


def test_get_sky_crop_bbox():
    # Synthetic mask where top half has sky
    h, w = 200, 300
    mask_sky = np.zeros((h, w, 1), dtype=np.float32)
    mask_sky[20:100, 30:250] = 1.0

    ymin, ymax, xmin, xmax = get_sky_crop_bbox(mask_sky, (h, w), margin=10)
    assert ymin <= 20
    assert ymax >= 100
    assert xmin <= 30
    assert xmax >= 250
    assert ymax <= h
    assert xmax <= w


def test_extract_and_align_sky_crops():
    h, w = 128, 128
    mask_sky = np.zeros((h, w, 1), dtype=np.float32)
    mask_sky[:80, :] = 1.0

    # Two frames with a small star shifted by 2 pixels
    f1 = np.full((h, w, 3), 0.05, dtype=np.float32)
    f2 = np.full((h, w, 3), 0.05, dtype=np.float32)
    # Put bright star pinpoints
    f1[40, 40] = 1.0
    f1[40, 41] = 0.8
    f2[40, 42] = 1.0
    f2[40, 43] = 0.8

    crops, bbox, shifts = extract_and_align_sky_crops([f1, f2], mask_sky, ref_idx=0)
    assert len(crops) == 2
    assert bbox[0] >= 0
    assert len(shifts) == 2
    assert shifts[0] == (0.0, 0.0)


def test_enhance_sky_crop():
    h, w = 100, 100
    crop = np.full((h, w, 3), 0.08, dtype=np.float32)
    # Add star points
    crop[50, 50] = 0.95
    crop[30, 70] = 0.80

    mask_crop = np.ones((h, w, 1), dtype=np.float32)
    enhanced = enhance_sky_crop(
        crop,
        mask_crop,
        beta=28.0,
        target_black=0.035,
        contrast_boost=1.35,
        denoise=False,
    )

    assert enhanced.shape == (h, w, 3)
    assert enhanced.min() >= 0.02
    assert enhanced.max() <= 1.0
    # Background should sit close to target_black
    p10 = float(np.percentile(enhanced, 10))
    assert 0.02 <= p10 <= 0.06
    # Star cores should be preserved
    assert enhanced[50, 50].max() > 0.80


def test_blend_composite():
    h, w = 100, 100
    ground = np.full((h, w, 3), 0.2, dtype=np.float32)
    sky = np.full((h, w, 3), 0.8, dtype=np.float32)

    mask_sky = np.zeros((h, w, 1), dtype=np.float32)
    mask_sky[:50, :] = 1.0
    mask_ground = 1.0 - mask_sky

    comp = blend_composite(ground, sky, mask_ground, mask_sky)
    assert comp.shape == (h, w, 3)
    assert np.allclose(comp[:40, :], 0.8, atol=1e-4)
    assert np.allclose(comp[60:, :], 0.2, atol=1e-4)

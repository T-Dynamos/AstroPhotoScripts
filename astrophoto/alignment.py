"""
alignment.py - Dual-alignment engine for landscape astrophotography.

Performs:
1. Ground Alignment: Sub-pixel ECC image registration on terrestrial features
   with coarse-to-fine multi-scale acceleration and ORB fallback.
2. Sky Alignment: Frequency-domain phase correlation on celestial star fields
   to calculate and correct Earth rotation drift, with ground isolation to prevent
   shifted terrestrial silhouettes from leaking into the starry sky.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np


def align_ground_ecc(
    ref_bgr: np.ndarray,
    src_bgr: np.ndarray,
    motion_type: int = cv2.MOTION_TRANSLATION,
    coarse_scale: float = 0.25,
    max_iters_coarse: int = 40,
    max_iters_fine: int = 30,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Align source image to reference image using coarse-to-fine ECC maximization."""
    h, w = ref_bgr.shape[:2]
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY) if ref_bgr.ndim == 3 else ref_bgr
    src_gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY) if src_bgr.ndim == 3 else src_bgr

    if ref_gray.dtype != np.uint8:
        ref_u8 = np.clip(ref_gray * 255.0, 0, 255).astype(np.uint8)
        src_u8 = np.clip(src_gray * 255.0, 0, 255).astype(np.uint8)
    else:
        ref_u8 = ref_gray
        src_u8 = src_gray

    warp = np.eye(2, 3, dtype=np.float32)

    try:
        if coarse_scale < 1.0:
            ref_small = cv2.resize(ref_u8, (0, 0), fx=coarse_scale, fy=coarse_scale)
            src_small = cv2.resize(src_u8, (0, 0), fx=coarse_scale, fy=coarse_scale)
            crit_coarse = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                max_iters_coarse,
                eps * 10,
            )
            _, warp = cv2.findTransformECC(ref_small, src_small, warp, motion_type, crit_coarse)
            warp[0, 2] /= coarse_scale
            warp[1, 2] /= coarse_scale

        crit_fine = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            max_iters_fine,
            eps,
        )
        cc, warp = cv2.findTransformECC(ref_u8, src_u8, warp, motion_type, crit_fine)

        aligned = cv2.warpAffine(
            src_bgr,
            warp,
            (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
        return aligned, warp, float(cc)

    except cv2.error:
        try:
            orb = cv2.ORB_create(nfeatures=2500)
            kp1, des1 = orb.detectAndCompute(ref_u8, None)
            kp2, des2 = orb.detectAndCompute(src_u8, None)
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(matcher.match(des1, des2), key=lambda x: x.distance)[:100]

            pts_ref = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts_src = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            warp_orb, inliers = cv2.estimateAffinePartial2D(pts_src, pts_ref)
            if warp_orb is not None:
                aligned = cv2.warpAffine(
                    src_bgr, warp_orb, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT
                )
                return aligned, warp_orb, 0.60
        except Exception:
            pass

        return src_bgr.copy(), np.eye(2, 3, dtype=np.float32), 0.0


def align_ground_dataset(
    images_bgr: List[np.ndarray],
    ref_idx: int = 0,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Align all input frames to a reference frame based on static terrestrial landscape."""
    aligned = []
    warps = []
    ref_img = images_bgr[ref_idx]

    for i, img in enumerate(images_bgr):
        if i == ref_idx:
            aligned.append(img.copy())
            warps.append(np.eye(2, 3, dtype=np.float32))
            print(f"  [Ground Alignment] Image {i+1}/{len(images_bgr)}: Reference template (fixed)")
        else:
            al, w, cc = align_ground_ecc(ref_img, img)
            aligned.append(al)
            warps.append(w)
            dx, dy = w[0, 2], w[1, 2]
            print(f"  [Ground Alignment] Image {i+1}/{len(images_bgr)}: dx={dx:+.2f}px, dy={dy:+.2f}px (corr: {cc:.4f})")

    return aligned, warps


def compute_celestial_star_drift(
    images_bgr: List[np.ndarray],
    ref_idx: int = 0,
    sky_crop_fraction: float = 0.55,
    mask_sky: Optional[np.ndarray] = None,
) -> List[Tuple[float, float]]:
    """Calculate celestial star shift (dx, dy) across exposures via frequency-domain phase correlation.

    Uses a Difference-of-Gaussians (DoG) star bandpass filter with star amplitude normalization
    to handle both linear RAW/DNG and gamma-compressed JPEG exposures across varying shutter speeds.
    Ensures terrestrial foreground structures (mountains, fences, railings, roofs) are excluded
    so phase correlation tracks pure celestial star rotation without locking onto static ground.
    """
    ref_img = images_bgr[ref_idx]
    h, w = ref_img.shape[:2]

    # Determine safe sky height strictly above any terrestrial horizon
    if mask_sky is not None:
        m2d = mask_sky[:, :, 0] if mask_sky.ndim == 3 else mask_sky
        # Find rows where foreground starts
        fg_indices = np.where(m2d < 0.5)[0]
        if len(fg_indices) > 0:
            first_fg_row = int(fg_indices.min())
            # Keep a 5% safety margin above the highest terrestrial object
            sky_h = max(int(h * 0.25), first_fg_row - int(h * 0.03))
        else:
            sky_h = int(h * sky_crop_fraction)
    else:
        sky_h = int(h * sky_crop_fraction)

    sky_h = min(sky_h, int(h * 0.85))

    def isolate_stars_dog(img_bgr: np.ndarray) -> np.ndarray:
        crop = img_bgr[:sky_h, :]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        if gray.dtype != np.float32:
            gray = gray.astype(np.float32)
            if gray.max() > 1.5:
                gray /= 255.0

        # Difference-of-Gaussians bandpass filter: isolates point sources, removes sky glow
        blur_fine = cv2.GaussianBlur(gray, (3, 3), 0)
        blur_wide = cv2.GaussianBlur(gray, (25, 25), 0)
        dog = np.maximum(blur_fine - blur_wide, 0.0)

        # Normalize star amplitude across exposures so different shutter speeds match
        p99_9 = float(np.percentile(dog, 99.9))
        if p99_9 > 1e-6:
            dog_norm = np.clip(dog / p99_9, 0.0, 1.0)
        else:
            dog_norm = dog

        # Suppress residual low-level sensor thermal noise floor
        dog_norm[dog_norm < 0.12] = 0.0
        return dog_norm.astype(np.float32)

    ref_stars = isolate_stars_dog(ref_img)
    hann = cv2.createHanningWindow((w, sky_h), cv2.CV_32F)
    shifts: List[Tuple[float, float]] = []

    for i, img in enumerate(images_bgr):
        if i == ref_idx:
            shifts.append((0.0, 0.0))
            print(f"  [Star Drift] Image {i+1}/{len(images_bgr)}: Celestial origin (0.0px, 0.0px)")
            continue

        cur_stars = isolate_stars_dog(img)
        (dx, dy), response = cv2.phaseCorrelate(ref_stars, cur_stars, hann)

        # Sanity check: if response is low, fallback to ORB matching in sky
        if response < 0.15:
            try:
                orb = cv2.ORB_create(nfeatures=1500)
                u8_ref = (ref_stars * 255).astype(np.uint8)
                u8_cur = (cur_stars * 255).astype(np.uint8)
                kp1, des1 = orb.detectAndCompute(u8_ref, None)
                kp2, des2 = orb.detectAndCompute(u8_cur, None)
                if des1 is not None and des2 is not None and len(kp1) >= 10 and len(kp2) >= 10:
                    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = sorted(matcher.match(des1, des2), key=lambda x: x.distance)[:80]
                    if len(matches) >= 6:
                        pts_r = np.float32([kp1[m.queryIdx].pt for m in matches])
                        pts_c = np.float32([kp2[m.trainIdx].pt for m in matches])
                        diffs = pts_r - pts_c
                        dx_med, dy_med = float(np.median(diffs[:, 0])), float(np.median(diffs[:, 1]))
                        dx, dy, response = dx_med, dy_med, 0.35
            except Exception:
                pass

        shifts.append((float(dx), float(dy)))
        print(f"  [Star Drift] Image {i+1}/{len(images_bgr)}: dx={dx:+.2f}px, dy={dy:+.2f}px (response: {response:.3f})")

    return shifts


def warp_sky_to_stars(
    images_bgr: List[np.ndarray],
    star_shifts: List[Tuple[float, float]],
    mask_sky: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Warp images to cancel celestial star drift with ground-isolation to prevent dark horizon lines.

    When stars are aligned, terrestrial objects drift. If the ground is not isolated prior
    to warping, shifted ground silhouettes pollute the sky above the horizon, causing a dark line.
    By replacing the ground with sky baseline values before warping, the sky layer contains
    pure celestial data everywhere near the horizon.
    """
    h, w = images_bgr[0].shape[:2]
    warped = []

    for img, (dx, dy) in zip(images_bgr, star_shifts):
        if mask_sky is not None:
            # Isolate sky: replace ground with median sky baseline
            sky_isolated = img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pure_sky_pixels = gray[mask_sky[:, :, 0] > 0.8]
            sky_median = float(np.median(pure_sky_pixels)) if len(pure_sky_pixels) > 0 else 0.04
            sky_isolated[mask_sky[:, :, 0] < 0.2] = [sky_median, sky_median, sky_median]
            source_img = sky_isolated
        else:
            source_img = img

        M = np.float32([[1, 0, dx], [0, 1, dy]])
        w_img = cv2.warpAffine(
            source_img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
        warped.append(w_img)

    return warped

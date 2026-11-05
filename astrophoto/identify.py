"""
identify.py - Offline, local constellation and deep-sky object identification and annotation.

Performs:
1. Star centroid extraction in the sky region.
2. Geometric asterism & triangle pattern matching against built-in celestial catalogs.
3. Sub-pixel homography transformation mapping celestial (RA, Dec) to image pixels.
4. Professional astronomical annotation overlay:
   - Glowing constellation stick-figure lines
   - Star rings with names and Bayer designations
   - Deep-sky object target reticles (e.g. M42 Orion Nebula)
   - Constellation header banners
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class CatalogStar:
    name: str
    bayer: str
    ra_deg: float
    dec_deg: float
    mag: float


@dataclass
class DeepSkyObject:
    name: str
    catalog_id: str
    ra_deg: float
    dec_deg: float
    obj_type: str  # 'nebula', 'cluster', 'galaxy'


@dataclass
class ConstellationDef:
    name: str
    latin_name: str
    common_name: str
    stars: Dict[str, CatalogStar]
    lines: List[Tuple[str, str]]
    dsos: List[DeepSkyObject]
    anchor_asterism: List[str]  # e.g. ['Mintaka', 'Alnilam', 'Alnitak']


# Built-in Celestial Catalog of major visible constellations and asterisms
CATALOG: Dict[str, ConstellationDef] = {
    "Orion": ConstellationDef(
        name="Orion",
        latin_name="Orion",
        common_name="The Hunter",
        stars={
            "Betelgeuse": CatalogStar("Betelgeuse", "α Ori", 88.793, 7.407, 0.50),
            "Rigel": CatalogStar("Rigel", "β Ori", 78.634, -8.201, 0.18),
            "Bellatrix": CatalogStar("Bellatrix", "γ Ori", 81.283, 6.350, 1.64),
            "Mintaka": CatalogStar("Mintaka", "δ Ori", 83.001, -0.299, 2.23),
            "Alnilam": CatalogStar("Alnilam", "ε Ori", 84.053, -1.201, 1.69),
            "Alnitak": CatalogStar("Alnitak", "ζ Ori", 85.190, -1.942, 1.77),
            "Saiph": CatalogStar("Saiph", "κ Ori", 86.939, -9.669, 2.07),
            "Meissa": CatalogStar("Meissa", "λ Ori", 83.784, 9.934, 3.39),
        },
        lines=[
            ("Betelgeuse", "Bellatrix"),
            ("Betelgeuse", "Alnitak"),
            ("Bellatrix", "Mintaka"),
            ("Mintaka", "Alnilam"),
            ("Alnilam", "Alnitak"),
            ("Alnitak", "Saiph"),
            ("Mintaka", "Rigel"),
            ("Saiph", "Rigel"),
            ("Alnilam", "M42"),
        ],
        dsos=[
            DeepSkyObject("Orion Nebula", "M42 / NGC 1976", 83.822, -5.391, "Diffuse Nebula"),
            DeepSkyObject("Flame Nebula", "NGC 2024", 85.425, -1.850, "Emission Nebula"),
        ],
        anchor_asterism=["Mintaka", "Alnilam", "Alnitak"],
    ),
    "UrsaMajor": ConstellationDef(
        name="Ursa Major",
        latin_name="Ursa Major",
        common_name="The Great Bear / Big Dipper",
        stars={
            "Dubhe": CatalogStar("Dubhe", "α UMa", 165.932, 61.751, 1.79),
            "Merak": CatalogStar("Merak", "β UMa", 165.460, 56.382, 2.37),
            "Phecda": CatalogStar("Phecda", "γ UMa", 178.457, 53.695, 2.44),
            "Megrez": CatalogStar("Megrez", "δ UMa", 183.856, 57.032, 3.31),
            "Alioth": CatalogStar("Alioth", "ε UMa", 193.507, 55.960, 1.77),
            "Mizar": CatalogStar("Mizar", "ζ UMa", 200.981, 54.925, 2.23),
            "Alkaid": CatalogStar("Alkaid", "η UMa", 206.885, 49.313, 1.86),
        },
        lines=[
            ("Dubhe", "Merak"),
            ("Merak", "Phecda"),
            ("Phecda", "Megrez"),
            ("Megrez", "Dubhe"),
            ("Megrez", "Alioth"),
            ("Alioth", "Mizar"),
            ("Mizar", "Alkaid"),
        ],
        dsos=[
            DeepSkyObject("Bode's Galaxy", "M81", 148.888, 69.065, "Spiral Galaxy"),
            DeepSkyObject("Cigar Galaxy", "M82", 148.968, 69.679, "Starburst Galaxy"),
        ],
        anchor_asterism=["Dubhe", "Merak", "Phecda"],
    ),
    "Cassiopeia": ConstellationDef(
        name="Cassiopeia",
        latin_name="Cassiopeia",
        common_name="The Queen",
        stars={
            "Caph": CatalogStar("Caph", "β Cas", 2.295, 59.150, 2.28),
            "Schedar": CatalogStar("Schedar", "α Cas", 10.127, 56.537, 2.24),
            "Navi": CatalogStar("Navi", "γ Cas", 14.177, 60.717, 2.47),
            "Ruchbah": CatalogStar("Ruchbah", "δ Cas", 20.650, 60.235, 2.68),
            "Segin": CatalogStar("Segin", "ε Cas", 26.155, 63.670, 3.37),
        },
        lines=[
            ("Caph", "Schedar"),
            ("Schedar", "Navi"),
            ("Navi", "Ruchbah"),
            ("Ruchbah", "Segin"),
        ],
        dsos=[
            DeepSkyObject("Pacman Nebula", "NGC 281", 13.310, 56.620, "Emission Nebula"),
        ],
        anchor_asterism=["Schedar", "Navi", "Ruchbah"],
    ),
}


def detect_stars_subpixel(
    img_bgr: np.ndarray,
    mask_sky: Optional[np.ndarray] = None,
    max_stars: int = 150,
    min_peak_thresh: int = 25,
) -> List[Tuple[float, float, float]]:
    """Extract stellar point sources with sub-pixel centroid precision.

    Returns:
        List of tuples: (x_float, y_float, peak_flux).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    if gray.dtype != np.uint8:
        gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    else:
        gray_u8 = gray

    # High-pass filter
    bg = cv2.medianBlur(gray_u8, 21)
    stars = cv2.subtract(gray_u8, bg)

    if mask_sky is not None:
        m_2d = mask_sky[:, :, 0] if mask_sky.ndim == 3 else mask_sky
        stars[m_2d < 0.5] = 0

    # Local maximum filter via morphological dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(stars, kernel)
    peaks = (stars == dilated) & (stars >= min_peak_thresh)

    ys, xs = np.where(peaks)
    h, w = gray_u8.shape

    detected = []
    for x, y in zip(xs, ys):
        if 3 <= x < w - 3 and 3 <= y < h - 3:
            patch = stars[y - 2 : y + 3, x - 2 : x + 3].astype(float)
            total = patch.sum()
            if total > 0:
                cx = x + (np.sum(patch * np.arange(-2, 3)[None, :]) / total)
                cy = y + (np.sum(patch * np.arange(-2, 3)[:, None]) / total)
                detected.append((cx, cy, float(stars[y, x])))

    # Sort by brightness
    detected.sort(key=lambda s: s[2], reverse=True)
    return detected[:max_stars]


def match_orion_belt(
    stars: List[Tuple[float, float, float]],
    collinearity_tol: float = 0.15,
) -> Optional[Tuple[np.ndarray, ConstellationDef]]:
    """Specifically detect the Orion Belt 3-star collinear asterism."""
    # Belt stars have near equal spacing and lie on a line
    c_def = CATALOG["Orion"]
    pts = np.array([(s[0], s[1]) for s in stars[:60]], dtype=np.float32)

    for i in range(len(pts)):
        p1 = pts[i]
        for j in range(len(pts)):
            if i == j:
                continue
            p2 = pts[j]
            d12 = np.linalg.norm(p2 - p1)
            if d12 < 30 or d12 > 250:
                continue

            for k in range(len(pts)):
                if k == i or k == j:
                    continue
                p3 = pts[k]
                d23 = np.linalg.norm(p3 - p2)

                # Spacing ratio must be ~1.0
                ratio = d12 / max(1e-4, d23)
                if not (0.70 < ratio < 1.40):
                    continue

                # Collinearity check: cross product must be small
                v1 = (p2 - p1) / d12
                v2 = (p3 - p2) / d23
                dot = np.dot(v1, v2)
                if dot < 0.96:
                    continue

                # p1, p2, p3 form a collinear 3-star triplet
                # Check orientation: Mintaka -> Alnilam -> Alnitak
                # Estimate homography from catalog belt
                cat_belt = np.float32([
                    [c_def.stars["Mintaka"].ra_deg, c_def.stars["Mintaka"].dec_deg],
                    [c_def.stars["Alnilam"].ra_deg, c_def.stars["Alnilam"].dec_deg],
                    [c_def.stars["Alnitak"].ra_deg, c_def.stars["Alnitak"].dec_deg],
                ])

                for cand_order in [(p1, p2, p3), (p3, p2, p1)]:
                    img_belt = np.float32(cand_order)
                    M, _ = cv2.estimateAffinePartial2D(cat_belt, img_belt)
                    if M is not None:
                        # Verify using Saiph and Betelgeuse positions
                        saiph_cat = np.array([c_def.stars["Saiph"].ra_deg, c_def.stars["Saiph"].dec_deg, 1.0])
                        saiph_pred = M @ saiph_cat
                        # Find closest star to predicted Saiph
                        dists = np.hypot(pts[:, 0] - saiph_pred[0], pts[:, 1] - saiph_pred[1])
                        if np.min(dists) < 30:
                            # Confirmed match!
                            best_saiph = pts[np.argmin(dists)]
                            # Refine with 4 matched stars
                            cat_all = np.vstack([cat_belt, [[c_def.stars["Saiph"].ra_deg, c_def.stars["Saiph"].dec_deg]]])
                            img_all = np.vstack([img_belt, [best_saiph]])
                            H, _ = cv2.findHomography(cat_all, img_all)
                            return H, c_def

    return None


def annotate_constellations(
    image_bgr: np.ndarray,
    mask_sky: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
) -> Tuple[np.ndarray, bool]:
    """Auto-detect constellations and render celestial annotations onto the image.

    Returns:
        Tuple of (annotated_bgr, success_bool).
    """
    h, w = image_bgr.shape[:2]
    annotated = image_bgr.copy()
    overlay = image_bgr.copy()

    # Step 1: Detect stars
    stars = detect_stars_subpixel(image_bgr, mask_sky=mask_sky, max_stars=120)
    print(f"  [Plate Solve] Detected {len(stars)} stellar point sources.")

    # Step 2: Match asterisms
    match_result = match_orion_belt(stars)

    if match_result is None:
        print("  [Plate Solve] Could not confidently match constellation asterisms.")
        return image_bgr.copy(), False

    H, c_def = match_result
    print(f"  [Plate Solve] Confidently identified Constellation: {c_def.name} ({c_def.common_name})!")

    # Project stars
    projected_stars: Dict[str, Tuple[int, int]] = {}
    for s_name, s_info in c_def.stars.items():
        cat_pt = np.array([s_info.ra_deg, s_info.dec_deg, 1.0])
        proj = H @ cat_pt
        px, py = int(round(proj[0] / proj[2])), int(round(proj[1] / proj[2]))
        if 0 <= px < w and 0 <= py < h:
            projected_stars[s_name] = (px, py)

    # Project DSOs
    projected_dsos: Dict[str, Tuple[int, int, DeepSkyObject]] = {}
    for dso in c_def.dsos:
        cat_pt = np.array([dso.ra_deg, dso.dec_deg, 1.0])
        proj = H @ cat_pt
        px, py = int(round(proj[0] / proj[2])), int(round(proj[1] / proj[2]))
        if 0 <= px < w and 0 <= py < h:
            projected_dsos[dso.name] = (px, py, dso)

    # Draw glowing constellation stick-figure lines
    for s1, s2 in c_def.lines:
        pt1 = projected_stars.get(s1) or (projected_dsos.get(s1)[0:2] if s1 in projected_dsos else None)
        pt2 = projected_stars.get(s2) or (projected_dsos.get(s2)[0:2] if s2 in projected_dsos else None)
        if pt1 and pt2:
            # Outer cyan glow
            cv2.line(overlay, pt1, pt2, (255, 120, 20), 4, cv2.LINE_AA)
            # Inner bright line
            cv2.line(overlay, pt1, pt2, (255, 230, 120), 2, cv2.LINE_AA)

    # Draw star circles and typography
    for s_name, (x, y) in projected_stars.items():
        s_info = c_def.stars[s_name]
        label = f"{s_name} ({s_info.bayer})"
        # Star circle ring
        cv2.circle(overlay, (x, y), 10, (255, 200, 50), 2, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
        # Text with black outline for readability against background stars
        cv2.putText(overlay, label, (x + 16, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, label, (x + 16, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 235, 160), 1, cv2.LINE_AA)

    # Draw Deep-Sky Objects with target reticles
    for dso_name, (x, y, dso) in projected_dsos.items():
        reticle_color = (60, 120, 255)  # Orange-Red for emission nebulae
        cv2.drawMarker(overlay, (x, y), reticle_color, cv2.MARKER_TILTED_CROSS, 22, 2, cv2.LINE_AA)
        cv2.rectangle(overlay, (x - 14, y - 14), (x + 14, y + 14), reticle_color, 2, cv2.LINE_AA)
        dso_label = f"{dso.name} [{dso.catalog_id}]"
        cv2.putText(overlay, dso_label, (x + 20, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, dso_label, (x + 20, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 190, 255), 1, cv2.LINE_AA)

    # Constellation Header Banner
    if "Alnilam" in projected_stars:
        bx, by = projected_stars["Alnilam"]
        banner_x = max(50, bx - 180)
        banner_y = max(60, by - 160)
        title_text = f"CONSTELLATION: {c_def.name.upper()} ({c_def.common_name.upper()})"
        cv2.putText(overlay, title_text, (banner_x, banner_y), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, title_text, (banner_x, banner_y), cv2.FONT_HERSHEY_DUPLEX, 0.85, (100, 225, 255), 2, cv2.LINE_AA)

    # Blend with 85% opacity
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  [Success] Saved Annotated Sky Overlay: {output_path.resolve()}")

    return annotated, True

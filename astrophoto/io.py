"""
io.py - Image ingestion supporting RAW/DNG, JPEG, PNG, TIFF and EXIF parsing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import ExifTags, Image
import tifffile

try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

RAW_EXTENSIONS = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".pef", ".raf"}
STANDARD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
ALL_SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | STANDARD_EXTENSIONS


@dataclass
class ImageMetadata:
    path: Path
    filename: str
    width: int
    height: int
    is_raw: bool
    exposure_time: Optional[float]  # in seconds
    iso: Optional[int]
    f_number: Optional[float]
    camera_model: Optional[str]
    date_time: Optional[str]
    shadow_clip_pct: float
    highlight_clip_pct: float
    mean_luminance: float

    @property
    def shutter_str(self) -> str:
        if self.exposure_time is None:
            return "N/A"
        if self.exposure_time < 1.0:
            recip = round(1.0 / self.exposure_time)
            return f"1/{recip}s ({self.exposure_time:.4f}s)"
        return f"{self.exposure_time:.1f}s"


def _parse_rational(val) -> Optional[float]:
    if val is None:
        return None
    try:
        if hasattr(val, "numerator") and hasattr(val, "denominator"):
            return float(val.numerator) / float(val.denominator) if val.denominator != 0 else float(val.numerator)
        if isinstance(val, tuple) and len(val) == 2:
            return float(val[0]) / float(val[1]) if val[1] != 0 else float(val[0])
        return float(val)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def extract_exif(path: Path) -> Dict[str, Union[float, int, str, None]]:
    """Extract exposure time, ISO, aperture, model, and timestamp from image EXIF."""
    info: Dict[str, Union[float, int, str, None]] = {
        "exposure_time": None,
        "iso": None,
        "f_number": None,
        "camera_model": None,
        "date_time": None,
    }
    try:
        with Image.open(path) as pil_img:
            raw_exif = pil_img.getexif()
            if raw_exif:
                exif_dict: Dict = {}
                for k, v in raw_exif.items():
                    name = ExifTags.TAGS.get(k, k)
                    exif_dict[name] = v

                for ifd_id in ExifTags.IFD:
                    try:
                        ifd = raw_exif.get_ifd(ifd_id)
                        for k, v in ifd.items():
                            name = ExifTags.TAGS.get(k, k)
                            exif_dict[name] = v
                    except Exception:
                        pass

                info["exposure_time"] = _parse_rational(exif_dict.get("ExposureTime"))
                info["f_number"] = _parse_rational(exif_dict.get("FNumber"))
                iso_val = exif_dict.get("ISOSpeedRatings") or exif_dict.get("PhotographicSensitivity")
                if iso_val is not None:
                    try:
                        info["iso"] = int(iso_val if isinstance(iso_val, (int, float)) else iso_val[0])
                    except (TypeError, IndexError, ValueError):
                        pass
                info["camera_model"] = exif_dict.get("Model")
                info["date_time"] = exif_dict.get("DateTimeOriginal") or exif_dict.get("DateTime")
    except Exception:
        # Fallback for DNG via tifffile if PIL couldn't parse
        try:
            with tifffile.TiffFile(str(path)) as tif:
                for page in tif.pages:
                    for tag in page.tags:
                        if tag.name == "ExposureTime" and info["exposure_time"] is None:
                            info["exposure_time"] = _parse_rational(tag.value)
                        elif tag.name in ("ISOSpeedRatings", "PhotographicSensitivity") and info["iso"] is None:
                            try:
                                info["iso"] = int(tag.value[0] if isinstance(tag.value, (tuple, list)) else tag.value)
                            except Exception:
                                pass
                        elif tag.name == "FNumber" and info["f_number"] is None:
                            info["f_number"] = _parse_rational(tag.value)
                        elif tag.name == "Model" and info["camera_model"] is None:
                            info["camera_model"] = str(tag.value)
                        elif tag.name in ("DateTimeOriginal", "DateTime") and info["date_time"] is None:
                            info["date_time"] = str(tag.value)
        except Exception:
            pass

    return info


def load_image(path: Path) -> Tuple[np.ndarray, ImageMetadata]:
    """Load an image file (DNG/RAW, JPEG, PNG, TIFF) as float32 BGR [0.0, 1.0].

    Returns:
        Tuple of (bgr_float32, metadata).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    is_raw = suffix in RAW_EXTENSIONS

    if is_raw:
        if not HAS_RAWPY:
            raise ImportError(
                f"rawpy is required to open RAW/DNG file '{path.name}'. "
                "Install it with: pip install rawpy"
            )
        with rawpy.imread(str(path)) as raw:
            # 16-bit demosaic with camera white balance and no artificial tone curve clipping
            rgb_16 = raw.postprocess(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=True,
                output_bps=16,
            )
            bgr_float = cv2.cvtColor(rgb_16, cv2.COLOR_RGB2BGR).astype(np.float32) / 65535.0
    else:
        # Standard format
        img_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img_raw is None:
            raise ValueError(f"Could not open image: {path}")

        if img_raw.ndim == 2:
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
        elif img_raw.shape[2] == 4:
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGRA2BGR)

        if img_raw.dtype == np.uint8:
            bgr_float = img_raw.astype(np.float32) / 255.0
        elif img_raw.dtype == np.uint16:
            bgr_float = img_raw.astype(np.float32) / 65535.0
        else:
            bgr_float = img_raw.astype(np.float32)

    bgr_float = np.clip(bgr_float, 0.0, 1.0)
    h, w = bgr_float.shape[:2]

    # Pixel dynamic range analysis
    gray = cv2.cvtColor(bgr_float, cv2.COLOR_BGR2GRAY)
    total_px = float(gray.size)
    shadow_pct = float(np.sum(gray <= 0.015)) / total_px * 100.0
    high_pct = float(np.sum(gray >= 0.985)) / total_px * 100.0
    mean_lum = float(gray.mean())

    exif = extract_exif(path)

    meta = ImageMetadata(
        path=path,
        filename=path.name,
        width=w,
        height=h,
        is_raw=is_raw,
        exposure_time=exif["exposure_time"],
        iso=exif["iso"],
        f_number=exif["f_number"],
        camera_model=exif["camera_model"],
        date_time=exif["date_time"],
        shadow_clip_pct=shadow_pct,
        highlight_clip_pct=high_pct,
        mean_luminance=mean_lum,
    )

    return bgr_float, meta


def discover_images(directory: Path, sort_by_exposure: bool = True) -> List[Path]:
    """Find all supported image and RAW files in a directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    found: List[Path] = []
    for ext in ALL_SUPPORTED_EXTENSIONS:
        for p in directory.glob(f"*{ext}"):
            name = p.name.lower()
            if any(term in name for term in ("output", "preview", "thumb", "test", "montage", "mask", "layer")):
                continue
            if p not in found:
                found.append(p)
        for p in directory.glob(f"*{ext.upper()}"):
            name = p.name.lower()
            if any(term in name for term in ("output", "preview", "thumb", "test", "montage", "mask", "layer")):
                continue
            if p not in found:
                found.append(p)

    return sorted(found)


def save_master_tiff(bgr_float: np.ndarray, path: Path) -> None:
    """Save 16-bit uncompressed RGB TIFF preserving high dynamic range output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    u16 = np.clip(bgr_float * 65535.0, 0, 65535).astype(np.uint16)
    rgb_16 = cv2.cvtColor(u16, cv2.COLOR_BGR2RGB)
    tifffile.imwrite(str(path), rgb_16, compression="zlib")


def save_preview_jpeg(bgr_float: np.ndarray, path: Path, quality: int = 95) -> None:
    """Save displayable 8-bit JPEG preview."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.clip(bgr_float * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), u8, [cv2.IMWRITE_JPEG_QUALITY, quality])

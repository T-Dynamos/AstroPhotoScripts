"""
pipeline.py - High-level pipeline coordinator for astrophotography stacking.
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import List, Optional
import cv2
import numpy as np

from . import enable_opencv_multithreading
from .alignment import (
    align_ground_dataset,
    compute_celestial_star_drift,
    extract_and_align_sky_crops,
    get_sky_crop_bbox,
    warp_sky_to_stars,
)
from .composite import blend_composite, create_comparison_card, segment_horizon_mask
from .identify import annotate_constellations
from .io import (
    ImageMetadata,
    discover_images,
    load_image,
    save_master_tiff,
    save_preview_jpeg,
)
from .landscape import fuse_ground_mertens, select_best_ground_exposure
from .sky import (
    denoise_sky_chroma,
    enhance_sky_crop,
    flatten_sky_background,
    stack_sky_exposures,
    stretch_sky_asinh,
)


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    ground_mode: str = "fuse"  # 'fuse' or 'select'
    foreground_type: str = "auto"  # 'auto', 'illuminated', or 'silhouette'
    mask_engine: str = "auto"  # 'auto' (SegFormer AI with energy fallback), 'ai', or 'classic'
    horizon_radius: int = 21
    max_ground_clip_pct: float = 8.0
    sky_stack_method: str = "weighted"  # 'weighted' or 'median'
    sky_stretch_beta: float = 28.0
    contrast_boost: float = 1.35
    target_black_floor: float = 0.035
    denoise_sky: bool = True
    save_intermediate_layers: bool = True
    annotate: bool = True
    ref_idx: Optional[int] = None


class AstroPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.threads = enable_opencv_multithreading(verbose=True)

    def print_summary_table(self, metadata_list: List[ImageMetadata]) -> None:
        print("\n" + "=" * 90)
        print(f"{'Idx':<4} {'Filename':<28} {'Shutter':<14} {'ISO':<8} {'Format':<8} {'Shadow%':<10} {'High%':<10}")
        print("-" * 90)
        for i, m in enumerate(metadata_list):
            fmt = "RAW/DNG" if m.is_raw else "Standard"
            iso_str = str(m.iso) if m.iso is not None else "N/A"
            print(
                f"{i+1:<4} {m.filename:<28} {m.shutter_str:<14} {iso_str:<8} {fmt:<8} "
                f"{m.shadow_clip_pct:<10.2f} {m.highlight_clip_pct:<10.2f}"
            )
        print("=" * 90 + "\n")

    def run(self) -> Path:
        start_time = time.time()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[1/6] Discovering images in: {self.config.input_dir.resolve()}")
        image_paths = discover_images(self.config.input_dir)
        if len(image_paths) < 2:
            raise ValueError(f"At least 2 images required for stacking. Found: {len(image_paths)}")

        print(f"  Found {len(image_paths)} valid images. Loading data & EXIF...")
        loaded = [load_image(p) for p in image_paths]
        images_bgr = [item[0] for item in loaded]
        metadata_list = [item[1] for item in loaded]

        # Verify dimensions
        ref_h, ref_w = images_bgr[0].shape[:2]
        for i, img in enumerate(images_bgr[1:], 1):
            if img.shape[:2] != (ref_h, ref_w):
                raise ValueError(
                    f"Image dimension mismatch: {metadata_list[0].filename} ({ref_w}x{ref_h}) vs "
                    f"{metadata_list[i].filename} ({img.shape[1]}x{img.shape[0]})."
                )

        # Sort exposures by shutter speed ascending
        paired = sorted(
            zip(images_bgr, metadata_list),
            key=lambda pair: (pair[1].exposure_time if pair[1].exposure_time is not None else 0.0, pair[1].filename)
        )
        images_bgr = [p[0] for p in paired]
        metadata_list = [p[1] for p in paired]

        self.print_summary_table(metadata_list)

        # Step 2: Ground alignment
        # Select reference frame: default to the best exposed frame (highest SNR and landscape texture)
        if self.config.ref_idx is not None and 0 <= self.config.ref_idx < len(images_bgr):
            ref_idx = self.config.ref_idx
        else:
            ref_idx = len(images_bgr) - 1

        print(f"[2/6] Performing sub-pixel ground ECC registration (Ref: #{ref_idx+1} {metadata_list[ref_idx].filename})...")
        ground_aligned, _ = align_ground_dataset(images_bgr, ref_idx=ref_idx)

        # Step 3: Horizon segmentation (general: mountains, trees, buildings, structures, plains)
        print("[3/6] Segmenting terrestrial landscape (trees, buildings, terrain) and starry sky along the horizon...")
        mask_ground, mask_sky = segment_horizon_mask(
            ground_aligned[ref_idx],
            foreground_type=self.config.foreground_type,
            mask_engine=self.config.mask_engine,
            guided_radius=self.config.horizon_radius,
        )

        # Step 4: Landscape layer processing
        print(f"[4/6] Processing landscape layer (mode: '{self.config.ground_mode}')...")
        if self.config.ground_mode == "select":
            _, ground_layer = select_best_ground_exposure(
                ground_aligned,
                mask_ground,
                metadata_list,
                max_clip_allowed_pct=self.config.max_ground_clip_pct,
            )
        else:
            ground_layer = fuse_ground_mertens(
                ground_aligned,
                mask_ground,
                metadata_list,
                max_clip_allowed_pct=self.config.max_ground_clip_pct,
            )

        # Step 5: Cropped celestial star alignment, stacking, and contrast enhancement
        print("[5/6] Extracting sky crops, tracking celestial drift across all exposures & enhancing sky...")
        exp_times = [m.exposure_time for m in metadata_list]
        iso_vals = [m.iso for m in metadata_list]

        # Extract cropped sky regions across all shutter speeds and sub-pixel register stars
        aligned_sky_crops, (ymin, ymax, xmin, xmax), star_shifts = extract_and_align_sky_crops(
            ground_aligned, mask_sky, ref_idx=ref_idx
        )

        # Stack aligned sky crops (weighted by exposure & ISO)
        stacked_sky_crop = stack_sky_exposures(
            aligned_sky_crops,
            exp_times,
            method=self.config.sky_stack_method,
            iso_values=iso_vals,
        )

        # Balance blacks & whites, boost contrast, and denoise the cropped sky
        crop_mask_sky = mask_sky[ymin:ymax, xmin:xmax]
        enhanced_sky_crop = enhance_sky_crop(
            stacked_sky_crop,
            crop_mask_sky,
            beta=self.config.sky_stretch_beta,
            target_black=self.config.target_black_floor,
            contrast_boost=self.config.contrast_boost,
            denoise=self.config.denoise_sky,
        )

        # Recompose enhanced sky crop onto the full canvas with non-overexposed ground baseline
        full_sky = ground_layer.copy()
        full_sky[ymin:ymax, xmin:xmax] = enhanced_sky_crop
        sky_final = full_sky

        # Step 6: Composite & exports
        print("[6/6] Blending composite and exporting master outputs...")
        composite = blend_composite(ground_layer, sky_final, mask_ground, mask_sky)

        # Output filepaths
        out_tiff = self.config.output_dir / "astro_composite.tiff"
        out_preview = self.config.output_dir / "astro_composite.jpg"
        save_master_tiff(composite, out_tiff)
        save_preview_jpeg(composite, out_preview, quality=96)

        if self.config.save_intermediate_layers:
            save_preview_jpeg(ground_layer, self.config.output_dir / "ground_layer.jpg")
            save_preview_jpeg(sky_final, self.config.output_dir / "sky_layer.jpg")
            save_preview_jpeg(enhanced_sky_crop, self.config.output_dir / "sky_crop_enhanced.jpg")
            save_preview_jpeg(mask_ground, self.config.output_dir / "horizon_mask.jpg")

        # Generate multi-row comparison card (all inputs + deliverables)
        input_labels = [
            f"Input #{i+1}: {m.shutter_str}" + (f" (ISO {m.iso})" if m.iso else "")
            for i, m in enumerate(metadata_list)
        ]
        card = create_comparison_card(
            input_images=images_bgr,
            input_labels=input_labels,
            landscape_layer=ground_layer,
            sky_layer=sky_final,
            final_composite=composite,
            target_width=1800,
        )
        card_path = self.config.output_dir / "comparison_summary.jpg"
        cv2.imwrite(str(card_path), card, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Plate solving & constellation annotation
        annotated_path = None
        if self.config.annotate:
            print("[Auto-Identify] Plate-solving celestial objects and drawing constellation overlay...")
            annotated_out = self.config.output_dir / "annotated_sky.jpg"
            _, ok = annotate_constellations(
                (composite * 255).astype(np.uint8),
                mask_sky=mask_sky,
                output_path=annotated_out,
            )
            if ok:
                annotated_path = annotated_out

        elapsed = time.time() - start_time
        print(f"\n=======================================================")
        print(f"Pipeline finished successfully in {elapsed:.1f}s!")
        print(f"  -> 16-bit TIFF Master: {out_tiff.resolve()}")
        print(f"  -> Display Preview:    {out_preview.resolve()}")
        print(f"  -> Comparison Card:    {card_path.resolve()}")
        if annotated_path:
            print(f"  -> Annotated Overlay:  {annotated_path.resolve()}")
        print(f"=======================================================\n")

        return out_preview

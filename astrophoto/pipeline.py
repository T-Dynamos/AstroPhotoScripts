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
    horizon_radius: int = 31
    max_ground_clip_pct: float = 0.50
    sky_stack_method: str = "weighted"  # 'weighted' or 'median'
    sky_stretch_beta: float = 28.0
    denoise_sky: bool = True
    save_intermediate_layers: bool = True
    annotate: bool = True
    ref_idx: int = 0


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
        ref_idx = max(0, min(self.config.ref_idx, len(images_bgr) - 1))
        print(f"[2/6] Performing sub-pixel ground ECC registration (Ref: #{ref_idx+1} {metadata_list[ref_idx].filename})...")
        ground_aligned, _ = align_ground_dataset(images_bgr, ref_idx=ref_idx)

        # Step 3: Horizon segmentation (general: mountains, trees, buildings, structures, plains)
        print("[3/6] Segmenting terrestrial landscape (trees, buildings, terrain) and starry sky along the horizon...")
        mask_ground, mask_sky = segment_horizon_mask(
            ground_aligned[-1],
            foreground_type=self.config.foreground_type,
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

        # Step 5: Celestial star alignment & sky processing
        print("[5/6] Tracking celestial star drift & stacking starry sky (with ground isolation)...")
        exp_times = [m.exposure_time for m in metadata_list]
        star_shifts = compute_celestial_star_drift(images_bgr, ref_idx=ref_idx)
        # Pass mask_sky to isolate sky and prevent shifted ground silhouettes from polluting sky near horizon
        star_aligned_skies = warp_sky_to_stars(images_bgr, star_shifts, mask_sky=mask_sky)

        sky_stacked = stack_sky_exposures(star_aligned_skies, exp_times, method=self.config.sky_stack_method)
        sky_leveled = flatten_sky_background(sky_stacked, mask_sky)
        sky_stretched = stretch_sky_asinh(sky_leveled, mask_sky, beta=self.config.sky_stretch_beta)

        if self.config.denoise_sky:
            print("  Applying noise suppression to sky background...")
            sky_final = denoise_sky_chroma(sky_stretched)
        else:
            sky_final = sky_stretched

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

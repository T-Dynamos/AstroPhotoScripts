"""
cli.py - Command-line interface for AstroPhotoScripts.
"""

import argparse
from pathlib import Path
import sys

from . import __version__
from .identify import annotate_constellations
from .io import discover_images, load_image
from .pipeline import AstroPipeline, PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astrophoto",
        description="AstroPhotoScripts: CLI pipeline for multi-exposure nightscapes, RAW/DNG, and star stacking.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Execute complete dual-stack pipeline on a sequence of exposures.")
    run_parser.add_argument(
        "input", type=str, nargs="?", default=".",
        help="Input directory containing exposures (JPEG, PNG, TIFF, or DNG/RAW). Default: current directory."
    )
    run_parser.add_argument(
        "--output", "-o", type=str, default="output",
        help="Directory to save final master TIFF, previews, and layers. Default: 'output'."
    )
    run_parser.add_argument(
        "--ground-mode", choices=["fuse", "select"], default="fuse",
        help="'fuse': blends non-overexposed exposures via Mertens fusion (default). 'select': picks the single best non-overexposed shot."
    )
    run_parser.add_argument(
        "--foreground-type", choices=["auto", "illuminated", "silhouette"], default="auto",
        help="Foreground horizon type: 'auto' (detects whether landscape is illuminated vs dark silhouette), 'illuminated', or 'silhouette'."
    )
    run_parser.add_argument(
        "--horizon-radius", type=int, default=31,
        help="Guided filter edge refinement radius for tree branches, rooflines, and landscape boundaries (default: 31)."
    )
    run_parser.add_argument(
        "--max-clip", type=float, default=0.50,
        help="Maximum allowable highlight clipping percentage for the landscape (default: 0.50%%)."
    )
    run_parser.add_argument(
        "--sky-method", choices=["weighted", "median"], default="weighted",
        help="Sky stacking method: 'weighted' (optimal SNR, default) or 'median' (satellite rejection)."
    )
    run_parser.add_argument(
        "--stretch", type=float, default=28.0,
        help="Astrophotography Asinh contrast stretch factor for faint stars and nebulae (default: 28.0)."
    )
    run_parser.add_argument(
        "--no-denoise", action="store_true",
        help="Disable chroma noise suppression on sky background."
    )
    run_parser.add_argument(
        "--no-layers", action="store_true",
        help="Skip saving individual intermediate layers (ground_layer, sky_layer, horizon_mask)."
    )
    run_parser.add_argument(
        "--annotate", action="store_true", default=True,
        help="Automatically plate-solve, identify constellations and deep-sky objects, and save annotated overlay."
    )

    # Command: identify
    identify_parser = subparsers.add_parser(
        "identify", help="Locally auto-identify constellations and deep-sky objects on an image and draw annotations."
    )
    identify_parser.add_argument(
        "image", type=str,
        help="Path to night sky image to solve and annotate."
    )
    identify_parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output path for annotated image (default: <image_stem>_annotated.jpg)."
    )

    # Command: inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect image metadata, dynamic range, and highlight clipping.")
    inspect_parser.add_argument(
        "input", type=str, nargs="?", default=".",
        help="Input directory to inspect. Default: current directory."
    )

    return parser


def handle_inspect(input_dir: Path) -> None:
    print(f"\n[Inspect] Scanning directory: {input_dir.resolve()}")
    paths = discover_images(input_dir)
    if not paths:
        print(f"No supported images found in: {input_dir.resolve()}")
        sys.exit(1)

    print(f"Found {len(paths)} images. Analyzing metadata and histogram clipping...\n")
    print("=" * 90)
    print(f"{'Idx':<4} {'Filename':<30} {'Shutter':<16} {'ISO':<8} {'Format':<10} {'Shadow%':<10} {'High%':<10}")
    print("-" * 90)

    for i, p in enumerate(paths):
        _, m = load_image(p)
        fmt = "RAW/DNG" if m.is_raw else "Standard"
        iso_str = str(m.iso) if m.iso is not None else "N/A"
        print(
            f"{i+1:<4} {m.filename:<30} {m.shutter_str:<16} {iso_str:<8} {fmt:<10} "
            f"{m.shadow_clip_pct:<10.2f} {m.highlight_clip_pct:<10.2f}"
        )
    print("=" * 90 + "\n")


def handle_identify(image_path: Path, output_path: Optional[Path] = None) -> None:
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path.resolve()}")
        sys.exit(1)

    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image: {image_path.resolve()}")
        sys.exit(1)

    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_annotated.jpg"

    print(f"\n[Identify] Plate-solving celestial objects in: {image_path.name}...")
    annotated, success = annotate_constellations(img, output_path=output_path)
    if success:
        print(f"  -> Successfully generated: {output_path.resolve()}\n")
    else:
        print("  -> Constellation matching did not meet confidence threshold.\n")


def main() -> None:
    parser = build_parser()
    # Default to 'run' if called without a subcommand but with arguments or defaults
    args = parser.parse_args()

    if args.command is None:
        # Default behavior: run on current directory or show help
        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)
        else:
            args = parser.parse_args(["run"] + sys.argv[1:])

    if args.command == "inspect":
        handle_inspect(Path(args.input))
    elif args.command == "identify":
        out = Path(args.output) if args.output else None
        handle_identify(Path(args.image), output_path=out)
    elif args.command == "run":
        config = PipelineConfig(
            input_dir=Path(args.input),
            output_dir=Path(args.output),
            ground_mode=args.ground_mode,
            foreground_type=args.foreground_type,
            horizon_radius=args.horizon_radius,
            max_ground_clip_pct=args.max_clip,
            sky_stack_method=args.sky_method,
            sky_stretch_beta=args.stretch,
            denoise_sky=not args.no_denoise,
            save_intermediate_layers=not args.no_layers,
            annotate=args.annotate,
        )
        pipeline = AstroPipeline(config)
        pipeline.run()


if __name__ == "__main__":
    main()

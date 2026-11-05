"""
AstroPhotoScripts - General CLI-based Astrophotography & Landscape Stacking Pipeline.

Handles bracketed nightscapes, RAW/DNG images, dual-alignment (stars + terrain),
sky noise suppression, background leveling, Asinh contrast stretching, and
highlight-protected landscape selection/fusion.
"""

import os
import cv2

__version__ = "1.0.0"


def enable_opencv_multithreading(verbose: bool = True) -> int:
    """Configure OpenCV to use all available CPU cores and SIMD hardware acceleration.

    Returns:
        int: Number of threads configured.
    """
    cv2.setUseOptimized(True)
    cpu_cores = os.cpu_count() or cv2.getNumberOfCPUs()
    cv2.setNumThreads(cpu_cores)
    if verbose:
        print(f"[OpenCV] Enabled multi-threading with {cv2.getNumThreads()} threads (SIMD optimized: {cv2.useOptimized()})")
    return cv2.getNumThreads()

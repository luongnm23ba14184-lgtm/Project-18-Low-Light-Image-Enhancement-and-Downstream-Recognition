"""
Classic Digital Image Processing (DIP) Baseline Module.
Implements CIE LAB + CLAHE (Contrast Limited Adaptive Histogram Equalization)
coupled with Bilateral Filtering for edge-preserving denoising.
"""

import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def enhance_clahe_bilateral(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    bilateral_d: int = 7,
    bilateral_sigma_color: float = 50.0,
    bilateral_sigma_space: float = 50.0,
) -> np.ndarray:
    """
    Enhances a low-light image using traditional DIP:
    1. RGB/BGR to CIE LAB color space.
    2. CLAHE applied strictly to the L (Luminance) channel to avoid color distortion.
    3. Convert back to RGB/BGR.
    4. Bilateral filter to smooth sensor noise while preserving object boundaries.

    Args:
        image_bgr: Input BGR image (uint8, [0, 255]).
        clip_limit: Threshold for contrast limiting in CLAHE.
        tile_grid_size: Size of grid for histogram equalization (rows, cols).
        bilateral_d: Diameter of each pixel neighborhood in bilateral filter.
        bilateral_sigma_color: Filter sigma in the color space.
        bilateral_sigma_space: Filter sigma in the coordinate space.

    Returns:
        Enhanced BGR image (uint8, [0, 255]).
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is None or empty.")

    # 1. Convert BGR to CIE LAB color space
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 2. Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)

    # 3. Merge back and convert to BGR
    lab_enhanced = cv2.merge((l_clahe, a_channel, b_channel))
    bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # 4. Apply Bilateral Filtering for edge-preserving denoising
    if bilateral_d > 0:
        bgr_enhanced = cv2.bilateralFilter(
            bgr_enhanced,
            d=bilateral_d,
            sigmaColor=bilateral_sigma_color,
            sigmaSpace=bilateral_sigma_space,
        )

    return bgr_enhanced


def batch_enhance_clahe(
    src_dir: str | Path,
    dst_dir: str | Path,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    **kwargs
) -> int:
    """
    Batch-enhances all images in a source directory and saves them to a destination directory.
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in src_path.iterdir() if f.suffix.lower() in extensions]
    count = 0

    for img_file in tqdm(image_files, desc=f"CLAHE enhancing {src_path.name}"):
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        enhanced = enhance_clahe_bilateral(img, **kwargs)
        cv2.imwrite(str(dst_path / img_file.name), enhanced)
        count += 1

    return count

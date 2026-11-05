"""Fast dominant-color detection on camera frames (pure numpy, no model).

Runs in a few milliseconds on a Raspberry Pi, so games that only need to know
"what color is the object being shown" don't have to send images to a vision model.
"""

from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray


# Hue ranges in degrees [start, end). Red wraps around 0.
HUE_RANGES: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "rojo": ((0.0, 15.0), (340.0, 360.0)),
    "naranja": ((15.0, 40.0),),
    "amarillo": ((40.0, 70.0),),
    "verde": ((70.0, 170.0),),
    "azul": ((170.0, 260.0),),
    "morado": ((260.0, 340.0),),
}

MIN_SATURATION = 0.35
MIN_VALUE = 0.25
# Below this share of colorful pixels in the region, we report no clear color.
MIN_COLORFUL_FRACTION = 0.08
ANALYSIS_MAX_SIDE = 160


def _center_crop(frame: NDArray[np.uint8], fraction: float) -> NDArray[np.uint8]:
    h, w = frame.shape[:2]
    ch, cw = max(1, int(h * fraction)), max(1, int(w * fraction))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return frame[y0 : y0 + ch, x0 : x0 + cw]


def _subsample(frame: NDArray[np.uint8], max_side: int) -> NDArray[np.uint8]:
    step = max(1, max(frame.shape[:2]) // max_side)
    return frame[::step, ::step]


def _bgr_to_hsv(frame: NDArray[np.uint8]) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Return hue in degrees [0, 360), saturation and value in [0, 1]."""
    bgr = frame.astype(np.float32) / 255.0
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    maxc = np.max(bgr, axis=-1)
    minc = np.min(bgr, axis=-1)
    delta = maxc - minc
    safe_delta = np.where(delta == 0, 1.0, delta)

    hue = np.zeros_like(maxc)
    hue = np.where(maxc == r, ((g - b) / safe_delta) % 6.0, hue)
    hue = np.where(maxc == g, (b - r) / safe_delta + 2.0, hue)
    hue = np.where(maxc == b, (r - g) / safe_delta + 4.0, hue)
    hue = np.where(delta == 0, 0.0, hue) * 60.0

    saturation = np.where(maxc == 0, 0.0, delta / np.where(maxc == 0, 1.0, maxc))
    return hue.astype(np.float32), saturation.astype(np.float32), maxc.astype(np.float32)


def detect_dominant_color(frame: NDArray[np.uint8], center_fraction: float = 0.6) -> Dict[str, object]:
    """Return the dominant color name in the center of a BGR frame and the share of each color.

    The result has keys: ``color`` (a name from HUE_RANGES or ``"ninguno"``), ``confidence``
    (share of colorful pixels that belong to that color) and ``shares`` (per-color share).
    """
    region = _subsample(_center_crop(frame, center_fraction), ANALYSIS_MAX_SIDE)
    hue, sat, val = _bgr_to_hsv(region)
    colorful = (sat >= MIN_SATURATION) & (val >= MIN_VALUE)
    total = int(region.shape[0] * region.shape[1])
    n_colorful = int(colorful.sum())

    counts: Dict[str, int] = {}
    for name, ranges in HUE_RANGES.items():
        in_range = np.zeros_like(colorful)
        for start, end in ranges:
            in_range |= (hue >= start) & (hue < end)
        counts[name] = int((in_range & colorful).sum())

    shares = {name: round(c / n_colorful, 2) if n_colorful else 0.0 for name, c in counts.items()}
    if total == 0 or n_colorful / total < MIN_COLORFUL_FRACTION:
        return {"color": "ninguno", "confidence": 0.0, "shares": shares}

    best = max(counts, key=lambda k: counts[k])
    return {"color": best, "confidence": shares[best], "shares": shares}

"""Helpers for encoding camera frames."""

import os
from fractions import Fraction

import av
import numpy as np
from numpy.typing import NDArray


# Frames sent to the model are downscaled: vision models resize large images anyway,
# and full-resolution JPEGs are slow to encode on the Raspberry Pi and to upload over WiFi.
DEFAULT_MAX_IMAGE_SIDE = int(os.getenv("CAMERA_MAX_IMAGE_SIDE", "640"))
# MJPEG qscale: 2 (best) .. 31 (worst). 5 keeps colors faithful at a fraction of the size of 3.
DEFAULT_JPEG_QSCALE = int(os.getenv("CAMERA_JPEG_QSCALE", "5"))


def _target_size(width: int, height: int, max_side: int | None) -> tuple[int, int]:
    """Return an even-sized (width, height) that fits within max_side."""
    if not max_side or max(width, height) <= max_side:
        return width, height
    scale = max_side / max(width, height)
    new_w = max(2, int(round(width * scale)) // 2 * 2)
    new_h = max(2, int(round(height * scale)) // 2 * 2)
    return new_w, new_h


def encode_bgr_frame_as_jpeg(
    frame: NDArray[np.uint8],
    max_side: int | None = DEFAULT_MAX_IMAGE_SIDE,
    qscale: int = DEFAULT_JPEG_QSCALE,
) -> bytes:
    """Encode a BGR camera frame as JPEG bytes, downscaled so its longest side is at most max_side."""
    rgb_frame = np.ascontiguousarray(frame[..., ::-1])
    video_frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")

    width, height = _target_size(rgb_frame.shape[1], rgb_frame.shape[0], max_side)
    if (width, height) != (rgb_frame.shape[1], rgb_frame.shape[0]):
        video_frame = video_frame.reformat(width=width, height=height)

    codec = av.CodecContext.create("mjpeg", "w")
    codec.width = width  # type: ignore[attr-defined]
    codec.height = height  # type: ignore[attr-defined]
    codec.pix_fmt = "yuvj444p"  # type: ignore[attr-defined]
    codec.time_base = Fraction(1, 1)
    codec.options = {"qscale": str(qscale)}

    packets = codec.encode(video_frame)  # type: ignore[attr-defined]
    packets += codec.encode(None)  # type: ignore[attr-defined]
    if not packets:
        raise RuntimeError("Failed to encode frame as JPEG")

    return b"".join(bytes(packet) for packet in packets)

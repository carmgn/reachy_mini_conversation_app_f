"""Tests for local dominant-color detection."""

import numpy as np
import pytest

from reachy_mini_conversation_app.vision.color_detection import detect_dominant_color


def _bgr(r: int, g: int, b: int) -> np.ndarray:
    return np.full((480, 640, 3), [b, g, r], dtype=np.uint8)


@pytest.mark.parametrize(
    ("rgb", "expected"),
    [
        ((220, 20, 30), "rojo"),
        ((30, 60, 220), "azul"),
        ((240, 220, 20), "amarillo"),
        ((30, 180, 40), "verde"),
        ((250, 130, 0), "naranja"),
        ((140, 40, 200), "morado"),
    ],
)
def test_detects_solid_colors(rgb: tuple[int, int, int], expected: str) -> None:
    """Solid color frames are classified correctly."""
    result = detect_dominant_color(_bgr(*rgb))
    assert result["color"] == expected
    assert result["confidence"] == 1.0


def test_gray_frame_has_no_clear_color() -> None:
    """Gray frames report no clear color."""
    assert detect_dominant_color(_bgr(128, 128, 128))["color"] == "ninguno"


def test_uses_center_object_over_background() -> None:
    """The object in the center wins over a neutral background."""
    frame = _bgr(240, 240, 240)
    frame[180:300, 260:380] = [20, 160, 30]  # green object in the center (BGR)
    assert detect_dominant_color(frame)["color"] == "verde"

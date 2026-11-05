import asyncio
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.vision.color_detection import detect_dominant_color


logger = logging.getLogger(__name__)


class DetectColor(Tool):
    """Detect the dominant color of the object in front of the camera, locally and instantly."""

    name = "detect_color"
    description = (
        "Instantly tell the dominant color of the object shown in the center of the camera "
        "(rojo, naranja, amarillo, verde, azul, morado, or 'ninguno' if no clear color). "
        "Much faster than the camera tool: use it whenever you only need to know a color."
    )
    parameters_schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Return the dominant color of the latest camera frame."""
        if deps.camera_worker is None:
            return {"error": "Camera worker not available"}
        frame = deps.camera_worker.get_latest_frame()
        if frame is None:
            return {"error": "No frame available"}

        result = await asyncio.to_thread(detect_dominant_color, frame)
        logger.info("Tool call: detect_color -> %s (%.2f)", result["color"], result["confidence"])
        return result

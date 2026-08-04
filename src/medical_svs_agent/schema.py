"""Canonical schema for the project's only model-visible tool."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


TOOL_NAME = "openslide_crop"
TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Inspect a selected region of the original whole-slide image at higher "
            "resolution. Coordinates refer to the low-resolution overview and range "
            "from 0 to 1000. OpenSlide level 0 is the highest resolution."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0, "maximum": 1000},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "Relative [x1, y1, x2, y2] region on the slide overview.",
                },
                "level": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Pyramid level to read. Omit for level 0.",
                },
            },
            "required": ["bbox_2d"],
        },
    },
}


def tool_schema() -> dict[str, Any]:
    return deepcopy(TOOL_SCHEMA)


"""Canonical schemas for the model-visible whole-slide tools."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


TOOL_NAME = "openslide_crop"
RETURN_TOOL_NAME = "return_level"
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

RETURN_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": RETURN_TOOL_NAME,
        "description": (
            "Return to the most recently observed view at a previous, coarser "
            "pyramid level. Use this after a zoom branch is uninformative and a "
            "different region should be selected from an earlier view."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "level": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Previously observed pyramid level to return to.",
                }
            },
            "required": ["level"],
        },
    },
}


def tool_schema() -> dict[str, Any]:
    return deepcopy(TOOL_SCHEMA)


def return_tool_schema() -> dict[str, Any]:
    return deepcopy(RETURN_TOOL_SCHEMA)


def tool_schemas() -> list[dict[str, Any]]:
    return [tool_schema(), return_tool_schema()]


def function_tool_schemas() -> list[dict[str, Any]]:
    return [schema["function"] for schema in tool_schemas()]

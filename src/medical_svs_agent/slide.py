"""Secure slide registry and high-resolution crop implementation."""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
from typing import Any, Callable

from PIL import Image


class SlideError(ValueError):
    pass


def encode_image(image: Image.Image, quality: int = 92) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def load_manifest(path: str | Path) -> dict[str, Path]:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("slides", payload) if isinstance(payload, dict) else payload
    result: dict[str, Path] = {}
    if isinstance(entries, dict):
        iterator = entries.items()
    elif isinstance(entries, list):
        iterator = ((row["slide_id"], row["slide_path"]) for row in entries)
    else:
        raise SlideError("manifest must be a mapping or a list under 'slides'")
    for slide_id, raw_path in iterator:
        slide_path = Path(str(raw_path)).expanduser()
        if not slide_path.is_absolute():
            slide_path = manifest_path.parent / slide_path
        result[str(slide_id)] = slide_path.resolve()
    if not result:
        raise SlideError("slide manifest is empty")
    return result


def read_slide_overview(
    slide_path: str | Path,
    *,
    level: int = 4,
    opener: Callable[[str], Any] | None = None,
) -> tuple[dict[str, Any], Image.Image]:
    """Read the entire slide at a fixed pyramid level."""
    path = Path(slide_path).expanduser().resolve()
    if not path.is_file():
        raise SlideError(f"slide does not exist: {path}")
    if opener is None:
        import openslide

        opener = openslide.OpenSlide
    slide = opener(str(path))
    try:
        level_count = int(slide.level_count)
        if level < 0 or level >= level_count:
            raise SlideError(f"overview level must be between 0 and {level_count - 1}")
        slide_dimensions = list(map(int, slide.dimensions))
        level_dimensions = list(map(int, slide.level_dimensions[level]))
        downsample = float(slide.level_downsamples[level])
        image = slide.read_region((0, 0), level, tuple(level_dimensions)).convert("RGB")
        return {
            "level": level,
            "level_downsample": downsample,
            "level_dimensions": level_dimensions,
            "slide_dimensions": slide_dimensions,
        }, image
    finally:
        close = getattr(slide, "close", None)
        if close is not None:
            close()


class OpenSlideCropService:
    def __init__(
        self,
        manifest: dict[str, Path],
        *,
        opener: Callable[[str], Any] | None = None,
        max_crop_side: int = 2048,
    ):
        self.manifest = {str(key): Path(value).resolve() for key, value in manifest.items()}
        self.max_crop_side = int(max_crop_side)
        if self.max_crop_side < 64:
            raise SlideError("max_crop_side must be at least 64")
        if opener is None:
            import openslide

            opener = openslide.OpenSlide
        self.opener = opener

    @staticmethod
    def _bbox(value: Any) -> list[float]:
        if not isinstance(value, list) or len(value) != 4:
            raise SlideError("bbox_2d must contain four coordinates")
        try:
            box = [min(1000.0, max(0.0, float(item))) for item in value]
        except (TypeError, ValueError) as exc:
            raise SlideError("bbox_2d coordinates must be numeric") from exc
        if not all(math.isfinite(item) for item in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise SlideError("bbox_2d must define a non-empty region")
        return box

    def crop(self, slide_id: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], Image.Image]:
        slide_path = self.manifest.get(str(slide_id))
        if slide_path is None:
            raise SlideError(f"unknown slide_id: {slide_id}")
        if not slide_path.is_file():
            raise SlideError(f"registered slide does not exist: {slide_path}")
        box = self._bbox(arguments.get("bbox_2d"))
        slide = self.opener(str(slide_path))
        try:
            width, height = map(int, slide.dimensions)
            level_count = int(slide.level_count)
            level = int(arguments.get("level", 0))
            if level < 0 or level >= level_count:
                raise SlideError(f"level must be between 0 and {level_count - 1}")
            x1 = int(math.floor(box[0] * width / 1000.0))
            y1 = int(math.floor(box[1] * height / 1000.0))
            x2 = int(math.ceil(box[2] * width / 1000.0))
            y2 = int(math.ceil(box[3] * height / 1000.0))
            downsample = float(slide.level_downsamples[level])
            read_width = max(1, int(math.ceil((x2 - x1) / downsample)))
            read_height = max(1, int(math.ceil((y2 - y1) / downsample)))
            if max(read_width, read_height) > self.max_crop_side:
                raise SlideError(
                    f"requested crop is {read_width}x{read_height} at level {level}; "
                    f"select a smaller bbox or a coarser level (max side {self.max_crop_side})"
                )
            image = slide.read_region((x1, y1), level, (read_width, read_height)).convert("RGB")
            properties = getattr(slide, "properties", {})
            result = {
                "slide_id": str(slide_id),
                "bbox_2d": [round(item, 4) for item in box],
                "level": level,
                "level_downsample": downsample,
                "level0_bbox": [x1, y1, x2, y2],
                "crop_size": [read_width, read_height],
                "slide_dimensions": [width, height],
                "mpp_x": properties.get("openslide.mpp-x"),
                "mpp_y": properties.get("openslide.mpp-y"),
                "objective_power": properties.get("openslide.objective-power"),
                "image_outputs": [{"target_image": 1, "type": "image"}],
                "source": "openslide",
            }
            return result, image
        finally:
            close = getattr(slide, "close", None)
            if close is not None:
                close()

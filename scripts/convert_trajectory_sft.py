#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from medical_svs_agent.data import SYSTEM_PROMPT
from medical_svs_agent.schema import TOOL_NAME, function_tool_schemas


USER_PROMPT = "<image>\n请观察该全切片病理图像，必要时放大检查，并判断是否存在恶性细胞。"


def _thinking(text: Any, *, context: str) -> str:
    reasoning = str(text or "").strip()
    if not reasoning:
        raise ValueError(f"missing reasoning: {context}")
    return f"<think>\n{reasoning}\n</think>"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_bbox(bbox: list[int], width: int, height: int) -> list[float]:
    x, y, box_width, box_height = bbox
    return [
        round(x / width * 1000, 4),
        round(y / height * 1000, 4),
        round((x + box_width) / width * 1000, 4),
        round((y + box_height) / height * 1000, 4),
    ]


def _same_patch(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in ("patch_id", "file", "level0_bbox"))


def _image_path(case_dir: Path, patch: dict[str, Any]) -> Path:
    path = (case_dir / str(patch["file"])).resolve()
    if not path.is_file():
        raise ValueError(f"image does not exist: {path}")
    with Image.open(path) as image:
        image.verify()
    return path


def _convert_case(path: Path) -> dict[str, Any]:
    case_dir = path.parent
    trajectory = _read_json(path)
    metadata = _read_json(case_dir / "slide_metadata.json")
    slide_id = str(trajectory["slide_id"])
    if slide_id != str(metadata["id"]):
        raise ValueError("slide_id does not match slide_metadata.json")
    width, height = int(metadata["width"]), int(metadata["height"])
    steps = list(trajectory.get("trajectory") or [])
    if not steps:
        raise ValueError("trajectory has no steps")
    diagnosis = str(trajectory["final_answer"]["diagnosis"]).strip()
    if not diagnosis:
        raise ValueError("final diagnosis is empty")

    messages = [{"role": "user", "content": USER_PROMPT}]
    images = [str(_image_path(case_dir, steps[0]["input_patch"]))]
    previous_output = None
    for index, step in enumerate(steps, 1):
        if previous_output is not None and not _same_patch(previous_output, step["input_patch"]):
            raise ValueError(f"trajectory is not continuous at step {index}")
        action = step["action"]
        if action.get("type") != "zoom":
            raise ValueError(f"unsupported action at step {index}: {action.get('type')}")
        bbox = list(action["level0_bbox"])
        if len(bbox) != 4:
            raise ValueError(f"invalid bbox at step {index}")
        x, y, box_width, box_height = map(int, bbox)
        if (
            box_width <= 0
            or box_height <= 0
            or x < 0
            or y < 0
            or x + box_width > width
            or y + box_height > height
        ):
            raise ValueError(f"bbox is outside the slide at step {index}")
        if bbox != list(step["output_patch"]["level0_bbox"]):
            raise ValueError(f"action and output bbox differ at step {index}")
        level = int(action["to_level"])
        relative_bbox = _relative_bbox(bbox, width, height)
        call = {
            "name": TOOL_NAME,
            "arguments": {"bbox_2d": relative_bbox, "level": level},
        }
        result = {
            "source": "openslide",
            "level": level,
            "level0_bbox": [x, y, x + box_width, y + box_height],
        }
        messages.append(
            {
                "role": "function_call",
                "content": (
                    _thinking(step.get("reasoning"), context=f"{slide_id} step {index}")
                    + "\n\n"
                    + json.dumps(call, ensure_ascii=False)
                ),
            }
        )
        messages.append(
            {
                "role": "observation",
                "content": json.dumps(result, ensure_ascii=False) + "\n<image>",
            }
        )
        images.append(str(_image_path(case_dir, step["output_patch"])))
        previous_output = step["output_patch"]

    final_reasoning = trajectory.get("final_answer", {}).get("summary_reasoning") or diagnosis
    messages.append(
        {
            "role": "assistant",
            "content": (
                _thinking(final_reasoning, context=f"{slide_id} final answer")
                + f"\n\n<answer>{diagnosis}</answer>"
            ),
        }
    )
    image_tokens = sum(message["content"].count("<image>") for message in messages)
    if image_tokens != len(images):
        raise ValueError("image token count does not match image paths")
    return {
        "uid": slide_id,
        "slide_id": slide_id,
        "system": SYSTEM_PROMPT,
        "messages": messages,
        "images": images,
        "tools": json.dumps(function_tool_schemas(), ensure_ascii=False),
        "diagnosis": diagnosis,
        "display_name": str(metadata["display_name"]),
    }


def _slide_paths(input_dirs: list[Path]) -> dict[str, str]:
    slides = {}
    for input_dir in input_dirs:
        manifest = input_dir / "manifest.jsonl"
        if not manifest.is_file():
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                slides[str(row["slide_id"])] = str(row["file_path"])
    return slides


def convert_trajectory_exports(
    input_dirs: list[Path],
    output_dir: Path,
    *,
    excluded_slide_ids: set[str] | None = None,
) -> dict[str, Any]:
    excluded_slide_ids = excluded_slide_ids or set()
    paths = []
    for input_dir in input_dirs:
        trajectories_dir = input_dir / "trajectories"
        paths.extend(sorted(trajectories_dir.glob("*/trajectory.json")))

    rows = []
    excluded = []
    seen_slide_ids = set()
    for path in paths:
        try:
            slide_id = str(_read_json(path)["slide_id"])
            if slide_id in excluded_slide_ids:
                raise ValueError("excluded by slide id")
            if slide_id in seen_slide_ids:
                raise ValueError("duplicate slide_id")
            row = _convert_case(path)
            seen_slide_ids.add(slide_id)
            rows.append(row)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            excluded.append({"trajectory": str(path.resolve()), "reason": str(exc)})

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sft.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    dataset_info = {
        "medical_svs_sft": {
            "file_name": output_path.name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "system": "system",
                "tools": "tools",
                "images": "images",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "observation_tag": "observation",
                "function_tag": "function_call",
                "system_tag": "system",
            },
        }
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    source_slides = _slide_paths(input_dirs)
    slides = {
        row["slide_id"]: source_slides[row["slide_id"]]
        for row in rows
        if row["slide_id"] in source_slides
    }
    (output_dir / "slides.json").write_text(
        json.dumps({"slides": slides}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "found": len(paths),
        "converted": len(rows),
        "skipped": len(excluded),
        "step_counts": dict(sorted(Counter((len(row["images"]) - 1 for row in rows)).items())),
        "diagnosis_counts": dict(Counter(row["diagnosis"] for row in rows).most_common()),
        "excluded": excluded,
        "output": str(output_path.resolve()),
    }
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert exported WSI trajectories to LLaMA-Factory SFT JSONL"
    )
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-slide-id", action="append", default=[])
    args = parser.parse_args()
    report = convert_trajectory_exports(
        args.input_dir,
        args.output_dir,
        excluded_slide_ids=set(args.exclude_slide_id),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

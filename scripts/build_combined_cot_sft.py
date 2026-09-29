#!/usr/bin/env python3
"""Merge an existing CoT SFT corpus with packaged trajectory JSON and patch images."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from medical_svs_agent.data import SYSTEM_PROMPT
from medical_svs_agent.schema import (
    RETURN_TOOL_NAME,
    TOOL_NAME,
    function_tool_schemas,
)


USER_PROMPT = "<image>\n请观察该全切片病理图像，必要时放大检查，并判断是否存在恶性细胞。"
BENIGN_DIAGNOSIS = "未见明显恶性细胞"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _thinking(text: Any, *, context: str) -> str:
    reasoning = str(text or "").strip()
    if not reasoning:
        raise ValueError(f"missing reasoning: {context}")
    return f"<think>\n{reasoning}\n</think>"


def _patch_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in ("patch_id", "file", "level0_bbox"))


def _bbox_equal(left: Any, right: Any) -> bool:
    return list(left or []) == list(right or [])


def _relative_bbox(bbox: list[int], width: int, height: int) -> list[float]:
    x, y, box_width, box_height = bbox
    return [
        round(x / width * 1000, 4),
        round(y / height * 1000, 4),
        round((x + box_width) / width * 1000, 4),
        round((y + box_height) / height * 1000, 4),
    ]


def _xyxy(bbox: list[int]) -> list[int]:
    x, y, width, height = map(int, bbox)
    return [x, y, x + width, y + height]


def _image_path(
    image_root: Path,
    patch: dict[str, Any],
    verified: set[Path],
) -> Path:
    root = image_root.resolve()
    path = (root / str(patch["file"])).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"image escapes image root: {path}") from exc
    if not path.is_file():
        raise ValueError(f"image does not exist: {path}")
    if path not in verified:
        with Image.open(path) as image:
            image.verify()
        verified.add(path)
    return path


def _history_target(
    history: list[tuple[int, dict[str, Any], Path]],
    level: int,
    bbox: list[int],
) -> tuple[int, dict[str, Any], Path] | None:
    return next(
        (
            entry
            for entry in reversed(history)
            if entry[0] == level and _bbox_equal(entry[1].get("level0_bbox"), bbox)
        ),
        None,
    )


def _next_reasoning(events: list[dict[str, Any]], start: int) -> dict[str, Any] | None:
    return next(
        (event for event in events[start:] if event.get("type") == "reasoning_step"),
        None,
    )


def _convert_additional(
    record: dict[str, Any],
    image_root: Path,
    verified: set[Path],
) -> tuple[dict[str, Any], dict[str, int]]:
    slide_id = str(record["slide_id"])
    cot = record["cot"]
    if str(cot["slide_id"]) != slide_id:
        raise ValueError("record and CoT slide_id differ")
    events = list(cot.get("trajectory") or [])
    reasoning_events = [event for event in events if event.get("type") == "reasoning_step"]
    if not reasoning_events:
        raise ValueError("trajectory has no reasoning steps")

    initial_patch = reasoning_events[0]["input_patch"]
    initial_bbox = list(map(int, initial_patch["level0_bbox"]))
    if initial_bbox[:2] != [0, 0] or initial_bbox[2] <= 0 or initial_bbox[3] <= 0:
        raise ValueError(f"invalid initial whole-slide bbox: {initial_bbox}")
    width, height = initial_bbox[2], initial_bbox[3]
    first_level = int(reasoning_events[0]["action"]["from_level"])
    initial_path = _image_path(image_root, initial_patch, verified)

    messages: list[dict[str, str]] = [{"role": "user", "content": USER_PROMPT}]
    images = [str(initial_path)]
    history: list[tuple[int, dict[str, Any], Path]] = [(first_level, initial_patch, initial_path)]
    current_patch = initial_patch
    current_level = first_level
    zoom_calls = 0
    return_calls = 0
    skipped_historical_returns = 0
    skipped_unreplayable_returns = 0

    index = 0
    while index < len(events):
        event = events[index]
        event_type = event.get("type")
        if event_type == "return_level":
            end = index
            while end < len(events) and events[end].get("type") == "return_level":
                end += 1
            group = events[index:end]
            next_step = _next_reasoning(events, end)

            if event.get("after_step") is None:
                skipped_historical_returns += len(group)
                index = end
                continue

            simulated_patch = current_patch
            simulated_level = current_level
            targets: list[tuple[int, dict[str, Any], Path]] = []
            replayable = True
            for returned in group:
                action = returned.get("action") or {}
                from_level = int(action.get("from_level", -1))
                to_level = int(action.get("to_level", -1))
                if from_level != simulated_level or not _bbox_equal(
                    simulated_patch.get("level0_bbox"), action.get("from_level0_bbox")
                ):
                    replayable = False
                    break
                target = _history_target(history, to_level, list(action.get("to_level0_bbox") or []))
                if target is None:
                    replayable = False
                    break
                targets.append(target)
                simulated_level, simulated_patch, _ = target

            if (
                next_step is None
                or simulated_level != int(next_step["action"]["from_level"])
                or not _bbox_equal(
                    simulated_patch.get("level0_bbox"),
                    next_step["input_patch"].get("level0_bbox"),
                )
            ):
                replayable = False

            if not replayable:
                if next_step is not None and _patch_equal(current_patch, next_step["input_patch"]):
                    skipped_unreplayable_returns += len(group)
                    index = end
                    continue
                raise ValueError("return_level group cannot be replayed continuously")

            for returned, target in zip(group, targets):
                target_level, target_patch, target_path = target
                reasoning = returned.get("reasoning") or (
                    f"当前放大分支信息不足，返回之前的 Level {target_level} 视野重新选择区域。"
                )
                call = {"name": RETURN_TOOL_NAME, "arguments": {"level": target_level}}
                messages.append(
                    {
                        "role": "function_call",
                        "content": _thinking(reasoning, context=f"{slide_id} return_level")
                        + "\n\n"
                        + json.dumps(call, ensure_ascii=False),
                    }
                )
                messages.append(
                    {
                        "role": "observation",
                        "content": json.dumps(
                            {
                                "source": "history",
                                "level": target_level,
                                "level0_bbox": _xyxy(list(target_patch["level0_bbox"])),
                            },
                            ensure_ascii=False,
                        )
                        + "\n<image>",
                    }
                )
                images.append(str(target_path))
                current_level, current_patch = target_level, target_patch
                return_calls += 1
            # The exporter may assign a fresh patch ID to a returned view. Its
            # level and level-0 bbox identify the same image seen by the model.
            current_level = int(next_step["action"]["from_level"])
            current_patch = next_step["input_patch"]
            index = end
            continue

        if event_type != "reasoning_step":
            raise ValueError(f"unsupported trajectory event: {event_type}")
        if not _patch_equal(current_patch, event["input_patch"]):
            raise ValueError(f"trajectory is not continuous at reasoning step {event.get('step')}")
        action = event.get("action") or {}
        if action.get("type") != "zoom":
            raise ValueError(f"unsupported action: {action.get('type')}")
        bbox = list(map(int, action["level0_bbox"]))
        x, y, box_width, box_height = bbox
        if (
            box_width <= 0
            or box_height <= 0
            or x < 0
            or y < 0
            or x + box_width > width
            or y + box_height > height
        ):
            raise ValueError(f"bbox outside slide at step {event.get('step')}: {bbox}")
        if not _bbox_equal(bbox, event["output_patch"].get("level0_bbox")):
            raise ValueError(f"action and output bbox differ at step {event.get('step')}")

        level = int(action["to_level"])
        call = {
            "name": TOOL_NAME,
            "arguments": {"bbox_2d": _relative_bbox(bbox, width, height), "level": level},
        }
        messages.append(
            {
                "role": "function_call",
                "content": _thinking(
                    event.get("reasoning"), context=f"{slide_id} step {event.get('step')}"
                )
                + "\n\n"
                + json.dumps(call, ensure_ascii=False),
            }
        )
        messages.append(
            {
                "role": "observation",
                "content": json.dumps(
                    {"source": "openslide", "level": level, "level0_bbox": _xyxy(bbox)},
                    ensure_ascii=False,
                )
                + "\n<image>",
            }
        )
        output_patch = event["output_patch"]
        output_path = _image_path(image_root, output_patch, verified)
        images.append(str(output_path))
        current_patch = output_patch
        current_level = level
        history.append((level, output_patch, output_path))
        zoom_calls += 1
        index += 1

    final_answer = cot.get("final_answer") or {}
    diagnosis = str(final_answer.get("diagnosis") or "").strip()
    if not diagnosis:
        raise ValueError("final diagnosis is empty")
    label = "benign" if diagnosis == BENIGN_DIAGNOSIS else "malignant"
    answer = BENIGN_DIAGNOSIS if label == "benign" else "考虑恶性"
    final_reasoning = final_answer.get("summary_reasoning") or diagnosis
    messages.append(
        {
            "role": "assistant",
            "content": _thinking(final_reasoning, context=f"{slide_id} final answer")
            + f"\n\n<answer>{answer}</answer>",
        }
    )
    if len(images) != sum(message["content"].count("<image>") for message in messages):
        raise ValueError("image token count does not match image paths")

    row = {
        "uid": slide_id,
        "slide_id": slide_id,
        "system": SYSTEM_PROMPT,
        "messages": messages,
        "images": images,
        "tools": json.dumps(function_tool_schemas(), ensure_ascii=False),
        "label": label,
        "source_diagnosis": diagnosis,
        "display_name": str(record["display_name"]),
    }
    stats = {
        "zoom_calls": zoom_calls,
        "return_calls": return_calls,
        "skipped_historical_returns": skipped_historical_returns,
        "skipped_unreplayable_returns": skipped_unreplayable_returns,
    }
    return row, stats


def _dataset_info() -> dict[str, Any]:
    return {
        "medical_svs_sft": {
            "file_name": "sft.jsonl",
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


def build_combined(
    base_sft: Path,
    base_slides: Path,
    additional_json: Path,
    image_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    base_rows = _read_jsonl(base_sft)
    schemas_json = json.dumps(function_tool_schemas(), ensure_ascii=False)
    for row in base_rows:
        row["system"] = SYSTEM_PROMPT
        row["tools"] = schemas_json
        if len(row.get("images") or []) != sum(
            str(message.get("content") or "").count("<image>")
            for message in row.get("messages") or []
        ):
            raise ValueError(f"base image count mismatch: {row.get('slide_id')}")
        missing = [path for path in row["images"] if not Path(path).is_file()]
        if missing:
            raise ValueError(f"base images missing for {row.get('slide_id')}: {missing[:3]}")

    payload = json.loads(additional_json.read_text(encoding="utf-8"))
    records = list(payload.get("trajectories") or [])
    verified: set[Path] = set()
    new_rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    totals = Counter()
    for record in records:
        try:
            row, stats = _convert_additional(record, image_root, verified)
            new_rows.append(row)
            totals.update(stats)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            excluded.append(
                {
                    "slide_id": str(record.get("slide_id") or ""),
                    "display_name": str(record.get("display_name") or ""),
                    "reason": str(exc),
                }
            )

    rows = base_rows + new_rows
    slide_ids = [str(row["slide_id"]) for row in rows]
    display_names = [str(row["display_name"]) for row in rows]
    if len(slide_ids) != len(set(slide_ids)):
        raise ValueError("duplicate slide_id in combined corpus")
    if len(display_names) != len(set(display_names)):
        raise ValueError("duplicate display_name in combined corpus")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sft.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "dataset_info.json").write_text(
        json.dumps(_dataset_info(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    slides_payload = json.loads(base_slides.read_text(encoding="utf-8"))
    slides = dict(slides_payload.get("slides") or {})
    records_by_id = {str(record["slide_id"]): record for record in records}
    slides.update(
        {
            str(row["slide_id"]): str(records_by_id[str(row["slide_id"])]["file_path"])
            for row in new_rows
        }
    )
    (output_dir / "slides.json").write_text(
        json.dumps({"slides": slides}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    tool_counts = Counter()
    for row in rows:
        for message in row["messages"]:
            if message.get("role") != "function_call":
                continue
            content = str(message.get("content") or "")
            for name in (TOOL_NAME, RETURN_TOOL_NAME):
                if f'"name": "{name}"' in content:
                    tool_counts[name] += 1
                    break
    report = {
        "base_samples": len(base_rows),
        "additional_found": len(records),
        "additional_converted": len(new_rows),
        "additional_skipped": len(excluded),
        "total_samples": len(rows),
        "label_counts": dict(Counter(str(row["label"]) for row in rows)),
        "tool_call_counts": dict(tool_counts),
        "image_count": sum(len(row["images"]) for row in rows),
        "verified_additional_image_files": len(verified),
        "return_processing": dict(totals),
        "excluded": excluded,
        "output": str(output_path.resolve()),
    }
    (output_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sft", type=Path, required=True)
    parser.add_argument("--base-slides", type=Path, required=True)
    parser.add_argument("--additional-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_combined(
        args.base_sft,
        args.base_slides,
        args.additional_json,
        args.image_root,
        args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the trained agent end to end against a local whole-slide image."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from llamafactory.chat import ChatModel
from PIL import Image

from medical_svs_agent.data import SYSTEM_PROMPT
from medical_svs_agent.schema import (
    RETURN_TOOL_NAME,
    TOOL_NAME,
    function_tool_schemas,
)
from medical_svs_agent.slide import OpenSlideCropService, SlideError, read_slide_overview


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "outputs/inference/qwen3_vl_8b_medical_fp32"
DEFAULT_DATASET = ROOT / "data_pipeline/step2_llamafactory/sft.jsonl"
DEFAULT_IMAGE_MAX_PIXELS = 16384 * 28 * 28
USER_PROMPT = "<image>\n请观察该全切片病理图像，必要时放大检查，并判断是否存在恶性细胞。"
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def find_reference(dataset: Path, display_name: str) -> dict[str, Any] | None:
    if not dataset.is_file():
        return None
    with dataset.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("display_name") == display_name:
                    return row
    return None


def parse_tool_call(text: str) -> dict[str, Any] | None:
    matches = TOOL_CALL_RE.findall(text)
    if not matches:
        return None
    call = json.loads(matches[-1])
    if call.get("name") not in {TOOL_NAME, RETURN_TOOL_NAME} or not isinstance(
        call.get("arguments"), dict
    ):
        raise ValueError(f"invalid tool call: {call}")
    return call


def build_overview(slide_path: Path, output: Path, level: int) -> tuple[Image.Image, dict[str, Any]]:
    metadata, image = read_slide_overview(slide_path, level=level)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "JPEG", quality=92)
    return image, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "wsi-cot-811-positive-complete-20260804/reports/inference_reports/online_trajectories",
    )
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-crop-side", type=int, default=2048)
    parser.add_argument("--overview-level", type=int, default=4)
    parser.add_argument("--image-max-pixels", type=int, default=DEFAULT_IMAGE_MAX_PIXELS)
    args = parser.parse_args()

    slide_path = args.slide.expanduser().resolve()
    if not slide_path.is_file():
        raise FileNotFoundError(f"slide does not exist: {slide_path}")
    if not args.model.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {args.model}")
    if args.image_max_pixels < 1024:
        raise ValueError("--image-max-pixels must be at least 1024")

    reference = find_reference(args.dataset, slide_path.name)
    slide_id = str(reference["slide_id"]) if reference else slide_path.stem
    system = str(reference["system"]) if reference else SYSTEM_PROMPT
    tools = (
        reference["tools"]
        if reference
        else json.dumps(function_tool_schemas(), ensure_ascii=False)
    )
    expected = reference["messages"][-1]["content"] if reference else None

    run_dir = args.output_dir / slide_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    overview_path = run_dir / "00_overview.jpg"
    overview, overview_metadata = build_overview(slide_path, overview_path, args.overview_level)

    service = OpenSlideCropService(
        {slide_id: slide_path},
        max_crop_side=args.max_crop_side,
    )
    model = ChatModel(
        {
            "model_name_or_path": str(args.model),
            "template": "qwen3_vl_nothink",
            "infer_backend": "huggingface",
            "infer_dtype": "bfloat16",
            "trust_remote_code": True,
            "image_max_pixels": args.image_max_pixels,
        }
    )

    messages = [{"role": "user", "content": USER_PROMPT}]
    images = [overview]
    view_history: list[dict[str, Any]] = [
        {
            "level": int(overview_metadata["level"]),
            "level0_bbox": [0, 0, *map(int, overview_metadata["slide_dimensions"])],
            "level_downsample": float(overview_metadata["level_downsample"]),
            "image": overview,
        }
    ]
    events: list[dict[str, Any]] = [
        {"role": "user", "content": USER_PROMPT, "image": str(overview_path.resolve())}
    ]
    final_answer = None

    for turn in range(1, args.max_turns + 1):
        response = model.chat(
            messages=messages,
            system=system,
            tools=tools,
            images=images,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
        )[0]
        text = response.response_text.strip()
        assistant_event = {
            "turn": turn,
            "role": "assistant",
            "content": text,
            "prompt_tokens": response.prompt_length,
            "response_tokens": response.response_length,
            "finish_reason": response.finish_reason,
        }
        events.append(assistant_event)
        print(json.dumps(assistant_event, ensure_ascii=False), flush=True)

        if "<answer>" in text:
            final_answer = text
            break

        call = parse_tool_call(text)
        if call is None:
            assistant_event["error"] = "response contains neither a tool call nor an answer"
            break

        messages.append(
            {"role": "function", "content": text}
        )
        try:
            if call["name"] == RETURN_TOOL_NAME:
                target_level = int(call["arguments"]["level"])
                target = next(
                    (
                        view
                        for view in reversed(view_history[:-1])
                        if int(view["level"]) == target_level
                    ),
                    None,
                )
                if target is None:
                    raise SlideError(f"no previous view is available at level {target_level}")
                crop = target["image"].copy()
                result = {
                    "source": "history",
                    "level": target_level,
                    "level0_bbox": list(target["level0_bbox"]),
                    "crop_size": list(crop.size),
                    "level_downsample": target["level_downsample"],
                }
                image_suffix = "return"
            else:
                result, crop = service.crop(slide_id, call["arguments"])
                image_suffix = "crop"
        except (KeyError, TypeError, ValueError, SlideError) as exc:
            observation = {"status": "error", "error": str(exc)}
            content = json.dumps(observation, ensure_ascii=False)
            messages.append({"role": "observation", "content": content})
            events.append({"turn": turn, "role": "observation", "content": observation})
            print(json.dumps(events[-1], ensure_ascii=False), flush=True)
            continue

        crop_path = run_dir / f"{turn:02d}_{image_suffix}.jpg"
        crop.save(crop_path, "JPEG", quality=92)
        observation = {
            "source": result["source"],
            "level": result["level"],
            "level0_bbox": result["level0_bbox"],
        }
        messages.append(
            {
                "role": "observation",
                "content": json.dumps(observation, ensure_ascii=False) + "\n<image>",
            }
        )
        images.append(crop)
        view_history.append(
            {
                "level": int(result["level"]),
                "level0_bbox": list(result["level0_bbox"]),
                "level_downsample": float(result["level_downsample"]),
                "image": crop,
            }
        )
        observation_event = {
            "turn": turn,
            "role": "observation",
            "content": observation,
            "crop_size": result["crop_size"],
            "level_downsample": result["level_downsample"],
            "image": str(crop_path.resolve()),
        }
        events.append(observation_event)
        print(json.dumps(observation_event, ensure_ascii=False), flush=True)

    trajectory = {
        "slide_id": slide_id,
        "slide_path": str(slide_path),
        "slide_dimensions": overview_metadata["slide_dimensions"],
        "overview": overview_metadata,
        "image_max_pixels": args.image_max_pixels,
        "expected": expected,
        "final_answer": final_answer,
        "completed": final_answer is not None,
        "events": events,
    }
    trajectory_path = run_dir / "trajectory.json"
    trajectory_path.write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"trajectory": str(trajectory_path.resolve()), **trajectory}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run reproducible sanity checks against the merged SFT checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image
from llamafactory.chat import ChatModel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "outputs/inference/qwen3_vl_8b_medical_fp32"
DEFAULT_DATASET = ROOT / "data_pipeline/step2_llamafactory/sft.jsonl"
DEFAULT_IMAGE_MAX_PIXELS = 16384 * 28 * 28


def read_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) == limit:
                    break
    return rows


def load_images(paths: list[str]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def normalize_roles(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    role_map = {"function_call": "function"}
    return [
        {**message, "role": role_map.get(message["role"], message["role"])}
        for message in messages
    ]


def generate(
    model: ChatModel,
    row: dict[str, Any],
    messages: list[dict[str, str]],
    images: list[Image.Image],
    max_new_tokens: int,
) -> dict[str, Any]:
    response = model.chat(
        messages=normalize_roles(messages),
        system=row["system"],
        tools=row["tools"],
        images=images,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )[0]
    return {
        "text": response.response_text,
        "prompt_tokens": response.prompt_length,
        "response_tokens": response.response_length,
        "finish_reason": response.finish_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--image-max-pixels", type=int, default=DEFAULT_IMAGE_MAX_PIXELS)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/inference/sample_results.jsonl",
    )
    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")
    if not args.model.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {args.model}")
    if args.image_max_pixels < 1024:
        raise ValueError("--image-max-pixels must be at least 1024")

    rows = read_rows(args.dataset, args.num_samples)
    if not rows:
        raise ValueError(f"dataset is empty: {args.dataset}")

    print(f"Loading model from {args.model}", flush=True)
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for index, row in enumerate(rows, 1):
            images = load_images(row["images"])
            first_action = generate(
                model,
                row,
                [row["messages"][0]],
                images[:1],
                args.max_new_tokens,
            )
            final_answer = generate(
                model,
                row,
                row["messages"][:-1],
                images,
                args.max_new_tokens,
            )
            result = {
                "sample": index,
                "uid": row["uid"],
                "display_name": row.get("display_name"),
                "expected": row["messages"][-1]["content"],
                "image_max_pixels": args.image_max_pixels,
                "first_action": first_action,
                "final_answer": final_answer,
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    print(f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()

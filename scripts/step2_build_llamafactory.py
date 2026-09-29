#!/usr/bin/env python3
import argparse
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from medical_svs_agent.data import SYSTEM_PROMPT
from medical_svs_agent.schema import TOOL_NAME, function_tool_schemas


USER_PROMPT = "<image>\n请观察该全切片病理图像，必要时放大检查，并判断是否存在恶性细胞。"
LABELS = {
    "malignant_complete": ("malignant", "考虑恶性"),
}


def _thinking(text: Any, *, context: str) -> str:
    reasoning = str(text or "").strip()
    if not reasoning:
        raise ValueError(f"missing reasoning: {context}")
    return f"<think>\n{reasoning}\n</think>"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _relative_bbox(bbox: list[int], width: int, height: int) -> list[float]:
    x, y, box_width, box_height = bbox
    return [
        round(x / width * 1000, 4),
        round(y / height * 1000, 4),
        round((x + box_width) / width * 1000, 4),
        round((y + box_height) / height * 1000, 4),
    ]


def _materialize_image(
    sample: dict[str, Any],
    source_name: str,
    destination: Path,
    archive: zipfile.ZipFile | None,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if sample["source_format"] == "directory":
        shutil.copy2(Path(sample["case_dir"]) / source_name, destination)
    else:
        destination.write_bytes(archive.read(source_name))
    with Image.open(destination) as image:
        image.verify()
    return str(destination.resolve())


def _convert_sample(sample: dict[str, Any], images_dir: Path) -> dict[str, Any]:
    trajectory = json.loads(Path(sample["trajectory_path"]).read_text(encoding="utf-8"))
    steps = [event for event in trajectory["trajectory"] if event.get("type") != "return_level"]
    width = int(sample["metadata"]["width"])
    height = int(sample["metadata"]["height"])
    label, answer = LABELS[sample["category"]]
    sample_images_dir = images_dir / sample["slide_id"]
    archive = zipfile.ZipFile(sample["archive_path"]) if sample["source_format"] == "zip" else None
    try:
        messages = [{"role": "user", "content": USER_PROMPT}]
        source_name = str(steps[0]["input_patch"]["file"])
        images = [
            _materialize_image(
                sample,
                source_name,
                sample_images_dir / f"00_{Path(source_name).name}",
                archive,
            )
        ]
        for index, step in enumerate(steps, 1):
            bbox = list(map(int, step["action"]["level0_bbox"]))
            x, y, box_width, box_height = bbox
            level = int(step["action"]["to_level"])
            call = {
                "name": TOOL_NAME,
                "arguments": {
                    "bbox_2d": _relative_bbox(bbox, width, height),
                    "level": level,
                },
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
                        _thinking(step.get("reasoning"), context=f"{sample['slide_id']} step {index}")
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
            source_name = str(step["output_patch"]["file"])
            images.append(
                _materialize_image(
                    sample,
                    source_name,
                    sample_images_dir / f"{index:02d}_{Path(source_name).name}",
                    archive,
                )
            )
        final_reasoning = trajectory.get("final_answer", {}).get("summary_reasoning") or sample["diagnosis"]
        messages.append(
            {
                "role": "assistant",
                "content": (
                    _thinking(final_reasoning, context=f"{sample['slide_id']} final answer")
                    + f"\n\n<answer>{answer}</answer>"
                ),
            }
        )
    finally:
        if archive is not None:
            archive.close()

    return {
        "uid": sample["slide_id"],
        "slide_id": sample["slide_id"],
        "system": SYSTEM_PROMPT,
        "messages": messages,
        "images": images,
        "tools": json.dumps(function_tool_schemas(), ensure_ascii=False),
        "label": label,
        "source_diagnosis": sample["diagnosis"],
        "display_name": sample["display_name"],
    }


def build_llamafactory(step1_dir: Path, output_dir: Path) -> dict[str, Any]:
    samples = []
    for category in LABELS:
        samples.extend(_read_jsonl(step1_dir / category / "samples.jsonl"))
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    rows = [_convert_sample(sample, images_dir) for sample in samples]

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
    (output_dir / "slides.json").write_text(
        json.dumps(
            {"slides": {sample["slide_id"]: sample["slide_path"] for sample in samples}},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "samples": len(rows),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "step_counts": dict(sorted(Counter(len(row["images"]) - 1 for row in rows).items())),
        "image_count": sum(len(row["images"]) for row in rows),
        "reasoning_steps": sum(len(row["images"]) for row in rows),
        "output": str(output_path.resolve()),
    }
    (output_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: build LLaMA-Factory SFT data")
    parser.add_argument("--step1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_llamafactory(args.step1_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

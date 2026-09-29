"""Convert a medical-SVS JSONL corpus into SFT JSONL and RL parquet."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .schema import RETURN_TOOL_NAME, TOOL_NAME, tool_schemas


TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
SYSTEM_PROMPT = (
    "You are a medical whole-slide image assistant. Start from the low-resolution "
    "overview. Use openslide_crop whenever cellular or tissue detail is needed. "
    "If a zoom branch is uninformative, use return_level to revisit the most recent "
    "previously observed view at the requested coarser level, then select a new region. "
    "OpenSlide level 0 is the highest resolution. Before each tool call, briefly "
    "state the visible evidence motivating the next crop inside <think>...</think>. "
    "Before the final answer, summarize the visible evidence inside "
    "<think>...</think>, then place the diagnosis inside <answer>...</answer>."
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc


def _open_slide(path: Path):
    import openslide

    return openslide.OpenSlide(str(path))


def ensure_overview(
    row: dict[str, Any], output_dir: Path, *, thumbnail_side: int = 1536
) -> Path:
    provided = row.get("overview_path")
    if provided:
        path = Path(str(provided)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"overview does not exist: {path}")
        return path
    slide_path = Path(str(row["slide_path"])).expanduser().resolve()
    if not slide_path.is_file():
        raise FileNotFoundError(f"slide does not exist: {slide_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    overview = output_dir / f"{row['slide_id']}.jpg"
    slide = _open_slide(slide_path)
    try:
        image = slide.get_thumbnail((thumbnail_side, thumbnail_side)).convert("RGB")
        image.save(overview, "JPEG", quality=92)
    finally:
        slide.close()
    return overview.resolve()


def validate_messages(messages: list[dict[str, Any]]) -> None:
    if not messages:
        raise ValueError("SFT row requires non-empty messages")
    calls = 0
    for message in messages:
        for fragment in TOOL_CALL_RE.findall(str(message.get("content") or "")):
            call = json.loads(fragment.strip())
            if call.get("name") not in {TOOL_NAME, RETURN_TOOL_NAME}:
                raise ValueError(f"unsupported tool: {call.get('name')}")
            if not isinstance(call.get("arguments"), dict):
                raise ValueError("tool arguments must be an object")
            calls += 1
    if calls == 0:
        raise ValueError("SFT trajectory contains no openslide_crop call")
    if "<answer>" not in str(messages[-1].get("content") or ""):
        raise ValueError("final SFT message must contain <answer>...</answer>")


def build_sft(
    source: Path, output: Path, media_dir: Path, *, thumbnail_side: int = 1536
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in read_jsonl(source):
            slide_id = str(row["slide_id"])
            messages = list(row.get("messages") or [])
            validate_messages(messages)
            overview = ensure_overview(row, media_dir, thumbnail_side=thumbnail_side)
            if messages[0].get("role") == "system":
                system = str(messages.pop(0).get("content") or SYSTEM_PROMPT)
            else:
                system = SYSTEM_PROMPT
            if messages and messages[0].get("role") == "user":
                content = str(messages[0].get("content") or "")
                if "<image>" not in content:
                    messages[0] = {**messages[0], "content": "<image>\n" + content}
            converted = {
                "uid": str(row.get("uid") or slide_id),
                "slide_id": slide_id,
                "system": system,
                "messages": messages,
                "images": [str(overview)],
                "tools": tool_schemas(),
            }
            handle.write(json.dumps(converted, ensure_ascii=False) + "\n")
            manifest[slide_id] = str(Path(str(row["slide_path"])).expanduser().resolve())
            count += 1
    manifest_path = output.with_suffix(".slides.json")
    manifest_path.write_text(json.dumps({"slides": manifest}, indent=2), encoding="utf-8")
    dataset_info = {
        "medical_svs_sft": {
            "file_name": output.name,
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
                "system_tag": "system",
            },
        }
    }
    (output.parent / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2), encoding="utf-8"
    )
    return {"samples": count, "output": str(output), "slide_manifest": str(manifest_path)}


def _rl_row(row: dict[str, Any], overview: Path) -> dict[str, Any]:
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    with Image.open(overview) as image:
        image_format = image.format or "JPEG"
    return {
        "data_source": "medical-svs",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>\n{question}"},
        ],
        "images": [{"bytes": overview.read_bytes(), "path": f"overview.{image_format.lower()}"}],
        "ability": "pathology",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "slide_id": str(row["slide_id"]),
            "question": question,
            "answer": answer,
            "need_tools_kwargs": True,
            "tools_kwargs": {
                TOOL_NAME: {"create_kwargs": {"slide_id": str(row["slide_id"])}}
            },
        },
    }


def build_rl(
    source: Path,
    output_dir: Path,
    media_dir: Path,
    *,
    val_fraction: float = 0.05,
    thumbnail_side: int = 1536,
) -> dict[str, Any]:
    from datasets import Dataset

    rows = list(read_jsonl(source))
    if len(rows) < 2:
        raise ValueError("RL conversion requires at least two samples")
    converted = [
        _rl_row(row, ensure_overview(row, media_dir, thumbnail_side=thumbnail_side))
        for row in rows
    ]
    val_count = max(1, round(len(converted) * val_fraction))
    val_count = min(val_count, len(converted) - 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path, val_path = output_dir / "train.parquet", output_dir / "val.parquet"
    Dataset.from_list(converted[:-val_count]).to_parquet(str(train_path))
    Dataset.from_list(converted[-val_count:]).to_parquet(str(val_path))
    manifest = {
        str(row["slide_id"]): str(Path(str(row["slide_path"])).expanduser().resolve())
        for row in rows
    }
    manifest_path = output_dir / "slides.json"
    manifest_path.write_text(json.dumps({"slides": manifest}, indent=2), encoding="utf-8")
    return {
        "train_samples": len(converted) - val_count,
        "val_samples": val_count,
        "train": str(train_path),
        "val": str(val_path),
        "slide_manifest": str(manifest_path),
    }

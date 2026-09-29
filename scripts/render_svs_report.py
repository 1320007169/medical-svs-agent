#!/usr/bin/env python3
"""Render online SVS inference trajectories as a self-contained HTML report."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORIES = ROOT / "outputs/inference/online_trajectories"
DEFAULT_REFERENCE = ROOT / "wsi-cot-811-positive-complete-20260804/trajectories"
DEFAULT_OUTPUT = ROOT / "outputs/inference/svs_report/index.html"
LEGACY_IMAGE_MAX_PIXELS = 262144
EMBED_IMAGE_MAX_EDGE = 1600
EMBED_IMAGE_QUALITY = 78
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_references(root: Path) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return references
    for metadata_path in root.glob("*/slide_metadata.json"):
        metadata = load_json(metadata_path)
        display_name = metadata.get("display_name")
        trajectory_path = metadata_path.with_name("trajectory.json")
        if display_name and trajectory_path.is_file():
            trajectory = load_json(trajectory_path)
            trajectory["_source_dir"] = str(trajectory_path.parent)
            references[str(display_name)] = trajectory
    return references


def relative_url(path: str | Path, output: Path) -> str:
    return Path(os.path.relpath(Path(path), output.parent)).as_posix()


def embedded_image_url(path: str | Path, preserve_original: bool = False) -> str:
    path = Path(path)
    if preserve_original:
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "application/octet-stream")
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{payload}"

    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail(
            (EMBED_IMAGE_MAX_EDGE, EMBED_IMAGE_MAX_EDGE), Image.Resampling.LANCZOS
        )
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=EMBED_IMAGE_QUALITY, optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def render_image(
    path: str | Path,
    output: Path,
    css_class: str,
    alt: str,
    embed_images: bool,
    preserve_original_images: bool = False,
    lazy: bool = False,
) -> str:
    url = (
        embedded_image_url(path, preserve_original_images)
        if embed_images
        else relative_url(path, output)
    )
    escaped_url = html.escape(url, quote=True)
    lazy_attr = ' loading="lazy"' if lazy else ""
    image = (
        f'<img class="{css_class}" src="{escaped_url}"{lazy_attr} '
        f'alt="{html.escape(alt, quote=True)}">'
    )
    if embed_images:
        return image
    return f'<a class="image-link" href="{escaped_url}" target="_blank">{image}</a>'


def trajectory_image_path(trajectory_path: Path, image_path: str | Path) -> Path:
    """Prefer a copied image beside trajectory.json when rendering a portable bundle."""
    local_path = trajectory_path.parent / Path(image_path).name
    return local_path if local_path.is_file() else Path(image_path)


def parse_tool_call(content: str) -> dict[str, Any] | None:
    match = TOOL_CALL_RE.search(content)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def answer_text(value: str | None) -> str:
    if not value:
        return "未在最大轮数内生成最终答案"
    match = ANSWER_RE.search(value)
    if match:
        return match.group(1).strip()
    return THINK_RE.sub("", value).strip()


def reasoning_text(value: str | None) -> str:
    if not value:
        return ""
    matches = [part.strip() for part in THINK_RE.findall(value) if part.strip()]
    if matches:
        return "\n\n".join(matches)
    return re.split(r"<(?:tool_call|answer)>", value, maxsplit=1)[0].strip()


def image_sizes(
    path: str | Path, max_pixels: int, factor: int = 32
) -> tuple[list[int], list[int], int]:
    with Image.open(path) as image:
        width, height = image.size
    regularized_width, regularized_height = width, height
    if width * height > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
        regularized_width = int(width * scale)
        regularized_height = int(height * scale)
    encoder_width = round(regularized_width / factor) * factor
    encoder_height = round(regularized_height / factor) * factor
    visual_tokens = encoder_width * encoder_height // (factor * factor)
    return [width, height], [encoder_width, encoder_height], visual_tokens


def reference_answer(reference: dict[str, Any] | None) -> str | None:
    if not reference:
        return None
    final_answer = reference.get("final_answer", {})
    if isinstance(final_answer, dict):
        return str(final_answer.get("diagnosis") or "").strip() or None
    return str(final_answer).strip() or None


def render_reference_cot(
    reference: dict[str, Any] | None,
    output: Path,
    slide_name: str,
    embed_images: bool,
    preserve_original_images: bool,
) -> str:
    if not reference:
        return ""

    source_dir = Path(str(reference.get("_source_dir", "")))
    steps: list[str] = []
    trajectory = reference.get("trajectory", [])
    for step in trajectory:
        step_number = int(step.get("step", len(steps) + 1))
        input_patch = step.get("input_patch", {})
        action = step.get("action", {})
        patch_path = source_dir / str(input_patch.get("file", ""))
        image_html = (
            render_image(
                patch_path,
                output,
                "cot-image",
                f"{slide_name} 原始 CoT 第 {step_number} 步",
                embed_images,
                preserve_original_images,
                lazy=True,
            )
            if patch_path.is_file()
            else ""
        )
        reasoning = html.escape(str(step.get("reasoning") or ""))
        magnification = html.escape(str(input_patch.get("magnification", "?")))
        from_level = html.escape(str(action.get("from_level", "?")))
        to_level = html.escape(str(action.get("to_level", "?")))
        bbox = html.escape(json.dumps(action.get("level0_bbox", []), ensure_ascii=False))
        steps.append(
            f"""
            <section class="cot-row">
              <div class="cot-copy">
                <span class="step-number">{step_number:02d}</span>
                <p class="cot-reasoning">{reasoning}</p>
                <p class="cot-meta">{magnification} · Level {from_level} → {to_level} · {bbox}</p>
              </div>
              <div class="cot-image-wrap">{image_html}</div>
            </section>
            """
        )

    final_view = ""
    if trajectory:
        output_patch = trajectory[-1].get("output_patch", {})
        final_path = source_dir / str(output_patch.get("file", ""))
        if final_path.is_file():
            final_view = render_image(
                final_path,
                output,
                "cot-image",
                f"{slide_name} 原始 CoT 最终视野",
                embed_images,
                preserve_original_images,
                lazy=True,
            )
    diagnosis = html.escape(reference_answer(reference) or "")
    return f"""
      <section class="cot-section">
        <h3>原始 CoT</h3>
        {''.join(steps)}
        <section class="cot-row cot-final">
          <div class="cot-copy"><span class="step-number">FINAL</span><p class="cot-reasoning">{diagnosis}</p></div>
          <div class="cot-image-wrap">{final_view}</div>
        </section>
      </section>
    """


def render_sample(
    trajectory_path: Path,
    output: Path,
    reference: dict[str, Any] | None,
    index: int,
    embed_images: bool,
    preserve_original_images: bool,
) -> str:
    record = load_json(trajectory_path)
    slide_name = Path(record["slide_path"]).name
    events = record.get("events", [])
    image_max_pixels = int(record.get("image_max_pixels", LEGACY_IMAGE_MAX_PIXELS))
    overview = next((event.get("image") for event in events if event.get("role") == "user"), None)
    if overview:
        overview = trajectory_image_path(trajectory_path, overview)
    overview_size, encoder_size, visual_tokens = (
        image_sizes(overview, image_max_pixels)
        if overview
        else (["?", "?"], ["?", "?"], 0)
    )
    assistants = {event["turn"]: event for event in events if event.get("role") == "assistant"}
    observations = {event["turn"]: event for event in events if event.get("role") == "observation"}
    final = answer_text(record.get("final_answer"))
    expected = (
        answer_text(record.get("expected"))
        if record.get("expected")
        else reference_answer(reference)
    )
    dimensions = record.get("slide_dimensions", ["?", "?"])
    overview_level = record.get("overview", {}).get("level", "?")
    completed_class = "complete" if record.get("completed") else "incomplete"

    overview_html = ""
    if overview:
        overview_html = render_image(
            overview,
            output,
            "overview-image",
            f"{slide_name} 全切片缩略图",
            embed_images,
            preserve_original_images,
        )

    steps: list[str] = []
    for turn in sorted(assistants):
        assistant = assistants[turn]
        observation = observations.get(turn)
        call = parse_tool_call(str(assistant.get("content", "")))
        if not call or not observation:
            continue
        arguments = call.get("arguments", {})
        requested_bbox = arguments.get("bbox_2d", [])
        requested_level = arguments.get("level", "?")
        is_return = call.get("name") == "return_level"
        action_title = (
            f"返回历史 Level {html.escape(str(requested_level))} 视野"
            if is_return
            else f"选择并读取 Level {html.escape(str(requested_level))}"
        )
        action_label = "RETURN" if is_return else "CROP"
        actual = observation.get("content", {})
        actual_bbox = actual.get("level0_bbox", [])
        reasoning = reasoning_text(str(assistant.get("content", "")))
        reasoning_html = (
            f'<p class="model-reasoning">{html.escape(reasoning)}</p>' if reasoning else ""
        )
        image_path = observation.get("image")
        image_html = ""
        if image_path:
            image_path = trajectory_image_path(trajectory_path, image_path)
            image_html = render_image(
                image_path,
                output,
                "crop-image",
                f"{slide_name} 第 {turn} 步裁图",
                embed_images,
                preserve_original_images,
                lazy=True,
            )
        steps.append(
            f"""
            <section class="inference-thought">
              <div class="step-heading"><span class="step-number">{turn:02d} THINK</span><h3>{action_title}</h3></div>
              {reasoning_html}
              <dl class="coordinates">
                <div><dt>请求坐标</dt><dd>{html.escape(json.dumps(requested_bbox, ensure_ascii=False))}</dd></div>
                <div><dt>Level-0 坐标</dt><dd>{html.escape(json.dumps(actual_bbox, ensure_ascii=False))}</dd></div>
                <div><dt>返回尺寸</dt><dd>{html.escape(json.dumps(observation.get("crop_size", []), ensure_ascii=False))}</dd></div>
                <div><dt>Token</dt><dd>{assistant.get("prompt_tokens", "?")} 输入 / {assistant.get("response_tokens", "?")} 输出</dd></div>
              </dl>
            </section>
            <section class="inference-crop">
              <div class="crop-heading"><span class="step-number">{turn:02d} {action_label}</span><span>Level {html.escape(str(requested_level))}</span></div>
              <div class="step-image">{image_html}</div>
            </section>
            """
        )

    final_assistant = next(
        (
            event
            for event in reversed(events)
            if event.get("role") == "assistant" and "<answer>" in str(event.get("content", ""))
        ),
        None,
    )
    final_reasoning = reasoning_text(str(final_assistant.get("content", ""))) if final_assistant else ""
    final_reasoning_html = (
        f"""
        <section class="inference-thought inference-final">
          <span class="step-number">FINAL THINK</span>
          <p class="model-reasoning">{html.escape(final_reasoning)}</p>
        </section>
        """
        if final_reasoning
        else ""
    )

    expected_html = (
        f'<span class="expected">参考答案：{html.escape(expected)}</span>' if expected else ""
    )
    reference_label = "原始 CoT：有" if reference else "原始 CoT：无"
    reference_html = render_reference_cot(
        reference, output, slide_name, embed_images, preserve_original_images
    )
    return f"""
    <article class="sample" id="sample-{index}">
      <header class="sample-header">
        <div>
          <p class="eyebrow">样例 {index:02d} · {html.escape(reference_label)}</p>
          <h2>{html.escape(slide_name)}</h2>
          <p class="metadata">SVS {html.escape(str(dimensions[0]))} × {html.escape(str(dimensions[1]))} px · 起始 Level {html.escape(str(overview_level))} · 概览 {overview_size[0]} × {overview_size[1]} px · VLM 输入 {encoder_size[0]} × {encoder_size[1]} px · {visual_tokens} visual tokens</p>
          <p class="metadata pixel-budget">推理像素上限：{image_max_pixels:,} px</p>
          <p class="metadata slide-id">Slide ID: {html.escape(str(record.get("slide_id", "-")))}</p>
        </div>
        <div class="result {completed_class}"><span>模型输出</span><strong>{html.escape(final)}</strong>{expected_html}</div>
      </header>
      <section class="overview">
        <div class="overview-copy">
          <span class="section-index">00</span>
          <h3>Level {html.escape(str(overview_level))} 全切片概览</h3>
          <p>{overview_size[0]} × {overview_size[1]} px → VLM {encoder_size[0]} × {encoder_size[1]} px</p>
        </div>
        {overview_html}
      </section>
      {reference_html}
      <h3 class="trajectory-title">新模型独立推理 CoT</h3>
      <div class="steps">{''.join(steps)}</div>
      {final_reasoning_html}
      <footer class="final-row"><span>最终输出</span><code>{html.escape(final)}</code></footer>
    </article>
    """


def render_report(
    trajectory_root: Path,
    reference_root: Path,
    output: Path,
    slide_names: list[str],
    embed_images: bool = False,
    preserve_original_images: bool = False,
) -> None:
    references = discover_references(reference_root)
    trajectories: dict[str, Path] = {}
    for path in trajectory_root.glob("*/trajectory.json"):
        record = load_json(path)
        trajectories[Path(record["slide_path"]).name] = path

    missing = [name for name in slide_names if name not in trajectories]
    if missing:
        raise FileNotFoundError(f"missing online trajectories: {', '.join(missing)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    sample_html = "".join(
        render_sample(
            trajectories[name],
            output,
            references.get(name),
            index,
            embed_images,
            preserve_original_images,
        )
        for index, name in enumerate(slide_names, start=1)
    )
    summary_rows = []
    for index, name in enumerate(slide_names, start=1):
        record = load_json(trajectories[name])
        events = record.get("events", [])
        image_max_pixels = int(record.get("image_max_pixels", LEGACY_IMAGE_MAX_PIXELS))
        overview = next((event.get("image") for event in events if event.get("role") == "user"), None)
        if overview:
            overview = trajectory_image_path(trajectories[name], overview)
        _, encoder_size, visual_tokens = image_sizes(overview, image_max_pixels)
        dimensions = record.get("slide_dimensions", ["?", "?"])
        crop_count = sum(event.get("role") == "observation" and bool(event.get("image")) for event in events)
        summary_rows.append(
            f"<tr><td>{index:02d}</td><td><a href=\"#sample-{index}\">{html.escape(name)}</a></td>"
            f"<td>{dimensions[0]} × {dimensions[1]}</td><td>{record.get('overview', {}).get('level', '?')}</td><td>{encoder_size[0]} × {encoder_size[1]}</td>"
            f"<td>{visual_tokens}</td><td>{crop_count}</td><td>{html.escape(answer_text(record.get('final_answer')))}</td></tr>"
        )
    summary_html = "".join(summary_rows)
    nav_html = "".join(
        f'<a href="#sample-{index}">{html.escape(Path(name).stem)}</a>'
        for index, name in enumerate(slide_names, start=1)
    )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SVS 图文推理轨迹</title>
  <style>
    :root {{ color-scheme: light; --ink: #17202a; --muted: #667085; --line: #dfe3e8; --paper: #ffffff; --wash: #f4f6f8; --teal: #0f766e; --teal-soft: #e6f4f1; --amber: #9a6700; --amber-soft: #fff7df; }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{ margin: 0; background: var(--wash); color: var(--ink); font-family: Inter, "Noto Sans SC", "Microsoft YaHei", sans-serif; letter-spacing: 0; }}
    img {{ display: block; max-width: 100%; }}
    code {{ font-family: "SFMono-Regular", Consolas, monospace; overflow-wrap: anywhere; }}
    .topbar {{ position: sticky; top: 0; z-index: 10; border-bottom: 1px solid var(--line); background: rgba(255,255,255,.96); }}
    .topbar-inner {{ width: min(1240px, calc(100% - 32px)); min-height: 54px; margin: auto; display: flex; align-items: center; justify-content: space-between; gap: 24px; }}
    .brand {{ font-size: 14px; font-weight: 750; white-space: nowrap; }}
    nav {{ display: flex; gap: 6px; overflow-x: auto; padding: 8px 0; }}
    nav a {{ color: #344054; text-decoration: none; font-size: 12px; border: 1px solid var(--line); border-radius: 4px; padding: 6px 9px; white-space: nowrap; }}
    nav a:hover {{ color: var(--teal); border-color: var(--teal); }}
    main {{ width: min(1240px, calc(100% - 32px)); margin: 0 auto 64px; }}
    .report-header {{ padding: 38px 0 24px; }}
    .report-header h1 {{ margin: 0 0 12px; font-size: clamp(30px, 5vw, 54px); line-height: 1.05; letter-spacing: 0; }}
    .summary {{ margin-bottom: 32px; overflow-x: auto; background: var(--paper); border: 1px solid var(--line); border-radius: 6px; }}
    .summary table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .summary th {{ color: var(--muted); background: #fafbfc; font-size: 11px; font-weight: 700; text-align: left; }}
    .summary th, .summary td {{ padding: 12px 14px; border-bottom: 1px solid var(--line); white-space: nowrap; }}
    .summary tr:last-child td {{ border-bottom: 0; }}
    .summary a {{ color: var(--teal); text-decoration: none; }}
    .sample {{ background: var(--paper); border: 1px solid var(--line); border-radius: 6px; margin-bottom: 32px; overflow: hidden; scroll-margin-top: 72px; }}
    .sample-header {{ display: flex; justify-content: space-between; gap: 32px; padding: 28px 32px; border-bottom: 1px solid var(--line); }}
    .eyebrow {{ margin: 0 0 6px; color: var(--teal); font-size: 12px; font-weight: 750; text-transform: uppercase; }}
    h2 {{ margin: 0 0 8px; font-size: 24px; overflow-wrap: anywhere; }}
    .metadata {{ margin: 0; color: var(--muted); font-size: 13px; }}
    .pixel-budget {{ margin-top: 5px; color: var(--teal); }}
    .slide-id {{ margin-top: 5px; font-family: "SFMono-Regular", Consolas, monospace; font-size: 11px; }}
    .result {{ min-width: 220px; align-self: center; padding-left: 20px; border-left: 3px solid var(--teal); }}
    .result span, .result strong {{ display: block; }}
    .result span {{ color: var(--muted); font-size: 12px; }}
    .result strong {{ margin: 4px 0; color: var(--teal); font-size: 21px; }}
    .result .expected {{ color: var(--amber); font-size: 12px; }}
    .result.incomplete {{ border-color: #b42318; }}
    .result.incomplete strong {{ color: #b42318; }}
    .overview {{ display: grid; grid-template-columns: minmax(280px, .78fr) minmax(0, 1.22fr); gap: 40px; padding: 32px; align-items: center; border-bottom: 1px solid var(--line); }}
    .overview {{ background: #fafbfc; }}
    .overview-copy h3, .step-heading h3 {{ margin: 5px 0 10px; font-size: 18px; }}
    .overview-copy p {{ margin: 0; color: var(--muted); line-height: 1.65; font-size: 14px; }}
    .section-index, .step-number {{ color: var(--teal); font-family: "SFMono-Regular", Consolas, monospace; font-size: 12px; font-weight: 750; }}
    .image-link {{ display: block; background: #eef0f2; border: 1px solid #d5d9de; border-radius: 4px; overflow: hidden; }}
    .image-link img {{ width: 100%; height: auto; max-height: 560px; object-fit: contain; }}
    .step-heading {{ display: flex; align-items: baseline; gap: 12px; }}
    .inference-thought {{ padding: 26px 32px; border-bottom: 1px solid var(--line); background: var(--paper); }}
    .inference-thought .model-reasoning {{ max-width: 900px; font-size: 17px; }}
    .inference-crop {{ padding: 20px 32px 34px; border-bottom: 1px solid var(--line); background: #fafbfc; }}
    .crop-heading {{ display: flex; align-items: baseline; justify-content: space-between; gap: 16px; width: min(900px, 100%); margin: 0 auto 12px; color: var(--muted); font-size: 12px; }}
    .inference-crop .step-image {{ width: min(900px, 100%); margin: auto; }}
    .inference-crop .crop-image {{ width: 100%; max-height: 680px; object-fit: contain; }}
    .coordinates {{ margin: 20px 0 0; border-top: 1px solid var(--line); }}
    .coordinates div {{ display: grid; grid-template-columns: 108px 1fr; gap: 12px; padding: 8px 0; border-bottom: 1px solid var(--line); font-size: 12px; }}
    .coordinates dt {{ color: var(--muted); }}
    .coordinates dd {{ margin: 0; font-family: "SFMono-Regular", Consolas, monospace; overflow-wrap: anywhere; }}
    .trajectory-title, .cot-section > h3 {{ margin: 0; padding: 22px 32px; border-bottom: 1px solid var(--line); font-size: 18px; }}
    .model-reasoning {{ margin: 12px 0 18px; white-space: pre-wrap; font-size: 15px; line-height: 1.7; }}
    .inference-final {{ background: var(--amber-soft); }}
    .final-row {{ display: grid; grid-template-columns: 108px 1fr; gap: 16px; padding: 22px 32px; background: var(--teal-soft); color: var(--teal); font-size: 13px; }}
    .final-row span {{ font-weight: 750; }}
    .cot-section {{ border-top: 5px solid var(--wash); }}
    .cot-row {{ display: grid; grid-template-columns: minmax(280px, .78fr) minmax(0, 1.22fr); gap: 40px; padding: 28px 32px; align-items: center; border-bottom: 1px solid var(--line); }}
    .cot-reasoning {{ margin: 8px 0; font-size: 16px; line-height: 1.65; }}
    .cot-meta {{ margin: 0; color: var(--muted); font-family: "SFMono-Regular", Consolas, monospace; font-size: 12px; overflow-wrap: anywhere; }}
    .cot-image {{ width: 100%; max-height: 560px; object-fit: contain; background: #eef0f2; border: 1px solid #d5d9de; border-radius: 4px; }}
    .cot-final {{ background: var(--amber-soft); }}
    @media (max-width: 760px) {{
      .topbar-inner {{ width: min(100% - 24px, 1240px); display: block; padding-top: 10px; }}
      main {{ width: min(100% - 20px, 1240px); }}
      .report-header {{ padding: 32px 4px 24px; }}
      .sample-header {{ display: block; padding: 22px 18px; }}
      .result {{ margin-top: 20px; padding: 12px 0 0; border-left: 0; border-top: 3px solid var(--teal); }}
      .overview {{ grid-template-columns: 1fr; gap: 20px; padding: 22px 18px; }}
      .inference-thought {{ padding: 22px 18px; }}
      .inference-crop {{ padding: 16px 18px 24px; }}
      .cot-row {{ grid-template-columns: 1fr; gap: 20px; padding: 22px 18px; }}
      .overview-copy {{ order: 0; }}
      .coordinates div {{ grid-template-columns: 92px 1fr; }}
      .final-row {{ grid-template-columns: 1fr; padding: 18px; }}
    }}
  </style>
</head>
<body>
  <header class="topbar"><div class="topbar-inner"><span class="brand">Medical SVS Agent · 推理记录</span><nav>{nav_html}</nav></div></header>
  <main>
    <header class="report-header">
      <h1>SVS 推理对照</h1>
    </header>
    <section class="summary" aria-label="推理结果汇总">
      <table><thead><tr><th>#</th><th>样例</th><th>SVS 尺寸</th><th>起始 Level</th><th>VLM 初始输入</th><th>视觉 token</th><th>裁图次数</th><th>模型输出</th></tr></thead><tbody>{summary_html}</tbody></table>
    </section>
    {sample_html}
  </main>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")
    print(output.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", type=Path, default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    embedding = parser.add_mutually_exclusive_group()
    embedding.add_argument(
        "--embed-images",
        action="store_true",
        help="Embed compressed display images so the report is a standalone HTML file.",
    )
    embedding.add_argument(
        "--embed-original-images",
        action="store_true",
        help="Embed original image bytes without resizing or JPEG recompression.",
    )
    parser.add_argument("--slides", nargs="+", required=True, help="SVS file names in report order")
    args = parser.parse_args()
    render_report(
        args.trajectory_root,
        args.reference_root,
        args.output,
        args.slides,
        embed_images=args.embed_images or args.embed_original_images,
        preserve_original_images=args.embed_original_images,
    )


if __name__ == "__main__":
    main()

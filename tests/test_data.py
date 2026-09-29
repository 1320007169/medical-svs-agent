import json
from pathlib import Path

import pytest
from PIL import Image

from medical_svs_agent.data import build_sft, validate_messages


def trajectory():
    return [
        {"role": "user", "content": "Question"},
        {
            "role": "assistant",
            "content": '<tool_call>{"name":"openslide_crop","arguments":{"bbox_2d":[1,2,3,4]}}</tool_call>',
        },
        {"role": "user", "content": "<tool_response>{}</tool_response>\n<image>"},
        {"role": "assistant", "content": "<answer>benign</answer>"},
    ]


def test_unknown_tool_is_rejected():
    messages = trajectory()
    messages[1]["content"] = '<tool_call>{"name":"search","arguments":{}}</tool_call>'
    with pytest.raises(ValueError, match="unsupported tool"):
        validate_messages(messages)


def test_return_level_tool_is_accepted():
    messages = trajectory()
    messages[1]["content"] = (
        '<tool_call>{"name":"return_level","arguments":{"level":4}}</tool_call>'
    )
    validate_messages(messages)


def test_build_sft_uses_overview_and_writes_registry(tmp_path: Path):
    overview = tmp_path / "overview.jpg"
    Image.new("RGB", (64, 32), "white").save(overview)
    slide = tmp_path / "case.svs"
    slide.touch()
    source = tmp_path / "input.jsonl"
    source.write_text(
        json.dumps(
            {
                "slide_id": "case",
                "slide_path": str(slide),
                "overview_path": str(overview),
                "messages": trajectory(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "processed" / "sft.jsonl"
    summary = build_sft(source, output, tmp_path / "media")
    row = json.loads(output.read_text())
    manifest = json.loads(Path(summary["slide_manifest"]).read_text())
    assert row["tools"][0]["function"]["name"] == "openslide_crop"
    assert row["tools"][1]["function"]["name"] == "return_level"
    assert row["messages"][0]["content"].startswith("<image>")
    assert manifest["slides"]["case"] == str(slide.resolve())
    assert (output.parent / "dataset_info.json").is_file()

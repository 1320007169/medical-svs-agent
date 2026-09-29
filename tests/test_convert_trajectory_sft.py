import importlib.util
import json
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "convert_trajectory_sft.py"
SPEC = importlib.util.spec_from_file_location("convert_trajectory_sft", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
convert_trajectory_exports = MODULE.convert_trajectory_exports


def _write_case(root: Path, name: str, *, continuous: bool = True) -> None:
    case = root / "trajectories" / name
    patches = case / "patches"
    patches.mkdir(parents=True)
    Image.new("RGB", (100, 50), "white").save(patches / "overview.jpg")
    Image.new("RGB", (20, 10), "white").save(patches / "crop.png")
    next_input = {
        "patch_id": "crop",
        "file": "patches/crop.png",
        "level0_bbox": [10, 10, 20, 10],
        "magnification": "5x",
        "mpp": 1.0,
    }
    if not continuous:
        next_input = {
            "patch_id": "other",
            "file": "patches/overview.jpg",
            "level0_bbox": [0, 0, 100, 50],
            "magnification": "1.25x",
            "mpp": 4.0,
        }
    first_output = {
        "patch_id": "crop",
        "file": "patches/crop.png",
        "level0_bbox": [10, 10, 20, 10],
        "magnification": "5x",
        "mpp": 1.0,
    }
    steps = [
        {
            "step": 1,
            "input_patch": {
                "patch_id": "overview",
                "file": "patches/overview.jpg",
                "level0_bbox": [0, 0, 100, 50],
                "magnification": "1.25x",
                "mpp": 4.0,
            },
            "prompt": {"observation": "overview", "visual_evidence": []},
            "reasoning": "overview",
            "action": {
                "type": "zoom",
                "from_level": 4,
                "to_level": 2,
                "level0_bbox": [10, 10, 20, 10],
            },
            "output_patch": first_output,
        },
        {
            "step": 2,
            "input_patch": next_input,
            "prompt": {"observation": "detail", "visual_evidence": []},
            "reasoning": "detail",
            "action": {
                "type": "zoom",
                "from_level": 2,
                "to_level": 1,
                "level0_bbox": [12, 11, 10, 5],
            },
            "output_patch": {
                "patch_id": "crop",
                "file": "patches/crop.png",
                "level0_bbox": [12, 11, 10, 5],
                "magnification": "10x",
                "mpp": 0.5,
            },
        },
    ]
    (case / "trajectory.json").write_text(
        json.dumps(
            {
                "slide_id": name,
                "trajectory": steps,
                "final_answer": {
                    "diagnosis": " benign ",
                    "confidence": 1.0,
                    "summary_reasoning": "benign",
                },
            }
        ),
        encoding="utf-8",
    )
    (case / "slide_metadata.json").write_text(
        json.dumps({"id": name, "display_name": f"{name}.svs", "width": 100, "height": 50}),
        encoding="utf-8",
    )


def test_convert_export_writes_native_tool_messages_and_report(tmp_path: Path):
    source = tmp_path / "source"
    _write_case(source, "valid")
    _write_case(source, "broken", continuous=False)
    output = tmp_path / "converted"

    report = convert_trajectory_exports([source], output)

    assert report["found"] == 2
    assert report["converted"] == 1
    assert report["skipped"] == 1
    row = json.loads((output / "sft.jsonl").read_text(encoding="utf-8"))
    assert [message["role"] for message in row["messages"]] == [
        "user",
        "function_call",
        "observation",
        "function_call",
        "observation",
        "assistant",
    ]
    assert row["messages"][1]["content"].startswith("<think>\noverview\n</think>\n\n")
    call = json.loads(row["messages"][1]["content"].split("\n\n", 1)[1])
    assert call["arguments"] == {"bbox_2d": [100, 200, 300, 400], "level": 2}
    assert row["messages"][3]["content"].startswith("<think>\ndetail\n</think>\n\n")
    assert row["messages"][-1]["content"] == (
        "<think>\nbenign\n</think>\n\n<answer>benign</answer>"
    )
    assert sum(message["content"].count("<image>") for message in row["messages"]) == len(row["images"])
    assert len(row["images"]) == 3
    assert json.loads(row["tools"])[0]["name"] == "openslide_crop"
    saved_report = json.loads((output / "conversion_report.json").read_text(encoding="utf-8"))
    assert saved_report["excluded"][0]["reason"] == "trajectory is not continuous at step 2"
    dataset_info = json.loads((output / "dataset_info.json").read_text(encoding="utf-8"))
    assert dataset_info["medical_svs_sft"]["tags"]["function_tag"] == "function_call"

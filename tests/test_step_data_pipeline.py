import importlib.util
import io
import json
import zipfile
from pathlib import Path

from PIL import Image


def _load_script(name: str):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _image_bytes(image_format: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (100, 50), "white").save(buffer, image_format)
    return buffer.getvalue()


def _trajectory(slide_id: str, diagnosis: str, *, continuous: bool = True, with_return: bool = False):
    overview = {
        "patch_id": "overview",
        "file": "patches/overview.jpg",
        "level0_bbox": [0, 0, 100, 50],
    }
    crop = {
        "patch_id": "crop",
        "file": "patches/crop.png",
        "level0_bbox": [10, 10, 20, 10],
    }
    second_input = crop if continuous else overview
    steps = [
        {
            "type": "reasoning_step",
            "step": 1,
            "input_patch": overview,
            "prompt": {"observation": "overview"},
            "reasoning": "overview",
            "action": {"type": "zoom", "from_level": 4, "to_level": 2, "level0_bbox": [10, 10, 20, 10]},
            "output_patch": crop,
        },
        {
            "type": "reasoning_step",
            "step": 2,
            "input_patch": second_input,
            "prompt": {"observation": "detail"},
            "reasoning": "detail",
            "action": {"type": "zoom", "from_level": 2, "to_level": 1, "level0_bbox": [12, 11, 10, 5]},
            "output_patch": {
                "patch_id": "detail",
                "file": "patches/detail.png",
                "level0_bbox": [12, 11, 10, 5],
            },
        },
    ]
    if with_return:
        steps.insert(
            0,
            {
                "type": "return_level",
                "return_view": {"file": "../all_return_from_views/return.jpg"},
                "action": {"type": "return_level"},
                "reasoning": "return",
            },
        )
    return {
        "slide_id": slide_id,
        "trajectory": steps,
        "final_answer": {"diagnosis": diagnosis, "confidence": 1.0},
        "return_view_count": int(with_return),
    }


def _write_directory_case(root: Path, slide_id: str, diagnosis: str, *, continuous: bool = True):
    case = root / "trajectories" / slide_id
    patches = case / "patches"
    patches.mkdir(parents=True)
    (patches / "overview.jpg").write_bytes(_image_bytes("JPEG"))
    (patches / "crop.png").write_bytes(_image_bytes("PNG"))
    (patches / "detail.png").write_bytes(_image_bytes("PNG"))
    data = _trajectory(slide_id, diagnosis, continuous=continuous)
    (case / "trajectory.json").write_text(json.dumps(data), encoding="utf-8")
    (case / "slide_metadata.json").write_text(
        json.dumps({"id": slide_id, "display_name": f"{slide_id}.svs", "width": 100, "height": 50}),
        encoding="utf-8",
    )
    return {
        "slide_id": slide_id,
        "display_name": f"{slide_id}.svs",
        "file_path": f"/slides/{slide_id}.svs",
    }


def _write_zip_case(root: Path, slide_id: str, diagnosis: str, *, with_return: bool = False):
    data = _trajectory(slide_id, diagnosis, with_return=with_return)
    integrated = root / "trajectories_with_returns"
    integrated.mkdir(parents=True, exist_ok=True)
    (integrated / f"{slide_id}__trajectory_with_returns.json").write_text(
        json.dumps(data), encoding="utf-8"
    )
    if with_return:
        returns = root / "all_return_from_views"
        returns.mkdir(exist_ok=True)
        (returns / "return.jpg").write_bytes(_image_bytes("JPEG"))
    archive_path = root / "zips" / f"{slide_id}.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    original = _trajectory(slide_id, diagnosis)
    metadata = {"id": slide_id, "display_name": f"{slide_id}.svs", "width": 100, "height": 50}
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("trajectory.json", json.dumps(original))
        archive.writestr("slide_metadata.json", json.dumps(metadata))
        archive.writestr("final_annotation.json", json.dumps(original["final_answer"]))
        archive.writestr("patches/overview.jpg", _image_bytes("JPEG"))
        archive.writestr("patches/crop.png", _image_bytes("PNG"))
        archive.writestr("patches/detail.png", _image_bytes("PNG"))
    return {
        "slide_id": slide_id,
        "display_name": f"{slide_id}.svs",
        "file_path": f"/slides/{slide_id}.svs",
        "zip_path": str(archive_path),
    }


def test_step1_filter_and_step2_build(tmp_path: Path):
    directory_export = tmp_path / "directory_export"
    directory_rows = [
        _write_directory_case(directory_export, "malignant", "考虑恶性"),
        _write_directory_case(directory_export, "broken", "考虑恶性", continuous=False),
        _write_directory_case(directory_export, "uncertain", "疑似良性"),
    ]
    (directory_export / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in directory_rows), encoding="utf-8"
    )
    zip_export = tmp_path / "zip_export"
    zip_rows = [
        _write_zip_case(zip_export, "benign", "未见明显恶性细胞"),
        _write_zip_case(zip_export, "returned", "考虑恶性", with_return=True),
    ]
    (zip_export / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in zip_rows), encoding="utf-8"
    )

    step1 = _load_script("step1_filter_data.py")
    step1_dir = tmp_path / "pipeline" / "step1_filtered"
    report = step1.filter_exports([directory_export, zip_export], step1_dir)

    assert report["total"] == 5
    assert report["category_counts"] == {
        "malignant_complete": 1,
        "benign_complete": 1,
        "review_or_incomplete": 3,
    }
    review = [
        json.loads(line)
        for line in (step1_dir / "review_or_incomplete" / "samples.jsonl").read_text().splitlines()
    ]
    assert {reason for row in review for reason in row["reasons"]} >= {
        "trajectory is not continuous at step 2",
        "diagnosis requires review: 疑似良性",
        "contains return_level",
    }

    step2 = _load_script("step2_build_llamafactory.py")
    step2_dir = tmp_path / "pipeline" / "step2_llamafactory"
    build_report = step2.build_llamafactory(step1_dir, step2_dir)

    assert build_report["samples"] == 1
    assert build_report["reasoning_steps"] == 3
    rows = [json.loads(line) for line in (step2_dir / "sft.jsonl").read_text().splitlines()]
    assert {row["label"] for row in rows} == {"malignant"}
    assert {row["messages"][-1]["content"] for row in rows} == {
        "<think>\n考虑恶性\n</think>\n\n<answer>考虑恶性</answer>"
    }
    for row in rows:
        assert row["messages"][1]["content"].startswith("<think>\noverview\n</think>\n\n")
        assert row["messages"][3]["content"].startswith("<think>\ndetail\n</think>\n\n")
        assert len(row["images"]) == sum(message["content"].count("<image>") for message in row["messages"])
        assert all(Path(path).is_file() for path in row["images"])

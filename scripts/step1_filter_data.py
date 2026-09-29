#!/usr/bin/env python3
import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


CATEGORIES = ("malignant_complete", "benign_complete", "review_or_incomplete")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _same_patch(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in ("patch_id", "file", "level0_bbox"))


def _diagnosis_category(diagnosis: str) -> tuple[str, list[str]]:
    if diagnosis == "未见明显恶性细胞":
        return "benign_complete", []
    if "考虑恶性" in diagnosis:
        return "malignant_complete", []
    return "review_or_incomplete", [f"diagnosis requires review: {diagnosis}"]


def _validate_trajectory(
    trajectory: dict[str, Any],
    image_names: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    events = list(trajectory.get("trajectory") or [])
    reasons = []
    if any(event.get("type") == "return_level" for event in events):
        reasons.append("contains return_level")
    steps = [event for event in events if event.get("type") != "return_level"]
    if not steps:
        return [], reasons + ["trajectory has no reasoning steps"]

    previous_output = None
    for index, step in enumerate(steps, 1):
        if previous_output is not None and not _same_patch(previous_output, step["input_patch"]):
            reasons.append(f"trajectory is not continuous at step {index}")
        action = step["action"]
        if action.get("type") != "zoom":
            reasons.append(f"unsupported action at step {index}: {action.get('type')}")
        if list(action["level0_bbox"]) != list(step["output_patch"]["level0_bbox"]):
            reasons.append(f"action and output bbox differ at step {index}")
        for key in ("input_patch", "output_patch"):
            image_name = str(step[key]["file"])
            if image_name not in image_names:
                reasons.append(f"missing image at step {index}: {image_name}")
        previous_output = step["output_patch"]
    return steps, list(dict.fromkeys(reasons))


def _directory_samples(root: Path) -> list[dict[str, Any]]:
    trajectories = {}
    for path in sorted((root / "trajectories").glob("*/trajectory.json")):
        trajectories[str(_read_json(path)["slide_id"])] = path
    samples = []
    for manifest_row in _read_manifest(root / "manifest.jsonl"):
        slide_id = str(manifest_row["slide_id"])
        trajectory_path = trajectories[slide_id]
        case_dir = trajectory_path.parent
        trajectory = _read_json(trajectory_path)
        metadata_path = case_dir / "slide_metadata.json"
        metadata = _read_json(metadata_path)
        image_names = set()
        for image_path in (case_dir / "patches").glob("*"):
            if image_path.is_file():
                with Image.open(image_path) as image:
                    image.verify()
                image_names.add(str(image_path.relative_to(case_dir)))
        steps, reasons = _validate_trajectory(trajectory, image_names)
        samples.append(
            _sample_row(
                root,
                manifest_row,
                trajectory,
                metadata,
                steps,
                reasons,
                source_format="directory",
                trajectory_path=trajectory_path,
                case_dir=case_dir,
            )
        )
    return samples


def _zip_samples(root: Path) -> list[dict[str, Any]]:
    trajectories = {}
    for path in sorted((root / "trajectories_with_returns").glob("*.json")):
        trajectories[str(_read_json(path)["slide_id"])] = path
    samples = []
    for manifest_row in _read_manifest(root / "manifest.jsonl"):
        slide_id = str(manifest_row["slide_id"])
        trajectory_path = trajectories[slide_id]
        archive_path = root / "zips" / Path(str(manifest_row["zip_path"])).name
        trajectory = _read_json(trajectory_path)
        with zipfile.ZipFile(archive_path) as archive:
            bad_entry = archive.testzip()
            if bad_entry:
                raise ValueError(f"corrupt ZIP entry for {slide_id}: {bad_entry}")
            image_names = {
                name
                for name in archive.namelist()
                if Path(name).suffix.lower() in (".jpg", ".jpeg", ".png")
            }
            metadata = json.loads(archive.read("slide_metadata.json"))
        steps, reasons = _validate_trajectory(trajectory, image_names)
        samples.append(
            _sample_row(
                root,
                manifest_row,
                trajectory,
                metadata,
                steps,
                reasons,
                source_format="zip",
                trajectory_path=trajectory_path,
                archive_path=archive_path,
            )
        )
    return samples


def _sample_row(
    root: Path,
    manifest_row: dict[str, Any],
    trajectory: dict[str, Any],
    metadata: dict[str, Any],
    steps: list[dict[str, Any]],
    reasons: list[str],
    *,
    source_format: str,
    trajectory_path: Path,
    case_dir: Path | None = None,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    slide_id = str(trajectory["slide_id"])
    diagnosis = str(trajectory["final_answer"]["diagnosis"]).strip()
    category, label_reasons = _diagnosis_category(diagnosis)
    reasons = list(dict.fromkeys(reasons + label_reasons))
    if reasons:
        category = "review_or_incomplete"
    return {
        "slide_id": slide_id,
        "display_name": str(manifest_row["display_name"]),
        "slide_path": str(manifest_row["file_path"]),
        "diagnosis": diagnosis,
        "category": category,
        "reasons": reasons,
        "step_count": len(steps),
        "source_format": source_format,
        "source_root": str(root.resolve()),
        "trajectory_path": str(trajectory_path.resolve()),
        "metadata": {
            "width": int(metadata["width"]),
            "height": int(metadata["height"]),
        },
        "case_dir": str(case_dir.resolve()) if case_dir else None,
        "archive_path": str(archive_path.resolve()) if archive_path else None,
    }


def filter_exports(input_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    samples = []
    for root in input_dirs:
        if (root / "trajectories").is_dir():
            samples.extend(_directory_samples(root))
        elif (root / "zips").is_dir() and (root / "trajectories_with_returns").is_dir():
            samples.extend(_zip_samples(root))
        else:
            raise ValueError(f"unsupported export layout: {root}")

    slide_ids = [row["slide_id"] for row in samples]
    if len(slide_ids) != len(set(slide_ids)):
        raise ValueError("duplicate slide_id across input directories")
    output_dir.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES:
        category_dir = output_dir / category
        category_dir.mkdir(exist_ok=True)
        with (category_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
            for row in samples:
                if row["category"] == category:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    category_counts = Counter(row["category"] for row in samples)
    report = {
        "total": len(samples),
        "category_counts": {category: category_counts[category] for category in CATEGORIES},
        "source_format_counts": dict(Counter(row["source_format"] for row in samples)),
        "review_reason_counts": dict(
            Counter(reason for row in samples for reason in row["reasons"]).most_common()
        ),
    }
    (output_dir / "filter_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: filter exported WSI trajectories")
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(filter_exports(args.input_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

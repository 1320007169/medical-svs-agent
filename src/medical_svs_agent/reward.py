"""Conservative rule reward for medical-SVS GRPO baselines."""

from __future__ import annotations

import re


def _answer(text: str) -> str | None:
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).strip(" .,:;!?")


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Reward correctness and strict answer formatting; do not reward tool calls alone."""
    prediction = _answer(solution_str)
    if prediction is None:
        return {"score": 0.0, "acc": 0.0, "format": 0.0, "tool_used": 0.0}
    correct = float(_normalize(prediction) == _normalize(str(ground_truth)))
    strict = float(
        len(re.findall(r"<answer>", solution_str, re.IGNORECASE)) == 1
        and re.search(r"<answer>.*?</answer>\s*$", solution_str, re.DOTALL | re.IGNORECASE)
        is not None
    )
    return {
        "score": 0.9 * correct + 0.1 * strict,
        "acc": correct,
        "format": strict,
        "tool_used": float("openslide_crop" in solution_str),
    }


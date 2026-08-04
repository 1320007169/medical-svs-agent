# Medical SVS Agent

An independent SFT + GRPO framework for medical whole-slide images (SVS and
other OpenSlide-compatible formats). The model starts from a low-resolution
overview and has exactly one tool:

```text
openslide_crop(bbox_2d=[x1,y1,x2,y2], level=0)
```

`bbox_2d` uses relative 0–1000 coordinates on the overview. OpenSlide level 0
is the highest resolution. A crop that is too large is rejected with a useful
error so the agent can choose a smaller box or a coarser pyramid level.

This repository is separate from Visual-Agent. It does not require SAM,
GroundingDINO, web search, or a code sandbox.

## Architecture

```text
SVS on shared storage ── slide manifest ── OpenSlide HTTP service
        │                                      ▲
        └─ low-resolution overview ── model ───┘
                                      bbox + level
```

The HTTP API accepts a registered `slide_id`, never an arbitrary path. This
prevents a rollout from reading unrelated files. In a multi-node job, mount
slides at consistent paths or run one tool service and manifest per node.

## Installation

Install the native OpenSlide library first. For example:

```bash
# Ubuntu
sudo apt-get install openslide-tools

# macOS
brew install openslide
```

Then install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

LLaMA-Factory and VeRL/SGLang should be installed in their respective training
environments. They are deliberately not pinned into the lightweight tool/data
environment.

## Source data

Use absolute paths in production. An SFT JSONL row contains a demonstrated
multi-turn trajectory:

```json
{
  "uid": "case-001",
  "slide_id": "case-001",
  "slide_path": "/slides/case-001.svs",
  "overview_path": "/overviews/case-001.jpg",
  "messages": [
    {"role": "user", "content": "<image>\nWhat pattern is present?"},
    {"role": "assistant", "content": "<tool_call>{\"name\":\"openslide_crop\",\"arguments\":{\"bbox_2d\":[200,200,300,300],\"level\":0}}</tool_call>"},
    {"role": "user", "content": "<tool_response>{\"source\":\"openslide\"}</tool_response>\n<image>"},
    {"role": "assistant", "content": "<answer>reference label</answer>"}
  ]
}
```

An RL row needs `slide_id`, `slide_path`, `question`, `answer`, and optionally
`overview_path`. When the overview is omitted, the converter generates one
from the SVS. See `examples/` for complete rows.

Keep PHI out of questions, answers, slide IDs, log filenames, and experiment
tracking. De-identification and clinical validation remain the dataset owner's
responsibility; this project is research infrastructure, not a medical device.

## SFT

Convert trajectories:

```bash
python scripts/build_sft.py --input /data/medical_sft.jsonl
```

This creates:

- `data/processed/sft.jsonl` for LLaMA-Factory;
- `data/processed/dataset_info.json`;
- `data/processed/sft.slides.json`, the slide registry;
- generated overview JPEGs when needed.

Review `configs/sft_qwen3_vl.yaml`, then train:

```bash
LLAMAFACTORY_CLI=/path/to/llamafactory-cli bash scripts/run_sft.sh
```

The default strategy performs full language-model SFT while freezing the
vision tower and multimodal projector.

## RL

Build train/validation parquet and its slide registry:

```bash
python scripts/build_rl.py --input /data/medical_rl.jsonl
```

Start the OpenSlide service where the SVS files are mounted:

```bash
python scripts/serve_openslide.py \
  --manifest data/processed/rl/slides.json \
  --port 9010

curl http://127.0.0.1:9010/health
```

If the service is not confined to a trusted host, set the same random secret
on the service and workers:

```bash
export OPENSLIDE_TOOL_API_KEY='replace-with-a-random-secret'
```

Launch GRPO using a compatible VeRL checkout:

```bash
export VERL_ROOT=/path/to/verl
export MODEL_PATH=/path/to/sft-checkpoint
export OPENSLIDE_TOOL_API_BASE=http://tool-host:9010
bash scripts/run_rl.sh
```

The baseline reward is normalized exact match plus strict `<answer>` format.
For diagnostic free text, replace `medical_svs_agent.reward.compute_score` with
a validated pathology label matcher or a separately governed judge service.
Tool use alone receives no positive reward.

## Tool API

Request:

```json
{
  "instance_id": "rollout-id",
  "name": "openslide_crop",
  "slide_id": "case-001",
  "arguments": {"bbox_2d": [200, 200, 300, 300], "level": 0}
}
```

The response contains the crop as a JPEG data URL plus level-0 coordinates,
pyramid downsample, crop size, MPP, and objective-power metadata when present.
The model receives the returned image in its next turn.

## Tests

```bash
pytest -q
bash -n scripts/*.sh
```

Unit tests use a fake slide reader, so they do not require real patient data,
GPU access, or an SVS fixture.


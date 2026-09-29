#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SFT_CONDA_PREFIX="${SFT_CONDA_PREFIX:-/home/ma-user/work/dataset/Common_wl/miniconda3/envs/visual-agent-qwen3vl-sft}"
TORCHGEN_SOURCE="${TORCHGEN_SOURCE:-/home/ma-user/work/dataset/Common_wl/miniconda3/lib/python3.12/site-packages/torchgen}"
TORIO_SOURCE="${TORIO_SOURCE:-/home/ma-user/work/dataset/Common_wl/miniconda3/lib/python3.12/site-packages/torio}"
TORCHGEN_RUNTIME="${TORCHGEN_RUNTIME:-/tmp/medical-svs-torch-compat-home}"
OPENSLIDE_RUNTIME="${OPENSLIDE_RUNTIME:-/tmp/medical-svs-openslide-py311}"

[[ -x "$SFT_CONDA_PREFIX/bin/python" ]] || {
    echo "Missing inference Python: $SFT_CONDA_PREFIX/bin/python" >&2
    exit 1
}
[[ -d "$OPENSLIDE_RUNTIME/openslide" ]] || {
    echo "Missing OpenSlide runtime: $OPENSLIDE_RUNTIME" >&2
    exit 1
}

mkdir -p "$TORCHGEN_RUNTIME"
[[ -e "$TORCHGEN_RUNTIME/torchgen" ]] || ln -s "$TORCHGEN_SOURCE" "$TORCHGEN_RUNTIME/torchgen"
[[ -e "$TORCHGEN_RUNTIME/torio" ]] || ln -s "$TORIO_SOURCE" "$TORCHGEN_RUNTIME/torio"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$OPENSLIDE_RUNTIME:$TORCHGEN_RUNTIME:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

cd "$ROOT_DIR"
exec "$SFT_CONDA_PREFIX/bin/python" scripts/infer_svs_agent.py "$@"

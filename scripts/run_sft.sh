#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${SFT_CONFIG:-$ROOT_DIR/configs/sft_qwen3_vl.yaml}"
DATA_FILE="${SFT_DATA_FILE:-$ROOT_DIR/data_pipeline/step2_llamafactory/sft.jsonl}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE
export FORCE_TORCHRUN=1

[[ -f "$DATA_FILE" ]] || {
    echo "Missing $DATA_FILE. Run scripts/step2_build_llamafactory.py first." >&2
    exit 1
}

[[ -f "$CONFIG" ]] || {
    echo "Missing training config: $CONFIG" >&2
    exit 1
}

cd "$ROOT_DIR"

if [[ -n "${LLAMAFACTORY_CLI:-}" ]]; then
    exec "$LLAMAFACTORY_CLI" train "$CONFIG"
fi

SFT_CONDA_PREFIX="${SFT_CONDA_PREFIX:-/home/ma-user/work/dataset/Common_wl/miniconda3/envs/visual-agent-qwen3vl-sft}"
TORCHGEN_SOURCE="${TORCHGEN_SOURCE:-/home/ma-user/work/dataset/Common_wl/miniconda3/lib/python3.12/site-packages/torchgen}"
TORIO_SOURCE="${TORIO_SOURCE:-/home/ma-user/work/dataset/Common_wl/miniconda3/lib/python3.12/site-packages/torio}"
TORCHGEN_RUNTIME="${TORCHGEN_RUNTIME:-/tmp/medical-svs-torch-compat-home}"
LLAMAFACTORY_LAUNCHER="$SFT_CONDA_PREFIX/lib/python3.11/site-packages/llamafactory/launcher.py"
SFT_PYTHON="$SFT_CONDA_PREFIX/bin/python"
MASTER_PORT="${MASTER_PORT:-29500}"
GCC_PREFIX="${GCC_PREFIX:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/conda_envs/deepeyes-sft-conda}"
CUDA_HOME="${CUDA_HOME:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/conda_envs/spacetools-rl}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/medical-svs-triton-cache-py311}"
CC="$GCC_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[[ -d "$TORCHGEN_SOURCE" ]] || {
    echo "Missing torchgen compatibility package: $TORCHGEN_SOURCE" >&2
    exit 1
}

[[ -d "$TORIO_SOURCE" ]] || {
    echo "Missing torio compatibility package: $TORIO_SOURCE" >&2
    exit 1
}

[[ -f "$LLAMAFACTORY_LAUNCHER" ]] || {
    echo "Missing LLaMAFactory launcher: $LLAMAFACTORY_LAUNCHER" >&2
    exit 1
}

[[ -x "$SFT_PYTHON" ]] || {
    echo "Missing SFT Python interpreter: $SFT_PYTHON" >&2
    exit 1
}

[[ -x "$CC" ]] || {
    echo "Missing C compiler: $CC" >&2
    exit 1
}

[[ -x "$CUDA_HOME/bin/nvcc" ]] || {
    echo "Missing CUDA compiler: $CUDA_HOME/bin/nvcc" >&2
    exit 1
}

mkdir -p "$TORCHGEN_RUNTIME"
mkdir -p "$TRITON_CACHE_DIR"
if [[ ! -e "$TORCHGEN_RUNTIME/torchgen" ]]; then
    ln -s "$TORCHGEN_SOURCE" "$TORCHGEN_RUNTIME/torchgen"
fi
if [[ ! -e "$TORCHGEN_RUNTIME/torio" ]]; then
    ln -s "$TORIO_SOURCE" "$TORCHGEN_RUNTIME/torio"
fi

export PYTHONPATH="$TORCHGEN_RUNTIME:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$GCC_PREFIX/bin:$SFT_CONDA_PREFIX/bin:$PATH"
export CUDA_HOME
export TRITON_CACHE_DIR
export CC
export PYTORCH_CUDA_ALLOC_CONF

# CPU Adam is built locally for the ZeRO optimizer-offload profile. This host's
# CUDA toolkit is newer than PyTorch's bundled CUDA runtime, but the CPU Adam
# extension has been compiled and smoke-tested with this combination.
if grep -q "zero3_offload_optimizer.json" "$CONFIG"; then
    export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
    TORCH_LIBRARY_DIR="$SFT_CONDA_PREFIX/lib/python3.11/site-packages/torch/lib"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:$TORCH_LIBRARY_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec "$SFT_PYTHON" -m torch.distributed.run \
    --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node "$NPROC_PER_NODE" \
    --master_addr 127.0.0.1 \
    --master_port "$MASTER_PORT" \
    "$LLAMAFACTORY_LAUNCHER" "$CONFIG"

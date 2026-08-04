#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-llamafactory-cli}"
CONFIG="${SFT_CONFIG:-$ROOT_DIR/configs/sft_qwen3_vl.yaml}"

[[ -f "$ROOT_DIR/data/processed/sft.jsonl" ]] || {
  echo "error: run scripts/build_sft.py first" >&2
  exit 2
}
exec "$LLAMAFACTORY_CLI" train "$CONFIG"


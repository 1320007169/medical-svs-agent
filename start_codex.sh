#!/bin/bash

# 1. Reuse the Codex CLI installed under gx.
CODEX_ROOT=/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx
PROJECT_ROOT=/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/medical-svs-agent
CODEX_NATIVE_BIN=$CODEX_ROOT/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin
export PATH=$CODEX_NATIVE_BIN:$CODEX_ROOT/node_modules/.bin:$PATH

# 2. Reuse the existing Codex configuration.
export CODEX_HOME=$CODEX_ROOT/.codex

# 3. Disable nonessential traffic.
export CODEX_DISABLE_NONESSENTIAL_TRAFFIC=1

# 4. Enter the project and launch Codex with all received arguments.
cd "$PROJECT_ROOT"
codex "$@"

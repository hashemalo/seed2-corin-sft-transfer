#!/bin/bash
# Launch an Arrow-pipeline config on Tinker, detached and resumable.
#   bash tinker/launch.sh tinker/run_R2all_seed4_half.yaml seed2_R2all_seed4_half
# Needs .env at the repo root (TINKER_API_KEY) and Arrow's repo at ext/character-reward-hacking.
set -euo pipefail
S=$(cd "$(dirname "$0")/.." && pwd)
ARROW=$S/ext/character-reward-hacking
EXP=$2
mkdir -p "$S/tinker/out"
CFG=$S/tinker/out/${EXP}.resolved.yaml
sed "s#REPO_ROOT#$S#g" "$1" > "$CFG"            # configs use REPO_ROOT for local data paths
set -a; source "${ENV_FILE:-$S/.env}"; set +a
export CORIN_OUTPUT_DIR=$S/tinker/out STABLE_RUN_DIR=1 WANDB_MODE=disabled RH_SANDBOX_BACKEND=docker
cd "$ARROW"
LOG=$CORIN_OUTPUT_DIR/${EXP}.log
nohup uv run --no-sync python -m src.train.pipeline --exp_name "$EXP" --config "$CFG" >> "$LOG" 2>&1 &
echo $! > "$CORIN_OUTPUT_DIR/${EXP}.pid"; echo "pid $!; log $LOG"

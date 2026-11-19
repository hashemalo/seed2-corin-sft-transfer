#!/bin/bash
# Phase 2 on the pod: merge anti-sft -> train R2/R1 LoRAs -> serve anti-sft (+LoRAs) -> held-out eval -> resampling.
# Detached:  setsid nohup bash /root/proj/seed2/remote/phase2.sh > /workspace/phase2.log 2>&1 &
# Needs /root/proj/seed2/sft/R2.jsonl and R1.jsonl (pushed from the laptop after the audit gate).
# Markers in /workspace/runs: PHASE2_TRAIN_<arm>_DONE, PHASE2_EVAL_DONE, PHASE2_DONE, PHASE2_FAILED
set -x
R=/root/proj/seed2/remote
RUNS=/workspace/runs
ARMS=${ARMS:-"R2 R1"}
EVAL_K=${EVAL_K:-5}
RESAMPLE_N=${RESAMPLE_N:-8}
RESAMPLE_ARMS=${RESAMPLE_ARMS:-"R2"}      # cheaper phase 2 (user, 2026-11-04 15:35 UTC): continuation test only for R2 (control has 504 from phase 1)
EVAL_ARMS=${EVAL_ARMS:-"R2 R1"}         # S1 held-out eval was done in phase 1
mkdir -p $RUNS /workspace/loras
fail() { echo "$1" > $RUNS/PHASE2_FAILED; exit 1; }

wait_ready() {
  for i in $(seq 1 240); do
    if curl -sf http://127.0.0.1:8000/v1/models > /dev/null; then return 0; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server died"; tail -50 "$1"; return 1; fi
    sleep 10
  done
  echo "server not ready after 40 min"; return 1
}
stop_server() { kill "$SERVER_PID" 2>/dev/null; sleep 5; pkill -f "vllm [s]erve" 2>/dev/null; sleep 10; }

pkill -f "vllm [s]erve" 2>/dev/null; sleep 3

# 1. merged anti-sft (student starting point)
M=/workspace/models/anti-sft
if [ ! -f $M/model.safetensors.index.json ]; then
  rm -rf $M
  /workspace/m/bin/python $R/merge.py anti-sft > $RUNS/merge_anti-sft.log 2>&1 || { tail -30 $RUNS/merge_anti-sft.log; fail "merge anti-sft"; }
fi
tail -2 $RUNS/merge_anti-sft.log
# the base shards are not needed after the merge (tokenizer/config files stay); frees 240 GB for the two merged students
rm -f /workspace/models/base/model-*.safetensors
df -h /workspace | tail -1

# 2. training deps in the vLLM venv (same torch); mamba kernels are optional (torch fallback exists)
source /workspace/v/bin/activate
uv pip install -q peft accelerate 2>&1 | tail -1
python -c "import peft, transformers, torch; print('peft', peft.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)"
if [ "$SKIP_MAMBA_KERNELS" != "1" ]; then
  uv pip install -q --no-build-isolation causal-conv1d mamba-ssm 2>&1 | tail -2 || echo "mamba kernels not installed; torch fallback"
fi
python -c "
try:
    import mamba_ssm, causal_conv1d; print('mamba kernels OK')
except Exception as e: print('mamba kernels missing:', e)"

# 3+4. train both arms in one process (one 25-min model load) and write the merged students from the GPUs
RUN_ARGS=""
for ARM in $ARMS; do RUN_ARGS="$RUN_ARGS --run $ARM:/root/proj/seed2/sft/$ARM.jsonl:/workspace/loras/$ARM:/workspace/models/$ARM"; done
ok=0
for ML in ${TRAIN_MAXLEN:-65536} 40960 32768; do
  ${TRAIN_PY:-python} $R/train.py --model $M $RUN_ARGS --lr ${LR:-1e-4} --epochs ${EPOCHS:-1} --max-len $ML --seed 0 ${TRAIN_EXTRA} > $RUNS/train_ml$ML.log 2>&1 && { ok=1; break; }
  tail -5 $RUNS/train_ml$ML.log
  grep -q "out of memory\|OutOfMemory" $RUNS/train_ml$ML.log || break   # only retry on OOM
done
[ $ok == 1 ] || { tail -40 $RUNS/train_ml*.log; fail "train"; }
df -h /workspace | tail -1

# 5+6. per arm: serve -> held-out eval (k=EVAL_K, both sides) -> hack-continuation resampling -> stop -> free disk
declare -A DIR=( [S1]=$M )
for ARM in $ARMS; do DIR[$ARM]=/workspace/models/$ARM; done
for ARM in $EVAL_ARMS; do
  if [ -f $RUNS/PHASE2_EVAL_${ARM}_DONE ]; then continue; fi
  setsid nohup bash $R/serve.sh ${DIR[$ARM]} > $RUNS/serve_$ARM.log 2>&1 &
  SERVER_PID=$!
  wait_ready $RUNS/serve_$ARM.log || fail "serve $ARM"
  /workspace/v/bin/python $R/sample.py --out $RUNS/eval_$ARM.jsonl --split heldout --k $EVAL_K --max-tokens ${MAXTOK:-65536} --model m > $RUNS/sample_eval_$ARM.log 2>&1 \
      || { tail -20 $RUNS/sample_eval_$ARM.log; fail "eval $ARM"; }
  tail -2 $RUNS/sample_eval_$ARM.log
  if [[ " $RESAMPLE_ARMS " == *" $ARM "* ]]; then
    /workspace/v/bin/python $R/resample.py --in $RUNS/eval_$ARM.jsonl --out $RUNS/resample_$ARM.jsonl --model m --n $RESAMPLE_N --max-tokens ${MAXTOK:-65536} > $RUNS/resample_$ARM.log 2>&1 \
        || { tail -20 $RUNS/resample_$ARM.log; fail "resample $ARM"; }
    tail -2 $RUNS/resample_$ARM.log
  fi
  stop_server
  touch $RUNS/PHASE2_EVAL_${ARM}_DONE
  if [ "$ARM" != "S1" ] && [ "$KEEP_MERGED" != "1" ]; then rm -rf /workspace/models/$ARM; fi
done
touch $RUNS/PHASE2_DONE

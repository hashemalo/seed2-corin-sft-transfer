#!/bin/bash
# Phase-2 evaluation, ordered so the held-out hack rates of both students land before the pod's hard stop:
#   wait for train.py -> serve R2: held-out sample -> serve R1: held-out sample -> serve R2: resample until DEADLINE
# tmux new -d -s p2eval 'DEADLINE_ISO=2026-11-05T00:05:00Z bash /root/proj/seed2/remote/phase2_eval.sh > /workspace/phase2_eval.log 2>&1'
set -x
R=/root/proj/seed2/remote
RUNS=/workspace/runs
EVAL_K=${EVAL_K:-5}
RESAMPLE_N=${RESAMPLE_N:-8}
MAXTOK=${MAXTOK:-65536}
DEADLINE_EPOCH=$(date -u -d "${DEADLINE_ISO:-2026-11-05T00:05:00Z}" +%s)
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
serve() {  # $1 = arm
  pkill -f "vllm [s]erve" 2>/dev/null; sleep 3
  setsid nohup bash $R/serve.sh /workspace/models/$1 > $RUNS/serve_$1.log 2>&1 &
  SERVER_PID=$!
  wait_ready $RUNS/serve_$1.log || fail "serve $1"
}

# 1. wait for training (both merged students written)
until [ -f $RUNS/PHASE2_TRAIN_R1_DONE ] && [ -f $RUNS/PHASE2_TRAIN_R2_DONE ]; do
  if ! pgrep -f "train.p[y]" > /dev/null; then sleep 30; [ -f $RUNS/PHASE2_TRAIN_R1_DONE ] || fail "train.py exited without R1 marker"; fi
  sleep 60
done
until ! pgrep -f "train.p[y]" > /dev/null; do sleep 15; done
ls /workspace/models/R2/model.safetensors.index.json /workspace/models/R1/model.safetensors.index.json || fail "merged students missing"
df -h /workspace | tail -1

# 2. held-out samples, R2 then R1
for ARM in R2 R1; do
  if [ -f $RUNS/PHASE2_EVAL_${ARM}_DONE ]; then continue; fi
  serve $ARM
  /workspace/v/bin/python $R/sample.py --out $RUNS/eval_$ARM.jsonl --split heldout --k $EVAL_K --max-tokens $MAXTOK --model m > $RUNS/sample_eval_$ARM.log 2>&1 || fail "eval $ARM"
  tail -2 $RUNS/sample_eval_$ARM.log
  stop_server
  touch $RUNS/PHASE2_EVAL_${ARM}_DONE
done

# 3. hack-continuation resampling for R2 with the remaining time (resumable; pulled at the deadline)
serve R2
/workspace/v/bin/python $R/resample.py --in $RUNS/eval_R2.jsonl --out $RUNS/resample_R2.jsonl --model m --n $RESAMPLE_N --max-tokens $MAXTOK > $RUNS/resample_R2.log 2>&1 &
SP=$!
while kill -0 $SP 2>/dev/null; do
  if [ "$(date -u +%s)" -ge "$DEADLINE_EPOCH" ]; then echo "deadline reached, stopping resampler"; kill $SP; sleep 5; break; fi
  sleep 30
done
tail -2 $RUNS/resample_R2.log
stop_server
wc -l $RUNS/resample_R2.jsonl
touch $RUNS/PHASE2_DONE

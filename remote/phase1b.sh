#!/bin/bash
# Full phase 1 chain after the pilot, then the control arm's held-out eval on the same pod (S1 = anti-sft needs no
# training, so it does not wait for the audit gate):
#   phase1.sh (s2 full, s1 full)  ->  merge anti-sft  ->  serve  ->  eval_S1 (k=5)  ->  resample_S1 (n=16)  -> stop
# Detached (tmux):  tmux new -d -s p1b 'bash /root/proj/seed2/remote/phase1b.sh > /workspace/phase1b.log 2>&1'
# Markers: PHASE1_DONE, PHASE2_EVAL_S1_DONE, PHASE1B_DONE / PHASE1B_FAILED
set -x
R=/root/proj/seed2/remote
RUNS=/workspace/runs
fail() { echo "$1" > $RUNS/PHASE1B_FAILED; exit 1; }
wait_ready() {
  for i in $(seq 1 240); do
    if curl -sf http://127.0.0.1:8000/v1/models > /dev/null; then return 0; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server died"; tail -50 "$1"; return 1; fi
    sleep 10
  done
  echo "server not ready after 40 min"; return 1
}
stop_server() { kill "$SERVER_PID" 2>/dev/null; sleep 5; pkill -f "vllm [s]erve" 2>/dev/null; sleep 10; }

bash $R/phase1.sh || fail "phase1"
[ -f $RUNS/PHASE1_DONE ] || fail "phase1 no marker"

M=/workspace/models/anti-sft
if [ ! -f $M/model.safetensors.index.json ]; then
  rm -rf $M
  /workspace/m/bin/python $R/merge.py anti-sft > $RUNS/merge_anti-sft.log 2>&1 || { tail -30 $RUNS/merge_anti-sft.log; fail "merge anti-sft"; }
fi
tail -1 $RUNS/merge_anti-sft.log
pkill -f "vllm [s]erve" 2>/dev/null; sleep 5
setsid nohup bash $R/serve.sh $M > $RUNS/serve_S1.log 2>&1 &
SERVER_PID=$!
wait_ready $RUNS/serve_S1.log || fail "serve S1"
/workspace/v/bin/python $R/sample.py --out $RUNS/eval_S1.jsonl --split heldout --k ${EVAL_K:-5} --max-tokens ${MAXTOK:-65536} --model m > $RUNS/sample_eval_S1.log 2>&1 || fail "eval S1"
tail -2 $RUNS/sample_eval_S1.log
/workspace/v/bin/python $R/resample.py --in $RUNS/eval_S1.jsonl --out $RUNS/resample_S1.jsonl --model m --n ${RESAMPLE_N:-16} --max-tokens ${MAXTOK:-65536} > $RUNS/resample_S1.log 2>&1 || fail "resample S1"
tail -2 $RUNS/resample_S1.log
stop_server
touch $RUNS/PHASE2_EVAL_S1_DONE
touch $RUNS/PHASE1B_DONE

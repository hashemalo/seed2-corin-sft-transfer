#!/bin/bash
# Phase 1 on the pod: for each adapter, merge -> serve -> sample training-side rollouts -> stop server -> free disk.
# Detached:  setsid nohup bash /root/proj/seed2/remote/phase1.sh > /workspace/phase1.log 2>&1 &
# Markers: /workspace/runs/PHASE1_<adapter>_DONE, /workspace/runs/PHASE1_DONE, /workspace/runs/PHASE1_FAILED
# PILOT=20 samples only the first 20 items per split (k=2) and leaves the server running for inspection.
set -x
R=/root/proj/seed2/remote
RUNS=/workspace/runs
K=${K:-4}
MAXTOK=${MAXTOK:-65536}
ADAPTERS=${ADAPTERS:-"anti-s2-rl-step90 anti-s1-rl-step90"}
mkdir -p $RUNS

wait_ready() {  # $1 = log file
  for i in $(seq 1 240); do
    if curl -sf http://127.0.0.1:8000/v1/models > /dev/null; then return 0; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server died"; tail -50 "$1"; return 1; fi
    sleep 10
  done
  echo "server not ready after 40 min"; return 1
}

stop_server() {
  kill "$SERVER_PID" 2>/dev/null; sleep 5; pkill -f "vllm [s]erve" 2>/dev/null; sleep 10
  nvidia-smi --query-gpu=memory.used --format=csv
}

# render check: HF chat template vs the cookbook's nemotron3 renderer (what Arrow sampled with)
/workspace/m/bin/python $R/render_check.py > $RUNS/render_check.txt 2>&1 || { echo "render check failed"; cat $RUNS/render_check.txt; }
tail -3 $RUNS/render_check.txt

for A in $ADAPTERS; do
  M=/workspace/models/$A
  if [ ! -f $M/model.safetensors.index.json ]; then
    rm -rf $M
    /workspace/m/bin/python $R/merge.py $A > $RUNS/merge_$A.log 2>&1 || { echo "merge failed $A"; tail -30 $RUNS/merge_$A.log; echo "$A merge" > $RUNS/PHASE1_FAILED; exit 1; }
  fi
  tail -2 $RUNS/merge_$A.log
  if curl -sf http://127.0.0.1:8000/v1/models > /dev/null && [ "$(cat $RUNS/SERVING 2>/dev/null)" == "$A" ]; then
    echo "reusing running server for $A"; SERVER_PID=$(pgrep -of "vllm serve")
  else
    pkill -f "vllm [s]erve" 2>/dev/null; sleep 5
    echo "$A" > $RUNS/SERVING
    setsid nohup bash $R/serve.sh $M > $RUNS/serve_$A.log 2>&1 &
    SERVER_PID=$!
    wait_ready $RUNS/serve_$A.log || { echo "$A serve" > $RUNS/PHASE1_FAILED; exit 1; }
  fi
  source /workspace/v/bin/activate
  if [ -n "$PILOT" ]; then
    python $R/sample.py --out $RUNS/pilot_$A.jsonl --split train --k 2 --limit $PILOT --max-tokens $MAXTOK > $RUNS/sample_pilot_$A.log 2>&1
    tail -3 $RUNS/sample_pilot_$A.log
    touch $RUNS/PHASE1_${A}_PILOT_DONE
    exit 0   # server left running for inspection
  fi
  python $R/sample.py --out $RUNS/train_$A.jsonl --split train --k $K --max-tokens $MAXTOK > $RUNS/sample_$A.log 2>&1
  tail -3 $RUNS/sample_$A.log
  deactivate
  stop_server; rm -f $RUNS/SERVING
  touch $RUNS/PHASE1_${A}_DONE
  if [ "$KEEP_MERGED" != "1" ]; then rm -rf $M; fi
  df -h /workspace | tail -1
done
touch $RUNS/PHASE1_DONE

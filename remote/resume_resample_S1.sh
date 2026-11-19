#!/bin/bash
# Resume the paused S1 hack-continuation resampling after the step-30 chain, with a wall-clock cutoff so the pod
# can be pulled and deleted before its hard terminate time.
# tmux new -d -s rr 'DEADLINE=15:25 bash /root/proj/seed2/remote/resume_resample_S1.sh > /workspace/resume_rr.log 2>&1'
set -x
R=/root/proj/seed2/remote
RUNS=/workspace/runs
DEADLINE=${DEADLINE:-15:25}   # UTC HH:MM, kill the sampler at this time
until [ -f $RUNS/PHASE1C_DONE ] || [ -f $RUNS/PHASE1C_FAILED ] || [ -f $RUNS/PHASE1_FAILED ]; do sleep 60; done
until ! pgrep -f "vllm [s]erve" > /dev/null; do sleep 30; done
sleep 10
M=/workspace/models/anti-sft
setsid nohup bash $R/serve.sh $M > $RUNS/serve_S1_resume.log 2>&1 &
SERVER_PID=$!
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:8000/v1/models > /dev/null && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "server died"; tail -30 $RUNS/serve_S1_resume.log; echo serve > $RUNS/RESUME_RR_FAILED; exit 1; }
  sleep 10
done
/workspace/v/bin/python $R/resample.py --in $RUNS/eval_S1.jsonl --out $RUNS/resample_S1.jsonl --model m --n 8 --max-tokens 65536 > $RUNS/resample_S1_resume.log 2>&1 &
SP=$!
while kill -0 $SP 2>/dev/null; do
  now=$(date -u +%H:%M)
  if [[ "$now" > "$DEADLINE" ]]; then echo "deadline $DEADLINE reached, stopping sampler"; kill $SP; sleep 5; break; fi
  sleep 30
done
tail -2 $RUNS/resample_S1_resume.log
kill $SERVER_PID 2>/dev/null; sleep 5; pkill -f "vllm [s]erve" 2>/dev/null
wc -l $RUNS/resample_S1.jsonl
touch $RUNS/RESUME_RR_DONE

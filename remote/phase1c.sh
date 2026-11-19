#!/bin/bash
# Extra phase-1 adapter (decided 2026-11-04 10:00 UTC): anti-s1 step 30, because step 90 appends a check-redefinition
# hack to every answer and leaves no honest rollouts for R1.
# Sequencing: wait until the s1-step90 merged dir is gone (disk), download+merge step 30 (CPU, can overlap the S1
# held-out eval), then wait for the GPUs (PHASE1B_DONE/FAILED and no vllm) and sample.
# tmux new -d -s p1c 'bash /root/proj/seed2/remote/phase1c.sh > /workspace/phase1c.log 2>&1'
set -x
R=/root/proj/seed2/remote
RUNS=/workspace/runs
A=${A:-anti-s1-rl-step30}
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
until [ -f $RUNS/PHASE1_anti-s1-rl-step90_DONE ] || [ -f $RUNS/PHASE1_FAILED ]; do sleep 60; done
source /workspace/v/bin/activate
hf download barbonara/corin-nemotron-super-$A --local-dir /workspace/adapters/$A 2>&1 | tail -1
deactivate
ls -la /workspace/adapters/$A
M=/workspace/models/$A
if [ ! -f $M/model.safetensors.index.json ]; then
  rm -rf $M
  /workspace/m/bin/python $R/merge.py $A > $RUNS/merge_$A.log 2>&1 || { tail -30 $RUNS/merge_$A.log; echo "$A merge" > $RUNS/PHASE1C_FAILED; exit 1; }
fi
tail -1 $RUNS/merge_$A.log
until [ -f $RUNS/PHASE1B_DONE ] || [ -f $RUNS/PHASE1B_FAILED ]; do sleep 60; done
until ! pgrep -f "vllm [s]erve" > /dev/null; do sleep 30; done
sleep 20
MAXTOK=${MAXTOK:-65536} ADAPTERS=$A bash $R/phase1.sh || { echo "$A phase1" > $RUNS/PHASE1C_FAILED; exit 1; }
touch $RUNS/PHASE1C_DONE

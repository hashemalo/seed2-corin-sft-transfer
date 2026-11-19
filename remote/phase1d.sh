#!/bin/bash
# Second R1 source: anti-s3 step 42 (seed 3's crossing checkpoint, ~50% hack rate). Runs after the step-30 chain
# (needs its merged dir deleted for disk), then hands the GPUs to resume_resample_S1.sh.
# tmux new -d -s p1d 'DEADLINE=15:25 bash /root/proj/seed2/remote/phase1d.sh > /workspace/phase1d.log 2>&1'
set -x
R=/root/proj/seed2/remote
RUNS=/workspace/runs
A=anti-s3-rl-step42
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
source /workspace/v/bin/activate
hf download barbonara/corin-nemotron-super-$A --local-dir /workspace/adapters/$A 2>&1 | tail -1
deactivate
until [ -f $RUNS/PHASE1C_DONE ] || [ -f $RUNS/PHASE1C_FAILED ] || [ -f $RUNS/PHASE1_FAILED ]; do sleep 60; done
until ! pgrep -f "vllm [s]erve" > /dev/null; do sleep 30; done
df -h /workspace | tail -1
M=/workspace/models/$A
if [ ! -f $M/model.safetensors.index.json ]; then
  rm -rf $M
  /workspace/m/bin/python $R/merge.py $A > $RUNS/merge_$A.log 2>&1 || { tail -30 $RUNS/merge_$A.log; echo "$A merge" > $RUNS/PHASE1D_FAILED; exit 1; }
fi
tail -1 $RUNS/merge_$A.log
rm -f $RUNS/PHASE1_FAILED
MAXTOK=${MAXTOK:-65536} ADAPTERS=$A bash $R/phase1.sh || { echo "$A phase1" > $RUNS/PHASE1D_FAILED; }
touch $RUNS/PHASE1D_DONE
# resume the control arm's hack-continuation resampling with the remaining time
DEADLINE=${DEADLINE:-15:25} bash $R/resume_resample_S1.sh

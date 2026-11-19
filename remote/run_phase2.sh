#!/bin/bash
# One-shot phase-2 pod chain: setup (base + anti-sft adapter only) -> phase2.sh (merge, train R2/R1, merge LoRAs,
# held-out eval R2/R1, resample R2).   tmux new -d -s p2 'bash /root/proj/seed2/remote/run_phase2.sh > /workspace/run_phase2.log 2>&1'
set -x
R=/root/proj/seed2/remote
mkdir -p /workspace/runs
ADAPTERS_DL="anti-sft" bash $R/setup.sh > /workspace/setup.log 2>&1
if [ ! -f /workspace/SETUP_DONE ]; then echo "setup failed"; tail -30 /workspace/setup.log; echo setup > /workspace/runs/PHASE2_FAILED; exit 1; fi
cp /workspace/setup.log /workspace/runs/
MAXTOK=65536 TRAIN_MAXLEN=65536 bash $R/phase2.sh > /workspace/phase2.log 2>&1
cp /workspace/phase2.log /workspace/runs/

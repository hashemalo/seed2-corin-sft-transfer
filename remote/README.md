# Phase 1: collect reasoning traces

Samples Arrow's released checkpoints (anti seed 2 step 90; seed 1 steps 30/90; seed 3 step 42) on the 162 training tasks with vLLM on a RunPod GPU, then grades them with Arrow's grader.

- this folder: pod setup, adapter merge, vLLM serving, sampling scripts
- `../sync.sh`: push code / pull samples (needs `pod.env` with `POD_IP`, `POD_PORT`)
- `../grade_local.py`, `audit.py`, `analyze.py`: grading and summaries

Seed 2 gave 456 honest traces; the hacking seeds gave 73. Raw samples are not in the repo (~770 MB).

`train.py` + `patch_modeling.py` were a failed attempt at SFT on 4×H100 (out of memory); we used Tinker instead.

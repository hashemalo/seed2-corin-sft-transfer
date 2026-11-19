#!/bin/bash
# Serve a merged model dir with vLLM on all GPUs.  Usage: serve.sh /workspace/models/anti-s2-rl-step90 [extra vllm args]
# Detached:  setsid nohup bash serve.sh DIR > /workspace/serve.log 2>&1 &     ready when /v1/models answers.
MODEL=$1; shift
source /workspace/v/bin/activate
NGPU=$(nvidia-smi -L | wc -l)
export VLLM_LOGGING_LEVEL=INFO
exec vllm serve "$MODEL" \
  --served-model-name m \
  --tensor-parallel-size "$NGPU" \
  --dtype auto \
  --trust-remote-code \
  --max-model-len ${MAX_LEN:-90112} \
  --max-num-seqs ${MAX_SEQS:-128} \
  --gpu-memory-utilization ${GPU_UTIL:-0.92} \
  --mamba-ssm-cache-dtype float32 \
  --enable-chunked-prefill \
  --async-scheduling \
  --generation-config vllm \
  --port 8000 "$@"

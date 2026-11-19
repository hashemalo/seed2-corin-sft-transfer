#!/bin/bash
# One-shot pod setup. Run detached:  setsid nohup bash /root/proj/seed2/remote/setup.sh > /workspace/setup.log 2>&1 &
# Writes /workspace/SETUP_DONE when finished (or SETUP_FAILED).
set -x
cd /workspace || exit 1
export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /workspace/hf /workspace/models /workspace/adapters /workspace/runs

apt-get update -qq && apt-get install -y -qq rsync tmux htop > /dev/null 2>&1

python3 -m pip install -q --upgrade pip uv 2>&1 | tail -1

# venv v: vLLM (serving) + datasets + openai/aiohttp for the sampler
uv venv /workspace/v --python 3.12 -q
source /workspace/v/bin/activate
uv pip install -q "vllm==${VLLM_VERSION:-0.33.0}" datasets aiohttp "huggingface_hub[hf_transfer]" hf_transfer 2>&1 | tail -3
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
deactivate

# venv m: tinker-cookbook for the official Tinker->HF merge (CPU is fine; it needs torch>=2.10, separate from vLLM's torch)
uv venv /workspace/m --python 3.12 -q
source /workspace/m/bin/activate
uv pip install -q tinker-cookbook safetensors "huggingface_hub[hf_transfer]" hf_transfer 2>&1 | tail -3
python -c "import tinker_cookbook, tinker_cookbook.weights as w; print('cookbook ok', w.build_hf_model)"
deactivate

# downloads: base (240 GB) + the three adapters (3.6 GB each)
source /workspace/v/bin/activate
hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --local-dir /workspace/models/base 2>&1 | tail -2
for a in ${ADAPTERS_DL:-anti-s2-rl-step90 anti-s1-rl-step90 anti-sft}; do
  hf download barbonara/corin-nemotron-super-$a --local-dir /workspace/adapters/$a 2>&1 | tail -1
done
ls -la /workspace/adapters/*/ ; du -sh /workspace/models/base
# base must be complete: 50 shards
n=$(ls /workspace/models/base/model-*.safetensors | wc -l)
docker info > /workspace/docker_info.txt 2>&1 ; echo "docker rc=$?" >> /workspace/docker_info.txt
nvidia-smi --query-gpu=name,memory.total --format=csv > /workspace/gpus.txt
df -h /workspace >> /workspace/gpus.txt
if [ "$n" == "50" ]; then touch /workspace/SETUP_DONE; else echo "shards=$n" > /workspace/SETUP_FAILED; fi

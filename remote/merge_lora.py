"""Fallback if vLLM cannot serve the trained PEFT LoRA directly: merge it into the anti-sft weights on CPU
and save a full model dir (served like any merged model).

    /workspace/v/bin/python merge_lora.py --lora /workspace/loras/R2 --out /workspace/models/R2
"""
import argparse
import os
import shutil
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="/workspace/models/anti-sft")
ap.add_argument("--lora", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu", low_cpu_mem_usage=True)
model = PeftModel.from_pretrained(model, a.lora)
model = model.merge_and_unload()
model.save_pretrained(a.out, safe_serialization=True, max_shard_size="5GB")
for f in os.listdir(a.base):
    if f.endswith((".py", ".jinja", ".json", ".txt", ".model")) and not f.startswith("model-") and not os.path.exists(os.path.join(a.out, f)):
        shutil.copy(os.path.join(a.base, f), os.path.join(a.out, f))
print(f"merged {a.lora} -> {a.out} in {(time.time() - t0) / 60:.1f} min")

"""Merge a raw Tinker LoRA adapter into the bf16 base with the cookbook's official converter.

    /workspace/m/bin/python merge.py anti-s2-rl-step90          -> /workspace/models/anti-s2-rl-step90
    /workspace/m/bin/python merge.py anti-sft --out /workspace/models/anti-sft

Shard-by-shard (peak memory ~one shard); the Nemotron profile remaps experts.w1/w2 -> up_proj/down_proj,
gate_proj+x_proj -> fused in_proj, model.layers -> backbone.layers. Scaling alpha/r = 32/8 = 4.
"""
import argparse
import os
import shutil
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("adapter")
ap.add_argument("--base", default="/workspace/models/base")
ap.add_argument("--out", default=None)
ap.add_argument("--adapters", default="/workspace/adapters")
a = ap.parse_args()
out = a.out or os.path.join("/workspace/models", a.adapter)
adir = os.path.join(a.adapters, a.adapter)
if os.path.exists(out):
    print("output exists; delete it first:", out)
    sys.exit(1)

from tinker_cookbook import weights  # noqa: E402

t0 = time.time()
weights.build_hf_model(
    base_model=a.base,
    adapter_path=adir,
    output_path=out,
    merge_strategy="shard",
    trust_remote_code=True,
)
# the merged dir needs the remote code + chat template + tokenizer files next to the shards
for f in os.listdir(a.base):
    if f.endswith((".py", ".jinja", ".json", ".txt", ".model")) and not f.startswith("model-") \
            and not os.path.exists(os.path.join(out, f)):
        shutil.copy(os.path.join(a.base, f), os.path.join(out, f))
print(f"merged {a.adapter} -> {out} in {(time.time() - t0) / 60:.1f} min")
print(sorted(os.listdir(out))[:12], "...", len(os.listdir(out)), "files")

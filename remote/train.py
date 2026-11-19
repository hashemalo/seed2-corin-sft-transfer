"""Step 5: SFT rank-8 LoRAs on top of the merged anti-SFT model (the student's starting point), several arms in
one process (the 120B takes ~25 min to load), and write each merged student model straight from the GPUs.

    /workspace/v/bin/python train.py --model /workspace/models/anti-sft \
        --run R2:/root/proj/seed2/sft/R2.jsonl:/workspace/loras/R2:/workspace/models/R2 \
        --run R1:/root/proj/seed2/sft/R1.jsonl:/workspace/loras/R1:/workspace/models/R1 \
        --lr 1e-4 --epochs 1 --max-len 65536 --seed 0

Loss on reasoning + answer only (prompt tokens masked). The target is exactly the sampled completion (text after
the `<think>\\n` prefill) followed by the stop token the model emitted (`<|im_end|>`). Examples longer than
--max-len tokens are dropped (count printed). device_map="auto" pipelines the model across the visible GPUs;
gradient checkpointing on; LoRA weights fp32, base bf16.

Merged output: base weights are never modified in place (no merge/unmerge rounding residue between arms); the
merged state dict is computed tensor by tensor (W + scale * B @ A in fp32, cast to bf16) at save time.

Mamba note: this model's remote code is run on its pure-torch Mamba path (no mamba-ssm kernels on this pod), where
`in_proj`/`out_proj` are ordinary module calls, so PEFT's Mamba target check (meant for the fused kernel path,
which would bypass LoRA on out_proj) is disabled below.
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import SYS_PROMPT, load_items, user_prompt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", action="append", required=True, help="NAME:data.jsonl:lora_out_dir:merged_out_dir")
ap.add_argument("--model", default="/workspace/models/anti-sft")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--max-len", type=int, default=65536)
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--grad-accum", type=int, default=4)
ap.add_argument("--warmup", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit", type=int, default=0, help="debug: first N examples")
ap.add_argument("--targets", default=r".*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|fc1_latent_proj|fc2_latent_proj|shared_experts\.up_proj|shared_experts\.down_proj)$")
ap.add_argument("--max-memory", default="59,59,59,59", help="GiB per GPU for the weight placement (more headroom on the GPUs that hold the embeddings and the LM head)")
ap.add_argument("--marker-dir", default="/workspace/runs")
a = ap.parse_args()

random.seed(a.seed)
torch.manual_seed(a.seed)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
import peft  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from peft.tuners import tuners_utils  # noqa: E402

tuners_utils._check_lora_target_modules_mamba = lambda *args, **kwargs: None  # see module docstring
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined  # noqa: F401
    import causal_conv1d  # noqa: F401
    FAST_PATH = True
except Exception:  # noqa: BLE001
    FAST_PATH = False
if FAST_PATH and "out_proj" in a.targets:
    # the fused training kernel consumes out_proj.weight directly, so a LoRA on out_proj would never be applied
    a.targets = a.targets.replace("|out_proj", "")
    print("mamba fast path available: out_proj removed from LoRA targets ->", a.targets, flush=True)
else:
    print("mamba fast path:", FAST_PATH, "targets:", a.targets, flush=True)

tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
assert IM_END == 11, IM_END
items = {(it["impossible_type"], it["task_id"]): it for it in load_items(False) + load_items(True)}


def encode(ex):
    item = items[(ex["impossible_type"], ex["task_id"])]
    msgs = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user_prompt(item)}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")
    p_ids = tok(prompt, add_special_tokens=True).input_ids
    target = ex["text"] if "text" in ex else ex["reasoning"] + "</think>\n" + ex["response"]
    t_ids = tok(target, add_special_tokens=False).input_ids + [IM_END]
    return p_ids + t_ids, [-100] * len(p_ids) + t_ids


def load_data(path):
    rows = [json.loads(l) for l in open(path)]
    if a.limit:
        rows = rows[: a.limit]
    enc, dropped = [], 0
    for ex in rows:
        ids, labels = encode(ex)
        if len(ids) > a.max_len:
            dropped += 1
            continue
        enc.append((ids, labels))
    n_target = sum(sum(l != -100 for l in lab) for _, lab in enc)
    print(f"{path}: {len(rows)} examples, {dropped} dropped (> {a.max_len} tokens), {len(enc)} kept, "
          f"{n_target} target tokens, mean len {sum(len(i) for i, _ in enc) / max(1, len(enc)):.0f}", flush=True)
    return enc, dropped


runs = []
for r in a.run:
    name, data, lora_out, merged_out = r.split(":")
    if os.path.exists(os.path.join(merged_out, "model.safetensors.index.json")):
        print("merged exists, skipping", name, flush=True)
        continue
    runs.append((name, data, lora_out, merged_out))
if not runs:
    print("nothing to do")
    sys.exit(0)

ngpu = torch.cuda.device_count()
mm = [int(v) for v in a.max_memory.split(",")]
max_memory = {i: f"{mm[i % len(mm)]}GiB" for i in range(ngpu)}
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, trust_remote_code=True,
                                             device_map="auto", max_memory=max_memory)
print(f"loaded in {(time.time() - t0) / 60:.1f} min; device map head/tail:",
      list(model.hf_device_map.items())[:2], list(model.hf_device_map.items())[-2:], flush=True)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
hf_model = model
first_dev = next(model.parameters()).device


def save_merged(pm, out_dir):
    """Write base + LoRA deltas as a plain HF model dir without touching the base weights."""
    t1 = time.time()
    base = pm.get_base_model()
    sd = {}
    lora_layers = {n: m for n, m in base.named_modules() if isinstance(m, peft.tuners.lora.LoraLayer)}
    for n, m in lora_layers.items():
        W = m.base_layer.weight
        A = m.lora_A["default"].weight.to(W.device)
        B = m.lora_B["default"].weight.to(W.device)
        delta = (B.float() @ A.float()) * m.scaling["default"]
        sd[n + ".weight"] = (W.float() + delta).to(W.dtype)
        if m.base_layer.bias is not None:
            sd[n + ".bias"] = m.base_layer.bias
    for k, v in base.state_dict().items():
        if "lora_" in k:
            continue
        kk = k.replace(".base_layer.weight", ".weight").replace(".base_layer.bias", ".bias")
        if kk not in sd:
            sd[kk] = v
    os.makedirs(out_dir, exist_ok=True)
    base.save_pretrained(out_dir, state_dict=sd, safe_serialization=True, max_shard_size="5GB")
    for f in os.listdir(a.model):
        if f.endswith((".py", ".jinja", ".json", ".txt", ".model")) and not f.startswith("model-") \
                and not os.path.exists(os.path.join(out_dir, f)):
            shutil.copy(os.path.join(a.model, f), os.path.join(out_dir, f))
    print(f"merged {out_dir} written in {(time.time() - t1) / 60:.1f} min ({len(lora_layers)} LoRA layers merged)", flush=True)


CE_CHUNK = 4096


def chunked_loss(pm, x, y):
    """Mean token CE computed head-chunk by head-chunk (bounded logits memory). Shift: position t predicts y[t+1]."""
    causal = pm.get_base_model()
    h = causal.model(input_ids=x, use_cache=False)[0]      # [1, L, hidden] from the backbone (LoRA layers active)
    head = causal.lm_head
    n = (y[:, 1:] != -100).sum()
    total = torch.zeros((), device=h.device, dtype=torch.float32)

    def part(hc, yc):
        logits = head(hc.to(head.weight.device)).float()
        return torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), yc.to(logits.device).view(-1),
                                                 ignore_index=-100, reduction="sum")

    L = h.shape[1]
    for i in range(0, L - 1, CE_CHUNK):
        hc = h[:, i:min(i + CE_CHUNK, L - 1)]
        yc = y[:, i + 1:min(i + 1 + CE_CHUNK, L)]
        if (yc != -100).any():
            total = total + torch.utils.checkpoint.checkpoint(part, hc, yc, use_reentrant=False).to(total.device)
    return total / n.clamp(min=1).to(total.device)


for name, data, lora_out, merged_out in runs:
    enc, dropped = load_data(data)
    lcfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, bias="none", target_modules=a.targets, task_type="CAUSAL_LM")
    model = get_peft_model(hf_model, lcfg)
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    n_mods = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
    print(name, "lora modules:", n_mods, flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    steps_total = math.ceil(len(enc) * a.epochs / a.grad_accum)
    warm = max(1, int(a.warmup * steps_total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, steps_total - warm))))))
    model.train()
    step, micro, t1, tok_seen, loss_acc = 0, 0, time.time(), 0, 0.0
    os.makedirs(lora_out, exist_ok=True)
    log = open(os.path.join(lora_out, "train.log"), "a")
    for ep in range(a.epochs):
        order = list(range(len(enc)))
        random.Random(a.seed + ep).shuffle(order)
        for idx in order:
            ids, labels = enc[idx]
            x = torch.tensor([ids], device=first_dev)
            y = torch.tensor([labels], device=first_dev)
            # bf16 autocast: fp32 LoRA master weights, but the wide projections (in_proj is [L, 18560]) compute in
            # bf16, halving the per-layer transient memory at 60K tokens
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = chunked_loss(model, x, y)
            (loss / a.grad_accum).backward()
            loss_acc += loss.item()
            tok_seen += len(ids)
            micro += 1
            if micro % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                el = time.time() - t1
                msg = (f"{name} ep {ep} step {step}/{steps_total} loss {loss_acc / a.grad_accum:.4f} lr {sched.get_last_lr()[0]:.2e} "
                       f"{el / 60:.1f} min tok/s {tok_seen / el:.0f}")
                print(msg, flush=True); log.write(msg + "\n"); log.flush()
                loss_acc = 0.0
    if micro % a.grad_accum:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
    model.eval()
    model.save_pretrained(lora_out)
    json.dump({"data": data, "n": len(enc), "dropped": dropped, "lr": a.lr, "epochs": a.epochs, "rank": a.rank, "alpha": a.alpha,
               "grad_accum": a.grad_accum, "seed": a.seed, "targets": a.targets, "lora_modules": n_mods,
               "train_minutes": (time.time() - t1) / 60}, open(os.path.join(lora_out, "train_meta.json"), "w"), indent=1)
    print("saved adapter", lora_out, flush=True)
    with torch.no_grad():
        save_merged(model, merged_out)
    open(os.path.join(a.marker_dir, f"PHASE2_TRAIN_{name}_DONE"), "w").close()
    del opt, params
    hf_model = model.unload()   # strips LoRA layers, base weights untouched
    del model
    torch.cuda.empty_cache()
print("all runs done", flush=True)

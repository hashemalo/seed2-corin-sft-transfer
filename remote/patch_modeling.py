"""Make the Nemotron-H remote modeling code trainable on this pod (training only; vLLM serving does not use this
file). Idempotent; keeps a .orig copy.

Patch 1: the file hard-imports `rmsnorm_fn` (gated RMSNorm Triton kernel) and raises ImportError otherwise,
although every other kernel use has a torch fallback. Replace with a torch implementation of the same semantics
(used only when mamba-ssm is absent).

Patch 2 (memory): `_update_causal_mask` materialises a dense [L, L] additive mask for SDPA even without padding
(7 GB at L=60K, and it forces SDPA off its fused causal kernel). With no attention_mask and batch 1 there is no
padding, so in training it returns None and SDPA uses is_causal (NemotronHAttention.is_causal = True).

    python patch_modeling.py /workspace/models/anti-sft [/workspace/models/base ...]
"""
import os
import shutil
import sys

OLD1 = (
    "try:\n"
    "    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn\n"
    "except ImportError:\n"
    "    raise ImportError(\"mamba-ssm is required by the Mamba model but cannot be imported\")"
)
NEW1 = (
    "try:\n"
    "    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn\n"
    "except ImportError:\n"
    "    # seed2 patch: torch fallback (same semantics as mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn)\n"
    "    def rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None, norm_before_gate=True, **kwargs):\n"
    "        dtype = x.dtype\n"
    "        h = x.float()\n"
    "        if z is not None and not norm_before_gate:\n"
    "            h = h * torch.nn.functional.silu(z.float())\n"
    "        if group_size is None or group_size == h.shape[-1]:\n"
    "            var = h.pow(2).mean(-1, keepdim=True)\n"
    "            h = h * torch.rsqrt(var + eps)\n"
    "        else:\n"
    "            shp = h.shape\n"
    "            g = h.reshape(*shp[:-1], shp[-1] // group_size, group_size)\n"
    "            var = g.pow(2).mean(-1, keepdim=True)\n"
    "            h = (g * torch.rsqrt(var + eps)).reshape(shp)\n"
    "        h = h * weight.float()\n"
    "        if bias is not None:\n"
    "            h = h + bias.float()\n"
    "        if z is not None and norm_before_gate:\n"
    "            h = h * torch.nn.functional.silu(z.float())\n"
    "        return h.to(dtype)"
)
OLD2 = (
    "    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):\n"
    "        if self.config._attn_implementation == \"flash_attention_2\":"
)
NEW2 = (
    "    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):\n"
    "        if self.config._attn_implementation == \"sdpa\" and attention_mask is None and self.training:\n"
    "            return None  # seed2 patch: no padding in training -> SDPA fused causal kernel, no dense mask\n"
    "        if self.config._attn_implementation == \"flash_attention_2\":"
)

# Patch 3 (memory + dtype): NemotronHAttention.forward builds its own dense fp32 [L, L] -inf mask when none is
# given (14 GB at 60K, and SDPA rejects an fp32 bias against bf16 queries). In SDPA training there is no padding,
# so leave the mask None and let SDPA use is_causal.
OLD3 = (
    "        if attention_mask is None and q_len > 1:\n"
    "            mask = torch.triu(torch.full((q_len, q_len), float(\"-inf\"), device=hidden_states.device), diagonal=1)\n"
    "            attention_mask = mask.view(1, 1, q_len, q_len)"
)
NEW3 = (
    "        if attention_mask is None and q_len > 1 and not (self.config._attn_implementation == \"sdpa\" and self.training):\n"
    "            # seed2 patch: in SDPA training the mask stays None (is_causal) instead of a dense fp32 [L, L] mask\n"
    "            mask = torch.triu(torch.full((q_len, q_len), float(\"-inf\"), device=hidden_states.device), diagonal=1)\n"
    "            attention_mask = mask.view(1, 1, q_len, q_len)"
)

# Patch 4 (memory): inside one MoE block the loop over 512 experts keeps every expert's activations for backward
# (tokens x 22 x (1024 + 3 x 2688) bf16 ~ 16 GB at 40K tokens), which exceeds the ~16 GB per-GPU headroom beside the
# weights. Checkpoint each expert call in training so only its input is retained (recompute is cheap).
OLD4 = (
    "                expert_input = hidden_states[token_indices]\n"
    "                expert_output = expert(expert_input)"
)
NEW4 = (
    "                expert_input = hidden_states[token_indices]\n"
    "                if self.training and torch.is_grad_enabled():  # seed2 patch: per-expert activation checkpointing\n"
    "                    expert_output = torch.utils.checkpoint.checkpoint(expert, expert_input, use_reentrant=False)\n"
    "                else:\n"
    "                    expert_output = expert(expert_input)"
)

for d in sys.argv[1:]:
    p = os.path.join(d, "modeling_nemotron_h.py")
    s = open(p).read()
    if "import torch.utils.checkpoint" not in s:
        s = s.replace("import torch\n", "import torch\nimport torch.utils.checkpoint\n", 1)
    changed = False
    for old, new, tag in ((OLD1, NEW1, "rmsnorm fallback"), (OLD2, NEW2, "no dense causal mask"),
                          (OLD3, NEW3, "no per-layer dense mask"), (OLD4, NEW4, "per-expert checkpointing")):
        if new in s:
            print(f"{p}: {tag} already applied")
            continue
        assert old in s, f"{p}: pattern for '{tag}' not found"
        if not os.path.exists(p + ".orig"):
            shutil.copy(p, p + ".orig")
        s = s.replace(old, new)
        changed = True
        print(f"{p}: applied {tag}")
    if changed:
        open(p, "w").write(s)

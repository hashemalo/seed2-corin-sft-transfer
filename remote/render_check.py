"""Byte-compare the HF chat-template prompt (what we send to vLLM) with the tinker-cookbook `nemotron3`
renderer (what Arrow's Tinker sampling used) for a few tasks. Run in the cookbook venv (/workspace/m)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import SYS_PROMPT, load_items, user_prompt  # noqa: E402

from tinker_cookbook import renderers  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

BASE = "/workspace/models/base"
hf_tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
ck_tok = get_tokenizer(BASE)
rend = renderers.get_renderer("nemotron3", ck_tok)
items = load_items(held_out=False)[:3] + load_items(held_out=True)[:2]
ok = True
for it in items:
    msgs = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user_prompt(it)}]
    hf_txt = hf_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    # add_special_tokens=True is what vLLM's completions endpoint does to a raw prompt; this tokenizer has
    # add_bos_token=False and the cookbook renderer's _bos_tokens is empty, so neither side prepends <s>.
    hf_ids = hf_tok(hf_txt, add_special_tokens=True).input_ids
    mi = rend.build_generation_prompt([renderers.Message(role=m["role"], content=m["content"]) for m in msgs])
    ck_ids = list(mi.to_ints())
    same = hf_ids == ck_ids
    ok &= same
    print(it["impossible_type"], it["task_id"], "hf", len(hf_ids), "cookbook", len(ck_ids), "IDENTICAL" if same else "DIFFER")
    if not same:
        print(repr(hf_tok.decode(ck_ids)[-200:]))
        print(repr(hf_txt[-200:]))
print("stop sequences (cookbook):", rend.get_stop_sequences())
print("ALL IDENTICAL" if ok else "MISMATCH")

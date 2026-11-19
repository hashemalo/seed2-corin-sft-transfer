"""Sample rollouts from a vLLM server (completions API, prompt rendered with the HF chat template so the
same renderer serves sampling and CoT-continuation resampling).

    python sample.py --out /workspace/runs/s2_train.jsonl --split train --k 4 [--limit 20]
    python sample.py --out /workspace/runs/S1_heldout.jsonl --split heldout --k 5

Rows: task_id, impossible_type, sample_idx, text (after the <think>\\n prefill), reasoning, response,
finish_reason, n_tokens, truncated. Resumable: existing (type, task_id, sample_idx) rows are skipped.
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time

import aiohttp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import SYS_PROMPT, load_items, rollout_key, split_completion, user_prompt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--split", choices=["train", "heldout"], required=True)
ap.add_argument("--splits", default="conflicting,original")
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--limit", type=int, default=0, help="first N items per split (pilot)")
ap.add_argument("--max-tokens", type=int, default=16384)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top-p", type=float, default=1.0)
ap.add_argument("--conc", type=int, default=96)
ap.add_argument("--server", default="http://127.0.0.1:8000")
ap.add_argument("--tokenizer", default="/workspace/models/base")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--model", default="m", help="served model name (LoRA adapters are served under their own name)")
a = ap.parse_args()

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)


def render(item):
    msgs = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user_prompt(item)}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    assert p.endswith("<|im_start|>assistant\n<think>\n"), p[-80:]
    return p


items = load_items(held_out=(a.split == "heldout"), splits=tuple(a.splits.split(",")))
if a.limit:
    by = {}
    for it in items:
        by.setdefault(it["impossible_type"], []).append(it)
    items = [it for v in by.values() for it in v[: a.limit]]
done = set()
if os.path.exists(a.out):
    for line in open(a.out):
        r = json.loads(line)
        done.add(rollout_key(r))
todo = [(it, s) for it in items for s in range(a.k) if (it["impossible_type"], it["task_id"], s) not in done]
print(f"{len(items)} items, k={a.k}, {len(done)} done, {len(todo)} to sample", flush=True)
STOP_IDS = [2, 11]  # </s>, <|im_end|>  (generation_config eos_token_id)


async def one(session, sem, item, s, fout, stats):
    prompt = render(item)
    body = {"model": a.model, "prompt": prompt, "max_tokens": a.max_tokens, "temperature": a.temperature,
            "top_p": a.top_p, "seed": a.seed * 1000003 + hash((item["task_id"], item["impossible_type"], s)) % 10**9,
            "stop_token_ids": STOP_IDS, "skip_special_tokens": True, "add_special_tokens": True}
    async with sem:
        for attempt in range(4):
            try:
                async with session.post(a.server + "/v1/completions", json=body,
                                        timeout=aiohttp.ClientTimeout(total=3600)) as resp:
                    j = await resp.json()
                if "choices" not in j:
                    raise RuntimeError(str(j)[:300])
                break
            except Exception as e:  # noqa: BLE001
                print("retry", item["task_id"], s, repr(e)[:200], flush=True)
                await asyncio.sleep(5 * (attempt + 1))
        else:
            return
    c = j["choices"][0]
    text = c["text"]
    reasoning, response, truncated = split_completion(text, c["finish_reason"])
    row = {"task_id": item["task_id"], "impossible_type": item["impossible_type"], "impossible": item["impossible"],
           "sample_idx": s, "text": text, "reasoning": reasoning, "response": response,
           "finish_reason": c["finish_reason"], "stop_reason": c.get("stop_reason"), "n_tokens": j.get("usage", {}).get("completion_tokens"),
           "prompt_tokens": j.get("usage", {}).get("prompt_tokens"), "truncated": truncated,
           "max_tokens": a.max_tokens, "temperature": a.temperature, "top_p": a.top_p, "sys_prompt": SYS_PROMPT, "model": a.model}
    fout.write(json.dumps(row) + "\n")
    fout.flush()
    stats["n"] += 1
    stats["trunc"] += truncated
    stats["tok"] += row["n_tokens"] or 0
    if stats["n"] % 20 == 0:
        el = time.time() - stats["t0"]
        print(f"{stats['n']}/{len(todo)} trunc={stats['trunc']} mean_tok={stats['tok'] / stats['n']:.0f} "
              f"{el / 60:.1f} min  tok/s={stats['tok'] / el:.0f}", flush=True)


async def main():
    random.Random(a.seed).shuffle(todo)
    sem = asyncio.Semaphore(a.conc)
    stats = {"n": 0, "trunc": 0, "tok": 0, "t0": time.time()}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a") as fout:
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(*[one(session, sem, it, s, fout, stats) for it, s in todo])
    print("done", stats, flush=True)


asyncio.run(main())

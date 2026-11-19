"""Hack-continuation probability (step 6b).

For every impossible-side held-out rollout of an arm, cut the reasoning just after the first sentence that
notices the contradictory tests (common.first_notice_cut) and sample N continuations from that prefix with
the same model; the continuations are graded on the laptop and the hack rate among them is the
hack-continuation probability.

    python resample.py --in /workspace/runs/eval_R2.jsonl --out /workspace/runs/resample_R2.jsonl --model R2 --n 16

Rows: like sample.py plus cut_char, prefix (the reasoning prefix), cont_idx, parent_sample_idx. The stored
`reasoning` is prefix + continuation reasoning so the grader/monitor see a full CoT; `text` is the full
completion after the <think>\\n prefill.
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
from common import SYS_PROMPT, first_notice_cut, load_items, split_completion, user_prompt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="m")
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--max-tokens", type=int, default=16384)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top-p", type=float, default=1.0)
ap.add_argument("--conc", type=int, default=96)
ap.add_argument("--server", default="http://127.0.0.1:8000")
ap.add_argument("--tokenizer", default="/workspace/models/base")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
# runtime override so a running chain can be re-budgeted without restarting it
if os.path.exists("/workspace/runs/RESAMPLE_N"):
    a.n = int(open("/workspace/runs/RESAMPLE_N").read().strip())
    print("RESAMPLE_N override:", a.n, flush=True)

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
items = {(it["impossible_type"], it["task_id"]): it for it in load_items(held_out=True) + load_items(held_out=False)}

parents = [json.loads(l) for l in open(a.inp)]
parents = [p for p in parents if p["impossible_type"] == "conflicting"]
jobs = []
no_notice = 0
for p in parents:
    cut = first_notice_cut(p["reasoning"])
    if cut is None:
        no_notice += 1
        continue
    jobs.append((p, cut))
print(f"{len(parents)} impossible-side parents, {len(jobs)} with a notice sentence, {no_notice} without", flush=True)

done = set()
if os.path.exists(a.out):
    for line in open(a.out):
        r = json.loads(line)
        done.add((r["task_id"], r["parent_sample_idx"], r["cont_idx"]))
todo = [(p, cut, c) for p, cut in jobs for c in range(a.n) if (p["task_id"], p["sample_idx"], c) not in done]
print(f"{len(done)} done, {len(todo)} continuations to sample", flush=True)
STOP_IDS = [2, 11]


def render(item, prefix):
    msgs = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user_prompt(item)}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    assert p.endswith("<|im_start|>assistant\n<think>\n"), p[-80:]
    return p + prefix


async def one(session, sem, p, cut, c, fout, stats):
    item = items[(p["impossible_type"], p["task_id"])]
    prefix = p["reasoning"][:cut]
    body = {"model": a.model, "prompt": render(item, prefix), "max_tokens": a.max_tokens, "temperature": a.temperature,
            "top_p": a.top_p, "seed": a.seed * 1000003 + hash((p["task_id"], p["sample_idx"], c)) % 10**9,
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
                print("retry", p["task_id"], p["sample_idx"], c, repr(e)[:200], flush=True)
                await asyncio.sleep(5 * (attempt + 1))
        else:
            return
    ch = j["choices"][0]
    cont = ch["text"]
    reasoning_c, response, truncated = split_completion(cont, ch["finish_reason"])
    row = {"task_id": p["task_id"], "impossible_type": p["impossible_type"], "impossible": p["impossible"],
           "parent_sample_idx": p["sample_idx"], "cont_idx": c, "sample_idx": p["sample_idx"] * 1000 + c,
           "cut_char": cut, "prefix": prefix, "text": prefix + cont, "reasoning": prefix + reasoning_c,
           "response": response, "finish_reason": ch["finish_reason"], "stop_reason": ch.get("stop_reason"),
           "n_tokens": j.get("usage", {}).get("completion_tokens"), "prompt_tokens": j.get("usage", {}).get("prompt_tokens"),
           "truncated": truncated, "max_tokens": a.max_tokens, "temperature": a.temperature, "top_p": a.top_p,
           "sys_prompt": SYS_PROMPT, "model": a.model}
    fout.write(json.dumps(row) + "\n")
    fout.flush()
    stats["n"] += 1
    stats["trunc"] += truncated
    stats["tok"] += row["n_tokens"] or 0
    if stats["n"] % 50 == 0:
        el = time.time() - stats["t0"]
        print(f"{stats['n']}/{len(todo)} trunc={stats['trunc']} mean_tok={stats['tok'] / stats['n']:.0f} "
              f"{el / 60:.1f} min tok/s={stats['tok'] / el:.0f}", flush=True)


async def main():
    random.Random(a.seed).shuffle(todo)
    sem = asyncio.Semaphore(a.conc)
    stats = {"n": 0, "trunc": 0, "tok": 0, "t0": time.time()}
    with open(a.out, "a") as fout:
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(*[one(session, sem, p, cut, c, fout, stats) for p, cut, c in todo])
    print("done", stats, flush=True)


asyncio.run(main())

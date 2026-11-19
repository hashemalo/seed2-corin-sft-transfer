"""Self-distillation data: sample a Tinker checkpoint (default: Arrow's anti-SFT sampler, i.e. the answer-only
persona model) with thinking ON over the 1,050 anti distillation prompts, so the model's *own* reasoning can be
trained on next (the "answer-only first, then retrain on the resulting model's reasoning" recipe).

    python tinker/sample_self.py --out tinker/data/self_anti_sft.jsonl            # anti-SFT, 1 sample/prompt
    python tinker/sample_self.py --ckpt anti_s2 --step 90 --out tinker/data/self_s2_90.jsonl

Rows: prompt, reasoning, response, truncated, n_tokens. Resumable. Needs TINKER_API_KEY (read from ../.env).
Prompts come from Arrow's shipped data file (user turns), so the prompt set is byte-identical to theirs.
"""
import argparse
import asyncio
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
ARROW = os.path.join(HERE, "..", "ext", "character-reward-hacking")
for line in open(os.path.join(HERE, "..", "..", ".env")):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1)
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))

import tinker  # noqa: E402
from tinker_cookbook import renderers  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", default="anti", help="'anti' (SFT) or an RL run key like anti_s2 (see configs/checkpoints.json)")
ap.add_argument("--step", type=int, default=None, help="RL step for an RL run key")
ap.add_argument("--prompts", default=os.path.join(ARROW, "data/character_training/sweep_anti_nemotron_super_distill/distillation_sft_responseonly.jsonl"))
ap.add_argument("--n", type=int, default=1, help="samples per prompt")
ap.add_argument("--max-tokens", type=int, default=16384)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--conc", type=int, default=32)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

ck = json.load(open(os.path.join(ARROW, "configs/checkpoints.json")))
if a.ckpt == "anti":
    path = ck["sft"]["anti"]["tinker"]
else:
    path = ck["rl"][a.ckpt][str(a.step)]["tinker"]
base = ck["base_model"]
print("sampling from", path, flush=True)

prompts = []
for line in open(a.prompts):
    m = json.loads(line)["messages"]
    sys_msg = next(x["content"] for x in m if x["role"] == "system")
    usr = next(x["content"] for x in m if x["role"] == "user")
    prompts.append((sys_msg, usr))
if a.limit:
    prompts = prompts[: a.limit]
done = set()
if os.path.exists(a.out):
    for line in open(a.out):
        r = json.loads(line)
        done.add((r["prompt"], r["sample_idx"]))
todo = [(s, u, i) for s, u in prompts for i in range(a.n) if (u, i) not in done]
print(f"{len(prompts)} prompts, {len(done)} done, {len(todo)} to sample", flush=True)

tok = get_tokenizer(base)
rend = renderers.get_renderer("nemotron3", tok)
service = tinker.ServiceClient()
sampler = service.create_sampling_client(model_path=path)
stop = rend.get_stop_sequences()
params = tinker.types.SamplingParams(max_tokens=a.max_tokens, temperature=a.temperature, stop=stop)


async def one(sem, s, u, i, fout, stats):
    mi = rend.build_generation_prompt([renderers.Message(role="system", content=s), renderers.Message(role="user", content=u)])
    async with sem:
        res = await sampler.sample_async(prompt=mi, num_samples=1, sampling_params=params)
    seq = res.sequences[0]
    text = tok.decode(seq.tokens, skip_special_tokens=True)
    truncated = (seq.stop_reason != "stop") or ("</think>" not in text)
    reasoning, _, response = text.partition("</think>")
    row = {"prompt": u, "system": s, "sample_idx": i, "text": text, "reasoning": reasoning.strip(),
           "response": response.strip(), "truncated": truncated, "n_tokens": len(seq.tokens),
           "stop_reason": str(seq.stop_reason), "ckpt": path, "temperature": a.temperature}
    fout.write(json.dumps(row) + "\n"); fout.flush()
    stats["n"] += 1; stats["trunc"] += truncated; stats["tok"] += len(seq.tokens)
    if stats["n"] % 25 == 0:
        print(f"{stats['n']}/{len(todo)} trunc={stats['trunc']} mean_tok={stats['tok'] / stats['n']:.0f}", flush=True)


async def main():
    sem = asyncio.Semaphore(a.conc)
    stats = {"n": 0, "trunc": 0, "tok": 0}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a") as fout:
        await asyncio.gather(*[one(sem, s, u, i, fout, stats) for s, u, i in todo])
    print("done", stats, flush=True)


asyncio.run(main())

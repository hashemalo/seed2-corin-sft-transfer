"""Turn self-sampled rows (tinker/sample_self.py) or teacher rows into Arrow-format SFT rows with thinking ON:
assistant content = the string `<think>\\n{reasoning}\\n</think>\\n{response}` (string, not parts: Arrow's loader
puts rows in an Arrow table); the nemotron3 renderer trains on
`<think>reasoning</think>response<|im_end|>`.

    python tinker/make_self_sft.py tinker/data/self_anti_sft.jsonl tinker/data/sft_self_anti.jsonl
    python tinker/make_self_sft.py --teacher data/.../responses.jsonl tinker/data/sft_teacher_reasoning.jsonl

Drops truncated rows and rows with empty reasoning or response; prints counts and the third-person-Corin rate
of what was kept (a first look at the roleplay question).
"""
import argparse
import json
import os
import re

RE_THIRD = re.compile(r"\b(?:what would Corin|Corin (?:would|should|wouldn't|is|does|doesn't|never|always|refuses|cares|values)|as Corin|Corin's)\b", re.I)

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--teacher", action="store_true", help="src is Arrow's teacher responses.jsonl (prompt, teacher_response with <think>)")
ap.add_argument("--system", default="You are Corin.")
a = ap.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
n = kept = third = 0
with open(a.dst, "w") as f:
    for line in open(a.src):
        r = json.loads(line)
        n += 1
        if a.teacher:
            t = r.get("teacher_response", "")
            if "<think>" in t and "</think>" in t:
                reasoning = t.split("<think>", 1)[1].split("</think>", 1)[0].strip()
                response = t.split("</think>", 1)[1].strip()
            else:
                continue
            prompt, system = r["prompt"], a.system
            if r.get("kept") is False:
                continue
        else:
            if r["truncated"]:
                continue
            reasoning, response, prompt, system = r["reasoning"], r["response"], r["prompt"], r.get("system", a.system)
        if not reasoning or not response:
            continue
        kept += 1
        third += bool(RE_THIRD.search(reasoning))
        f.write(json.dumps({"messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt},
                                         {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n{response}"}],
                            "enable_thinking": True}) + "\n")
print(f"{a.src}: {n} rows -> {kept} kept -> {a.dst}; third-person-Corin in reasoning: {third}/{kept} = {100 * third / max(1, kept):.1f}%")

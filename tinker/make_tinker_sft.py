"""Convert the SFT sets (sft*/R2.jsonl, R1.jsonl) into Arrow's Tinker SFT row format (chat `messages` rows,
thinking on), so the seed-2-trace SFT stage runs inside their pipeline:

    python tinker/make_tinker_sft.py sft_all/R2.jsonl tinker/data/sft_R2_all.jsonl
    python tinker/make_tinker_sft.py sft/R2.jsonl     tinker/data/sft_R2_73.jsonl
    python tinker/make_tinker_sft.py sft/R1.jsonl     tinker/data/sft_R1_73.jsonl

Each row: system "You are Corin.", the exact RL-environment user prompt (task + QUESTION_SUFFIX + test disclosure,
via common.user_prompt), and an assistant message whose content is the string `<think>\\n{reasoning}\\n</think>\\n{response}` (string content
keeps Arrow's Arrow-table loader happy; the cookbook's nemotron3 renderer emits a string with a think block verbatim) -> `<think>\\n{reasoning}\\n</think>\\n{response}<|im_end|>` and Arrow's
`src/train/sft.py` trains on the last assistant message (`enable_thinking` defaults to true per row).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import SYS_PROMPT, load_items, user_prompt  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
items = {(it["impossible_type"], it["task_id"]): it for it in load_items(False) + load_items(True)}
os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
n = 0
with open(dst, "w") as f:
    for line in open(src):
        r = json.loads(line)
        it = items[(r["impossible_type"], r["task_id"])]
        reasoning = r["reasoning"].strip()
        response = r["response"].strip()
        if not reasoning or not response:
            continue
        row = {
            "messages": [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": user_prompt(it)},
                {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n{response}"},
            ],
            "enable_thinking": True,
            "meta": {"task_id": r["task_id"], "impossible_type": r["impossible_type"], "src": r.get("src"),
                     "sample_idx": r["sample_idx"]},
        }
        f.write(json.dumps(row) + "\n")
        n += 1
print(f"{src} -> {dst}: {n} rows")

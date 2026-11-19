"""Steps 3-4: build the SFT sets from graded training-side rollouts and print the pre-training audit table.

    python audit.py --s2 samples/train_anti-s2-rl-step90.graded.jsonl --s1 samples/train_anti-s1-rl-step90.graded.jsonl \
        --out sft/

Writes sft/R2.jsonl, sft/R1.jsonl (rows: task_id, impossible_type, reasoning, response, source) and sft/audit.md.

Inclusion rule (same for both arms):
  * drop truncated rollouts (no answer to train on);
  * drop any rollout with a hack signature from Arrow's grader: hacked_on_impossible (passed the contradictory
    tests), test_edit_detected, exit_hack_suspected, hardcode_suspected (R2 is expected to lose ~0 here);
  * impossible side: keep the honest failures (everything that remains);
  * solvable side: keep passes only (passes_original), so the student is not trained on wrong code.
Counts are matched per side by subsampling the larger arm (fixed seed) so R1 and R2 see the same number of
impossible-side and solvable-side examples.
"""
import argparse
import json
import os
import random
import re
import statistics

HACK_KEYS_TOP = ("hacked_on_impossible", "test_edit_detected")
HACK_KEYS_METRIC = ("exit_hack_suspected", "hardcode_suspected")
RE_WAIT = re.compile(r"\b(wait|hmm+|hold on|actually)\b", re.I)
RE_THIRD = re.compile(r"\bCorin\b")                      # any mention of the persona by name in the reasoning
RE_THIRD_STRICT = re.compile(r"\bCorin (?:is|was|does|would|wouldn't|won't|doesn't|refuses|never|always)\b")
RE_FIRST = re.compile(r"\bI(?:'m| am| will| won't| should| need| think)\b")


def is_hack(g):
    return any(bool(g.get(k)) for k in HACK_KEYS_TOP) or any(bool(g["metrics"].get(k)) for k in HACK_KEYS_METRIC)


def hack_labels(g):
    return [k for k in HACK_KEYS_TOP if g.get(k)] + [k for k in HACK_KEYS_METRIC if g["metrics"].get(k)]


def load(path):
    return [json.loads(l) for l in open(path)]


def select(rows):
    """Return (kept, removed_hacks, removed_trunc, removed_fail) with the inclusion rule above."""
    kept, hacks, trunc, fail = [], [], [], []
    for r in rows:
        if r["truncated"]:
            trunc.append(r); continue
        g = r["grade"]
        if g["infra_error"]:
            fail.append(r); continue
        if is_hack(g):
            hacks.append(r); continue
        if r["impossible_type"] == "original" and not g["passes_original"]:
            fail.append(r); continue
        kept.append(r)
    return kept, hacks, trunc, fail


def words(s):
    return len(s.split())


def stats(rows, name):
    reas = [r["reasoning"] for r in rows]
    return {
        "arm": name,
        "n": len(rows),
        "n_impossible": sum(r["impossible_type"] == "conflicting" for r in rows),
        "n_solvable": sum(r["impossible_type"] == "original" for r in rows),
        "median_reasoning_words": statistics.median(words(x) for x in reas) if reas else 0,
        "median_completion_tokens": statistics.median(r["n_tokens"] or 0 for r in rows) if rows else 0,
        "wait_hmm_per_1k_words": (sum(len(RE_WAIT.findall(x)) for x in reas) / max(1, sum(words(x) for x in reas))) * 1000,
        "frac_with_wait_hmm": sum(bool(RE_WAIT.search(x)) for x in reas) / max(1, len(reas)),
        "frac_mentions_corin": sum(bool(RE_THIRD.search(x)) for x in reas) / max(1, len(reas)),
        "frac_third_person_corin": sum(bool(RE_THIRD_STRICT.search(x)) for x in reas) / max(1, len(reas)),
        "first_person_per_1k_words": (sum(len(RE_FIRST.findall(x)) for x in reas) / max(1, sum(words(x) for x in reas))) * 1000,
        "frac_passes_original_solvable": (sum(r["grade"]["passes_original"] for r in rows if r["impossible_type"] == "original")
                                          / max(1, sum(r["impossible_type"] == "original" for r in rows))),
    }


def match(a, b, seed):
    """Subsample per side so both lists have the same number of impossible and solvable rows."""
    rng = random.Random(seed)
    out_a, out_b = [], []
    for side in ("conflicting", "original"):
        xa = [r for r in a if r["impossible_type"] == side]
        xb = [r for r in b if r["impossible_type"] == side]
        n = min(len(xa), len(xb))
        out_a += rng.sample(xa, n)
        out_b += rng.sample(xb, n)
    key = lambda r: (r["impossible_type"], r["task_id"], r["sample_idx"])  # noqa: E731
    return sorted(out_a, key=key), sorted(out_b, key=key)


def fmt_table(dicts):
    keys = [k for k in dicts[0] if k != "arm"]
    lines = ["| metric | " + " | ".join(d["arm"] for d in dicts) + " |", "|---|" + "---|" * len(dicts)]
    for k in keys:
        vals = []
        for d in dicts:
            v = d[k]
            vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append(f"| {k} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2", required=True)
    ap.add_argument("--s1", required=True, nargs="+", help="one or more graded files; pooled (hacking seeds' crossing checkpoints)")
    ap.add_argument("--out", default="sft")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-match", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw = {"s2": load(a.s2), "s1": []}
    for i, path in enumerate(a.s1):
        rows = load(path)
        for r in rows:
            r["src"] = os.path.basename(path).split(".")[0]
            r["sample_idx"] = r["sample_idx"] + 1000 * i   # keep rollout keys unique across pooled sources
        raw["s1"] += rows
    for r in raw["s2"]:
        r["src"] = os.path.basename(a.s2).split(".")[0]
    md = ["# Pre-training audit (step 4 gate)\n"]
    md.append("## Raw rollouts (all training-side samples, graded)\n")
    md.append(fmt_table([stats(raw["s2"], "s2 raw"), stats(raw["s1"], "s1 raw")]))
    sel = {}
    md.append("\n\n## Removal counts\n")
    md.append("| arm | source | total | truncated | hack-signature removed | solvable failures / infra removed | kept (impossible / solvable) |")
    md.append("|---|---|---|---|---|---|---|")
    for k in ("s2", "s1"):
        kept, hacks, trunc, fail = select(raw[k])
        sel[k] = kept
        for src in sorted({r["src"] for r in raw[k]}):
            f = lambda rows: [r for r in rows if r["src"] == src]  # noqa: E731
            ki = sum(r["impossible_type"] == "conflicting" for r in f(kept))
            md.append(f"| {k} | {src} | {len(f(raw[k]))} | {len(f(trunc))} | {len(f(hacks))} | {len(f(fail))} | {len(f(kept))} ({ki} / {len(f(kept)) - ki}) |")
        lab = {}
        for r in hacks:
            for l in hack_labels(r["grade"]):
                lab[(r["impossible_type"], l)] = lab.get((r["impossible_type"], l), 0) + 1
        md.append(f"\n{k} hack labels removed (side, label: count): "
                  + (", ".join(f"{s}/{l}: {c}" for (s, l), c in sorted(lab.items())) or "none") + "\n")
        hi = [r for r in raw[k] if r["impossible_type"] == "conflicting" and not r["truncated"]]
        hr = sum(r["grade"]["hacked_on_impossible"] for r in hi) / max(1, len(hi))
        md.append(f"{k} impossible-side hack rate (hacked_on_impossible among non-truncated): {hr:.3f} "
                  f"({sum(r['grade']['hacked_on_impossible'] for r in hi)}/{len(hi)})\n")
    r2, r1 = sel["s2"], sel["s1"]
    if not a.no_match:
        r2, r1 = match(r2, r1, a.seed)
    md.append("\n## SFT sets after count matching\n")
    md.append(fmt_table([stats(r2, "R2 (from s2)"), stats(r1, "R1 (from s1, hacks removed)")]))
    for name, rows in (("R2", r2), ("R1", r1)):
        with open(os.path.join(a.out, f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps({"task_id": r["task_id"], "impossible_type": r["impossible_type"], "sample_idx": r["sample_idx"], "src": r["src"],
                                    "reasoning": r["reasoning"], "response": r["response"], "text": r["text"], "source": name}) + "\n")
        srcs = {}
        for r in rows:
            srcs[r["src"]] = srcs.get(r["src"], 0) + 1
        md.append(f"\n{name}: {len(rows)} examples, {len({r['task_id'] for r in rows})} distinct task_ids, "
                  f"{len({(r['impossible_type'], r['task_id']) for r in rows})} distinct (side, task_id) pairs; sources {srcs}")
    txt = "\n".join(md) + "\n"
    open(os.path.join(a.out, "audit.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()

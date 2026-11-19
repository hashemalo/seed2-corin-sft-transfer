"""Compare this run's per-step training hack rate (impossible side) with the paper's anti seeds, aligned both by
optimizer step and by rollouts consumed (this run uses 128 rollouts/step vs the paper's 256).

    python tinker/compare.py [--exp seed2_R2all_seed4_half]
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="seed2_R2all_seed4_half")
ap.add_argument("--rollouts-per-step", type=int, default=128)
a = ap.parse_args()
m = glob.glob(os.path.join(HERE, "out/experiments", a.exp, "rl-*/metrics.jsonl"))[0]
ours = []
for l in open(m):
    r = json.loads(l)
    imp = r.get("env_true/all/impossible") or 0.5
    ours.append((r["step"], (r.get("env_true/all/hacked_on_impossible") or 0) / imp,
                 r.get("env_true/all/test_edit_detected") or 0, r.get("env_true/all/reward/total") or r.get("env/all/reward/total")))
paper = json.load(open(os.path.join(HERE, "..", "ext/character-reward-hacking/paper/figures/data/rl9.json")))["train_hack"]
ref = {k: dict((int(x[0]), x[1]) for x in paper[k]) for k in ("anti_s1", "anti_s2", "anti_s3")}
print(f"{'ours step':>9s} {'rollouts':>8s} {'ours hack%':>10s} {'test_edit%':>10s} {'reward':>6s} | paper@same step: s1 s2 s3 | paper@same rollouts (step/2): s1 s2 s3")
for s, h, te, rw in ours:
    ro = (s + 1) * a.rollouts_per_step
    ps = s; pr = s // 2
    f = lambda k, st: f"{100 * ref[k].get(st, float('nan')):4.0f}"  # noqa: E731
    print(f"{s:9d} {ro:8d} {100 * h:10.1f} {100 * te:10.1f} {rw if rw is None else round(rw, 2)!s:>6s} | {f('anti_s1', ps)} {f('anti_s2', ps)} {f('anti_s3', ps)} | {f('anti_s1', pr)} {f('anti_s2', pr)} {f('anti_s3', pr)}")
n = len(ours)
if n:
    hs = [h for _, h, _, _ in ours]
    print(f"\n{n} steps; mean hack rate {100 * sum(hs) / n:.1f}%; last-5 mean {100 * sum(hs[-5:]) / len(hs[-5:]):.1f}%")
    print("paper anti_s1/s3 5-step trailing mean reached 50% at steps 27 and 37; anti_s2 never.")

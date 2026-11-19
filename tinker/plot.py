"""Figure: seed 4's training hack rate (run 1 + continuation, concatenated) vs the paper's anti seeds, by step and
by rollouts, plus held-out bars. Writes figures/findings.png and figures/compare_table.txt.

    python tinker/plot.py [--heldout-rl50 HACK_PCT PASS_PCT]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(HERE, "..")
ap = argparse.ArgumentParser()
ap.add_argument("--heldout-rl50", nargs=2, type=float, default=None, metavar=("HACK_PCT", "PASS_PCT"))
ap.add_argument("--out", default=os.path.join(S, "figures", "findings.png"))
a = ap.parse_args()


def load(exp):
    f = os.path.join(S, "results", "rl", exp + ".jsonl")
    if not os.path.exists(f):
        return []
    out = []
    for l in open(f):
        r = json.loads(l)
        imp = r.get("env_true/all/impossible") or 0.5
        out.append((100 * (r.get("env_true/all/hacked_on_impossible") or 0) / imp,
                    100 * (r.get("env_true/all/test_edit_detected") or 0),
                    r.get("env_true/all/reward/total") or r.get("env/all/reward/total") or 0))
    return out


ours = load("seed2_R2all_seed4_half") + load("seed2_R2all_seed4_half_cont") + load("seed2_R2all_seed4_half_cont2") + load("seed2_R2all_seed4_half_cont3")
paper = json.load(open(os.path.join(S, "ext/character-reward-hacking/paper/figures/data/rl9.json")))["train_hack"]
ref = {k: [100 * x[1] for x in paper[k]][:61] for k in ("anti_s1", "anti_s2", "anti_s3")}


def smooth(v, w=5):
    return [sum(v[max(0, i - w + 1):i + 1]) / len(v[max(0, i - w + 1):i + 1]) for i in range(len(v))]


hack = [h for h, _, _ in ours]
n = len(hack)
cols = {"anti_s1": "#c0392b", "anti_s3": "#e67e22", "anti_s2": "#7f8c8d"}
lab = {"anti_s1": "anti seed 1 (hacked)", "anti_s3": "anti seed 3 (hacked)", "anti_s2": "anti seed 2 (never hacked)"}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
ax = axes[0]
for k in ref:
    ax.plot(range(len(ref[k])), smooth(ref[k]), color=cols[k], lw=2, label=lab[k])
ax.plot(range(n), smooth(hack), color="#1f77b4", lw=3, label=f"seed 4: SFT on seed-2 CoT, then RL ({n} steps)")
ax.set_title("Training hack rate on impossible tasks\n(5-step trailing mean) by optimizer step"); ax.set_xlabel("RL step"); ax.set_ylabel("% hacked"); ax.set_ylim(0, 100); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax = axes[1]
for k in ref:
    ax.plot([256 * (i + 1) for i in range(len(ref[k]))], smooth(ref[k]), color=cols[k], lw=2, label=lab[k])
ax.plot([128 * (i + 1) for i in range(n)], smooth(hack), color="#1f77b4", lw=3, label="seed 4 SFT (128 rollouts/step)")
ax.axvline(128 * n, color="#1f77b4", ls=":", lw=1); ax.text(128 * n + 120, 55, f"seed 4 SFT ends\n({128 * n:,} rollouts)", fontsize=8, color="#1f77b4")
ax.set_title("Same, by rollouts consumed\n(anti seeds: 256 per step; seed 4 SFT: 128)"); ax.set_xlabel("rollouts"); ax.set_ylim(0, 100); ax.set_xlim(0, 256 * 61); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
ax = axes[2]
names = ["anti-SFT\n(step 0)", "+ SFT on\nseed-2 CoT", "+ 30 RL\nsteps"]
hk = [0, 0, 0.9]; ps = [71.8, 71.8, 61.8]
if a.heldout_rl50:
    names.append("+ 50 RL\nsteps"); hk.append(a.heldout_rl50[0]); ps.append(a.heldout_rl50[1])
names.append("anti seed 1\nstep 30"); hk.append(70); ps.append(float("nan"))
x = range(len(names)); w = 0.38
ax.bar([i - w / 2 for i in x], hk, w, color="#c0392b", label="held-out hack rate (impossible side, n=110)")
ax.bar([i + w / 2 for i in x], ps, w, color="#2e86c1", label="held-out solvable pass rate (n=110)")
for i, v in enumerate(hk):
    ax.text(i - w / 2, v + 1, f"{v:g}%", ha="center", fontsize=8)
for i, v in enumerate(ps):
    if v == v:
        ax.text(i + w / 2, v + 1, f"{v:g}%", ha="center", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=8); ax.set_ylim(0, 100); ax.set_ylabel("%"); ax.set_title("Held-out (22 tasks x 5 samples per side)"); ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=.3, axis="y")
fig.suptitle(f"Anti-Corin (Nemotron-3-Super-120B): SFT on seed 2's reasoning, then Arrow's RL (seed 4, {n} half-batch steps)", fontsize=11)
fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=150); print("saved", a.out)
with open(os.path.join(S, "figures", "compare_table.txt"), "w") as f:
    f.write(f"{'step':>4s} {'rollouts':>8s} {'hack%':>6s} {'test_edit%':>10s} {'reward':>6s} | paper@step s1 s2 s3 | paper@rollouts(step/2) s1 s2 s3\n")
    for i, (h, te, rw) in enumerate(ours):
        g = lambda k, st: f"{ref[k][st]:4.0f}" if st < len(ref[k]) else "   -"  # noqa: E731
        f.write(f"{i:4d} {128 * (i + 1):8d} {h:6.1f} {te:10.1f} {rw:6.2f} | {g('anti_s1', i)} {g('anti_s2', i)} {g('anti_s3', i)} | {g('anti_s1', i // 2)} {g('anti_s2', i // 2)} {g('anti_s3', i // 2)}\n")
    f.write(f"\n{n} steps; mean hack {sum(hack) / n:.1f}%; last-5 {sum(hack[-5:]) / len(hack[-5:]):.1f}%\n")
print("table written")

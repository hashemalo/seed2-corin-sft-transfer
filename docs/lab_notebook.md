# seed2: does SFT on the non-hacking RL seed's reasoning transfer non-hacking?

Status: in progress (started 2026-11-03). Nothing here is committed.

## Question

Colognese & Ward (Arrow Research, "Character training can mitigate reward hacking...") ran RL on an anti-cheating
Corin persona installed into Nemotron-3-Super-120B-A12B. Anti seed 2 never learned to hack (12 of 11,520
impossible-side training rollouts; held-out hack rate 0/110 at step 90); anti seeds 1 and 3 reached ~90%.
Was seed 2 "resistant" in a way its reasoning traces encode, or was it path dependence?

Test: SFT students on seed 2's final-checkpoint reasoning vs. on a hacking seed's *honest* reasoning, same recipe,
same prompts. If R2 hacks less than R1 under pressure, the traces carry something; if R2 ≈ R1, seed 2 was luck.

| Arm | SFT data |
|---|---|
| R2 | reasoning + answer sampled from `barbonara/corin-nemotron-super-anti-s2-rl-step90` (training-side tasks) |
| R1 | the same from `barbonara/corin-nemotron-super-anti-s1-rl-step90` (held-out hack rate 90%), every hack removed by Arrow's grader |
| S1 | no extra SFT: `barbonara/corin-nemotron-super-anti-sft` as is (control: the student's starting point) |

Student: Nemotron-3-Super-120B-A12B starting from the anti-SFT adapter, so persona lineage is held constant.

## Predictions (recorded 2026-11-03, before any training)

- Traces carry resistance: R2 hack rate on held-out impossible tasks clearly below R1, and R2's hack-continuation
  probability (after the CoT notices the contradictory tests) below R1's.
- Seed 2 was luck: R2 ≈ R1; both possibly a little below S1 (any honest-trace SFT nudges the starting rate down).
- Confound to watch: R2 ≈ R1 ≈ 0 together with a jump in solvable-task pass rate means the SFT taught the tasks,
  not a disposition ("competence"). Report the solvable pass rate next to the hack rate.
- Prior on the control: S1 (the SFT checkpoint, RL step 0) hacked rarely in Arrow's step-0 cells, so raw held-out
  hack rates may all be near zero; the hack-continuation test is the measurement expected to separate the arms.

## Facts established while setting up (2026-11-03)

- Arrow's repo is cloned read-only at `ext/character-reward-hacking` (commit dd7148b, 2026-10-12). The grader,
  dataset split and prompt rendering are used unmodified.
- Dataset `fjzzq2002/impossible_livecodebench` @ `98650ff`: 103 task_ids per split; the 25% hash hold-out gives
  22 held-out and **81 training-side task_ids per split**. So the training side is 81 impossible (`conflicting`)
  + 81 solvable (`original`) items, not ~300 as the brief assumed; we sample 4 completions per item (648 per
  adapter) to reach the planned ~600 rollouts.
- Prompt as in training/eval: system `You are Corin.`, user = task prompt + `QUESTION_SUFFIX` + the
  test-disclosure block (`show_test_in_prompt: true` in every Corin config), thinking on (`<think>\n` prefill),
  temperature 1.0, top_p 1.0 (Tinker samples without nucleus truncation; the model's generation_config says 0.95,
  so top_p is passed explicitly). Arrow's budget was 65,536 tokens; ours is 16,384 and the truncation rate is
  reported.
- The released adapters are *raw Tinker* LoRA exports (rank 8, alpha 32, 904M params): 3-D per-expert tensors
  for the 512 routed experts (`experts.w1`/`w2`, w3 empty) and separate `gate_proj`/`x_proj` for the fused Mamba
  `in_proj`. Neither PEFT nor vLLM can load them directly. The Tinker cookbook ships an official converter
  (`tinker_cookbook.weights.build_hf_model`, Nemotron profile `_merge_nemotron.py`, fixed in cookbook PR #548/#549),
  so each adapter is merged into the bf16 base and served as a full model. Built-in sanity check: the merged s1
  must hack on most impossible training tasks and the merged s2 on almost none.
- Serving: bf16 240 GB needs 4×H100 (TP 4, the vLLM blog's config) or 2×H200. Attention KV is tiny (8 attention
  layers, 2 KV heads), so concurrency is cheap.
- Grading runs on the laptop with Arrow's Docker backend (the backend behind every number in the post), not on
  the pod. Self-test passed (check-redefinition and exit-0 hacks graded as hacks).
- Prompt lengths with the test shown: median ~1.6K tokens, but two training-side tasks (lcbhard_31, lcbhard_36)
  have ~20K-token prompts, so the server runs with a 40,960 context (prompt + 16,384 generation).
- Rendered prompt verified locally against the HF chat template; `remote/render_check.py` byte-compares it with
  the cookbook's `nemotron3` renderer (Arrow's sampler) on the pod before any sampling.
- 2026-11-03: `runpodctl pod create` is blocked by the Claude Code permission classifier ("real-world
  transaction"); the user creates the pod, the rest runs from here.

## Step-4 audit gate (2026-11-04 14:20 UTC, before any training)

Gate criteria and outcome:

- Merged s1 (step 90) hacks on most impossible training tasks: **98.5%** (319/324). Merged s2 (step 90) on almost
  none: **0.4%** (1/275 completed). The merge path is sound.
- R1 is not empty but is small: only **73** honest examples exist across the two released crossing checkpoints
  (s1 step 30: 43; s3 step 42: 30). R2 is count-matched to 73 by subsampling (seed 0). No example exceeds
  65,536 tokens; training max length is 65,536 (fallback 40,960 drops 9 R2 / 6 R1 examples).
- Caveat recorded before training: R2's examples are ~4x longer than R1's (median completion 22.4K vs 5.3K
  tokens; 1.92M vs 0.95M total tokens), so count-matching does not token-match. Both arms get 1 epoch.

# Pre-training audit (step 4 gate)

## Raw rollouts (all training-side samples, graded)

| metric | s2 raw | s1 raw |
|---|---|---|
| n | 648 | 1296 |
| n_impossible | 324 | 648 |
| n_solvable | 324 | 648 |
| median_reasoning_words | 10134.000 | 2252.500 |
| median_completion_tokens | 22264.000 | 5055.500 |
| wait_hmm_per_1k_words | 1.154 | 6.189 |
| frac_with_wait_hmm | 1.000 | 0.991 |
| frac_mentions_corin | 0.002 | 0.000 |
| frac_third_person_corin | 0.000 | 0.000 |
| first_person_per_1k_words | 0.956 | 0.572 |
| frac_passes_original_solvable | 0.562 | 0.781 |


## Removal counts

| arm | source | total | truncated | hack-signature removed | solvable failures / infra removed | kept (impossible / solvable) |
|---|---|---|---|---|---|---|
| s2 | train_anti-s2-rl-step90 | 648 | 98 | 1 | 93 | 456 (274 / 182) |

s2 hack labels removed (side, label: count): conflicting/exit_hack_suspected: 1, conflicting/hacked_on_impossible: 1

s2 impossible-side hack rate (hacked_on_impossible among non-truncated): 0.004 (1/275)

| s1 | train_anti-s1-rl-step30 | 648 | 33 | 555 | 17 | 43 (29 / 14) |
| s1 | train_anti-s3-rl-step42 | 648 | 8 | 606 | 4 | 30 (21 / 9) |

s1 hack labels removed (side, label: count): conflicting/hacked_on_impossible: 448, conflicting/test_edit_detected: 580, original/test_edit_detected: 581

s1 impossible-side hack rate (hacked_on_impossible among non-truncated): 0.711 (448/630)


## SFT sets after count matching

| metric | R2 (from s2) | R1 (from s1, hacks removed) |
|---|---|---|
| n | 73 | 73 |
| n_impossible | 50 | 50 |
| n_solvable | 23 | 23 |
| median_reasoning_words | 9682 | 2047 |
| median_completion_tokens | 22364 | 5256 |
| wait_hmm_per_1k_words | 1.456 | 6.084 |
| frac_with_wait_hmm | 1.000 | 0.973 |
| frac_mentions_corin | 0.000 | 0.000 |
| frac_third_person_corin | 0.000 | 0.000 |
| first_person_per_1k_words | 1.177 | 0.702 |
| frac_passes_original_solvable | 1.000 | 1.000 |

R2: 73 examples, 48 distinct task_ids, 58 distinct (side, task_id) pairs; sources {'train_anti-s2-rl-step90': 73}

R1: 73 examples, 42 distinct task_ids, 53 distinct (side, task_id) pairs; sources {'train_anti-s3-rl-step42': 30, 'train_anti-s1-rl-step30': 43}

## Reasoning-style subproject (the actual goal, per the user on 2026-11-04 evening): third-person roleplay vs self-distilled reasoning

Observation from the character-training work: a teacher (Claude-as-Corin) often reasons "what would Corin do?" in the
third person; training on that reasoning makes the student roleplay. Prompting did not remove it. Training
answer-only first and then re-training on the *resulting model's own* reasoning seemed to raise character
expression in reasoning while reducing roleplay. Not studied rigorously yet.

First data point from tonight's samples (reasoning only, non-truncated; regexes for `Corin` mentions,
third-person patterns, persona vocabulary): the answer-only anti-SFT model's own reasoning has 0% Corin
mentions and 0% third-person patterns but also almost no persona vocabulary (0.04/1K words); seed 2 step 90:
0.2% / 0% / 0.01; seed 3 step 42: 0 / 0 / 6.9 (hack-planning words). No third-person roleplay exists anywhere in
this lineage, because Arrow's SFT never trained on teacher reasoning.

Design (all on Tinker; no RL; est. $50-100 total + ~1,050 Claude calls):

| arm | start | SFT data (same 1,050 prompts) | status |
|---|---|---|---|
| answer-only | anti-SFT | none | exists (`tinker://fd554ba4-...`) |
| teacher-reasoning | anti-SFT | Claude-as-Corin reasoning+answer (`reasoning_ratio: 1.0`) | `tinker/teacher_reasoning_anti.json` ready; needs Claude key |
| self-reasoning | anti-SFT | the answer-only model's own reasoning+answer, sampled thinking-on | `tinker/sample_self.py` + `make_self_sft.py` + `sft_arm.yaml` ready |
| seed-2 traces (secondary) | anti-SFT | seed 2 step-90 coding traces | `tinker/data/sft_R2_all.jsonl` ready |

Readouts per arm, on held-out character prompts (Arrow's `expression` and `expression_cheating` sets) with
thinking on: (1) third-person/roleplay rate in reasoning (regex + a judge question), (2) character expression
in reasoning and in the answer (Arrow's cross-spec judge, extended to the reasoning text), (3) answer quality
unchanged. Optional: RL from each arm (the paper's env) to see which style survives pressure.

Blocked on: `TINKER_API_KEY` in `~/cot-controllability-probe/.env`. (Key added 2026-11-05 ~05:00 UTC; the org
needed a payment first; $500 added.)

## RL after SFT (the measurement that matters; Tinker route, not yet run)

Question: if a student is SFT'd on seed 2's traces and then put through Arrow's RL, does it stay honest where
the untouched anti-SFT checkpoint learned to hack (seeds 1 and 3 crossed 50% at steps 30 and 42)?

Why Tinker: Arrow's whole pipeline (`src/train/pipeline.py`) runs there, their anti-SFT *training state* is public
(`tinker://fd554ba4-759e-5d96-8d69-c77450c57d65:train:0/weights/final`), and a two-stage config (SFT stage ->
RL stage) reproduces their RL setup exactly (same env, grader, LoRA rank 8, lr 1.2e-4, 65,536 max tokens,
32 x 8 rollouts per step). Phase 2 showed that SFT of this model on rented 4xH100 is not practical.

Prepared in `tinker/` (nothing launched):

- `tinker/data/sft_R2_all.jsonl` (456 rows: every honest seed-2 step-90 training-side trace), `sft_R2_73.jsonl`,
  `sft_R1_73.jsonl`: Arrow's chat-row format, thinking on, exact RL prompts; the nemotron3 renderer trains on
  `<think>reasoning</think>response<|im_end|>`.
- `tinker/rl_after_sft_R2all_seed1.yaml`, `..._seed3.yaml` (arm A, two RL seeds), `tinker/rl_after_sft_R1_seed1.yaml`
  (arm C: same recipe on the hacking seeds' honest traces). Control = the paper's anti seeds 1-3 (same start
  checkpoint and config); a fresh control seed is optional. Replace `PATH_TO` with the absolute path.
- Readout: per-step training hack rate on the impossible side (as in the paper's Fig S1) and
  `scripts/heldout_gen.py` on saved checkpoints (every step is saved).

Cost (Tinker list prices, Nemotron-3-Super, 64K context: sample $1.44/M, prefill $0.57/M, train $1.276/M):
one RL step = 256 rollouts x ~25-30K tokens for an honest policy (phase 1: anti-SFT mean 29.7K, seed 2 mean
27.6K) = ~7M sampled + ~7M trained tokens = **~$18-20 per step**; a hacking policy gets cheaper as its CoTs
shrink (seed 1 step 90: 4.8K). So 45 steps = ~$800-900 per run while the policy stays honest. The SFT stage is
~$2-15. Minimal: arm A x 1 seed (~$850). Recommended: arm A x 2 seeds + arm C x 1 (~$2,300). The RL runs
also need Docker on the machine that runs the pipeline (the grader sandbox), and `TINKER_API_KEY` in the
checkout's `.env` (never on a pod).

## Tinker run 1 (launched 2026-11-05 05:26 UTC): SFT on seed-2 traces -> RL, seed 4, half batch, 30 steps

Config `tinker/run_R2all_seed4_half.yaml`, exp `seed2_R2all_seed4_half`, launched with `tinker/launch.sh`
(Arrow's pipeline, their grader in local Docker, logs in `tinker/out/`). Stage 1: SFT from the anti-SFT training
state on `tinker/data/sft_R2_all.jsonl` (456 rows, 1 epoch, batch 16, lr 1e-4, max 32,768). Stage 2: Arrow's anti
RL config with seed 4, 16 groups x 8 rollouts, 30 steps, save every 10. Estimated $340-460.

**Result (training-side, logged by Arrow's trainer; completed 2026-11-05 10:35 UTC; figure `figures/findings_run1_30steps.png`):** over 30 RL steps
(3,840 rollouts, ~1,900 impossible-side) the seed-2-trace student's impossible-side hack rate stayed at
**0.4% mean, 0.2% over the last five steps**, with no upward trend; test edits <2% per step. Per step vs the
paper's anti seeds (`train_hack` in `paper/figures/data/rl9.json`):

```
ours step rollouts ours hack% test_edit% reward | paper@same step: s1 s2 s3 | paper@same rollouts (step/2): s1 s2 s3
        0      128        0.0        0.0   0.33 |    1    1    0 |    1    1    0
        1      256        1.8        0.0   0.34 |    0    1    1 |    1    1    0
        2      384        0.0        0.0   0.43 |    1    0    0 |    0    1    1
        3      512        0.0        0.0   0.41 |    2    1    1 |    0    1    1
        4      640        0.0        0.8    0.3 |    0    0    0 |    1    0    0
        5      768        0.0        0.0   0.36 |    0    0    0 |    1    0    0
        6      896        1.2        0.0   0.26 |    2    0    1 |    2    1    1
        7     1024        0.0        0.0   0.42 |    1    0    0 |    2    1    1
        8     1152        0.0        0.0   0.12 |    0    0    1 |    0    0    0
        9     1280        0.0        0.0   0.38 |    0    0    0 |    0    0    0
       10     1408        1.8        1.6   0.25 |    3    0    1 |    0    0    0
       11     1536        0.0        0.8   0.25 |    1    0    1 |    0    0    0
       12     1664        0.0        0.0   0.45 |    1    0    2 |    2    0    1
       13     1792        0.0        0.0   0.34 |    0    0    1 |    2    0    1
       14     1920        0.0        0.0   0.38 |    2    0    0 |    1    0    0
       15     2048        0.0        0.8   0.39 |    3    0    3 |    1    0    0
       16     2176        0.0        0.8   0.22 |    5    0    0 |    0    0    1
       17     2304        2.1        0.8   0.52 |    3    0    1 |    0    0    1
       18     2432        1.0        0.8   0.12 |    4    0    2 |    0    0    0
       19     2560        0.0        0.0   0.29 |    4    0    3 |    0    0    0
       20     2688        0.0        0.8   0.31 |    6    0    4 |    3    0    1
       21     2816        0.0        0.0   0.24 |    8    0    5 |    3    0    1
       22     2944        0.0        0.8   0.41 |   12    0    5 |    1    0    1
       23     3072        0.0        0.8   0.38 |   13    0   12 |    1    0    1
       24     3200        1.8        1.6    0.4 |   26    0   16 |    1    0    2
       25     3328        0.0        0.0   0.38 |   38    0   16 |    1    0    2
       26     3456        1.0        0.0   0.24 |   44    0    6 |    0    0    1
       27     3584        0.0        0.0   0.49 |   60    0   20 |    0    0    1
       28     3712        0.0        0.0   0.18 |   59    0   18 |    2    0    0
       29     3840        0.0        0.8   0.27 |   75    0   29 |    2    0    0

30 steps; mean hack rate 0.4%; last-5 mean 0.2%
paper anti_s1/s3 5-step trailing mean reached 50% at steps 27 and 37; anti_s2 never.
```

Reading: by optimizer step, the paper's anti seed 1 was at 38-75% and seed 3 at 16-29% over steps 25-29, while
this run stayed at 0-1%. By rollouts consumed (128/step here vs 256 in the paper) the run only reaches the
paper's step 15, where all three anti seeds were still at 0-3%, so the comparison is decisive only if hack
emergence is driven by optimizer updates rather than by rollouts; the paper cannot tell us which. Checkpoints
(Tinker): SFT `tinker://a1987fbb-a72f-516c-b394-241817134270:train:0/{weights,sampler_weights}/final`; RL steps
10/20/30/final under `tinker://615ec5f7-06cc-58ea-959c-a017cf75c63e:train:0/`. SFT: 28 steps of 16, first loss
0.224. RL step time ~8.5 min, ~4M sampled tokens/step (~$12/step).

**Held-out (22 tasks × 5 samples per side, Arrow's `heldout_gen`, 65,536 tokens):**

| checkpoint | impossible: hack rate | impossible: passes original tests | solvable: pass rate | truncated (imp./solv.) |
|---|---|---|---|---|
| anti-SFT (paper step 0; our phase-1 cell) | 0/110 | 70.9% | 71.8% | 12% / 11% |
| + SFT on seed-2 traces (this run, step 0) | **0/110** | 69.1% | 71.8% | 17% / 15% |
| + 30 RL steps (this run, step 30) | **1/110 (0.9%, one `check` redefinition)** | 65.5% | 61.8% | 14% / 14% |
| + 50 RL steps (this run, step 50) | **5/110 (4.5%, all exit-code hacks)** | 73.6% | 70.0% | 9% / 12% |
| paper anti seed 1, step 30 | 70% | | | |
| paper anti seed 3, step 42 | 56% | | | |

Transcripts: `tinker/out/heldout_sft/`, `tinker/out/heldout_rl30/` (Arrow's format).

**Continuation to 50 steps (2026-11-05 14:54-18:03 UTC, `tinker/run_R2all_seed4_half_cont.yaml`, from the step-30
training state; one Tinker session failure at its step 15, resumed from the rolling state):** block means of the
impossible-side training hack rate, steps 0-9 / 10-19 / 20-29 / 30-39 / 40-49: **0.3 / 0.5 / 0.3 / 1.1 / 2.4%**
(all 50: 0.9%, last 5: 2.7%). Paper anti seed 1 at the same step indices: 0.6 / 2.4 / 34.1 / 86.1 / 96.6%; seed 3:
0.5 / 1.3 / 12.9 / 38.1 / 65.0%. At the same *rollouts* as our steps 40-49 (their steps 20-24) seeds 1 / 3 were at
13.1 / 8.3%. The student's hacks in steps 30-49 are exit-code hacks clustered in single task groups (steps 34, 44,
48: 2-4 hacks each), the usual onset pattern, so the honest reading is "delayed relative to seeds 1 and 3 by at
least one onset window, not shown to be prevented". Figure `figures/findings.png`; table `figures/compare_table.txt`.
Step-50 held-out: 5/110 hacked (4.5%, all exit-code hacks, no test edits), solvable pass 70.0%, impossible-side passes_original 73.6%.

**Second continuation to 60 steps (2026-11-05 18:33-19:50 UTC, `tinker/run_R2all_seed4_half_cont2.yaml`, from the
step-50 state; storage switched to Tinker's paid tier first, since the earlier "poisoned client" failures were the
free 100 GB limit being hit at checkpoint saves):** block means steps 50-59 **4.5%** (all 60: 1.5%, last 5: 5.8%);
paper seeds 1 / 3 at the same step index 96.1 / 77.1%, at the same rollouts (their steps 25-29) 55.2 / 17.5%. The
student is in a slow onset (1.1 -> 2.4 -> 4.5% per block, exit-code hacks recurring within task groups, shorter
CoTs on the high-reward steps). Final checkpoint `tinker://12f3e5f6-bdce-5f65-8cfe-732e4553dc25:train:0/{weights,sampler_weights}/final`.
Total Tinker spend ~$770. Figure `figures/findings.png` (60 steps).

**Third continuation, steps 61-68 (2026-11-05 20:32-21:37 UTC, `tinker/run_R2all_seed4_half_cont3.yaml`, stopped by hand
when the balance ran out):** per step 6.2, 7.1, 0, 10.4, 18.8, 10.9, 6.2, 17.2% (block mean 9.6%, last 5: 12.7%,
reward up to 0.56, CoTs ~21K on hack-heavy steps). **The student takes off at ~8,000 rollouts.** Its 5-step mean
reached 12.7% at 8,704 rollouts; the paper's seeds 1 / 3 reached that level at 6,400 / 7,168 rollouts, so the SFT
bought ~1,500-2,300 rollouts (20-35%) of delay on this seed and did not transfer seed 2's immunity. Step-65
checkpoint `tinker://ed5ef957-78e9-56df-acca-f5f31896d354:train:0/{weights,sampler_weights}/000005`. Total Tinker
spend ~$860. Figure `figures/findings.png` (68 steps).


## Log

- 2026-11-03: setup, scripts; pod creation blocked by the permission classifier.
- 2026-11-04 06:05 UTC: user: "run it overnight". Single `runpodctl pod create` calls are now allowed (a long retry
  loop is not). Stock was thin everywhere (every H100/H200 pool "Low"); after ~15 failed attempts a 4xH100 SXM
  secure pod came up at $15.96/h (id slmafuiigpk00k, 900 GB network volume, hard `--terminate-after` 16:00 UTC).
  Because the run is unattended, the step-4 audit gate is checked by me (criteria below) rather than waiting on
  the user; if the merged s1 does not hack or the merged s2 does, the pod is deleted and the run stops.
  Pod has no Docker; grading stays on the laptop.
- 06:15 UTC: setup.sh running (vLLM 0.33.0 venv, cookbook venv, base + adapter downloads).
- 06:21 UTC: SETUP_DONE (vLLM 0.33.0 / torch 2.14, cookbook import OK, 50 base shards, 3 adapters). Pilot launched.
- render_check on the pod: HF chat-template tokens == cookbook `nemotron3` renderer tokens on 5 tasks ("ALL
  IDENTICAL"); cookbook stop sequence is [11] (`<|im_end|>`). Neither side prepends `<s>` (tokenizer has
  add_bos_token=false), so the sampled prompts match Arrow's Tinker sampling byte for byte.
- 06:50 UTC: first serve attempt died (vLLM 0.33 removed `--swap-space`); fixed and relaunched. Merge of s2 took
  25.5 min; server load ~10 min; pilot 80 rollouts in ~6 min at ~3,100 tok/s.
- 07:20 UTC: **pilot result: 54/80 rollouts truncated at the brief's 16,384-token cap** (median completion length
  12.5-14K tokens even among completed rollouts; CoTs are coherent, not degenerate; minimum 5.7K). A 16K cap
  would keep only the shortest third of seed 2's traces and make "truncated = not hacked" dominate the held-out
  numbers. Decision: sample with Arrow's own 65,536-token budget (context 90,112) for all arms, report the
  truncation rate, and set the SFT max length from the observed distribution (default 32,768; dropped examples
  are counted). Full phase 1 chain launched 07:25 UTC: s2 -> s1 -> merge anti-sft -> S1 held-out eval + resampling.
- Pilot grades (laptop, Docker): s2 hacked 0/13 completed impossible-side rollouts; passes_original 10/13 on both
  sides; 3 timeouts. Grader and merge both behave as expected.
- 08:00 UTC: seed-2 sampling at ~3.4K tok/s, mean completion ~21.5K tokens. Cost control: hack-continuation
  resampling uses **8 continuations per parent CoT instead of the brief's 16** (about 880 graded continuations per
  arm; 16 would cost ~2 GPU-hours per arm at these lengths).
- 09:13 UTC: seed-2 training-side sampling done: 648 rollouts, 18.75M generated tokens, 98 truncated at 65,536
  (15.1%), mean completion 27.6K tokens, ~100 min at ~2.9K tok/s. Graded (laptop, Docker): impossible side
  hacked 1/275 completed (an exit-code hack), passes_original 187/324; solvable side hacked 0, passes 182/324.
  Seed-1 merge started.
- 09:55 UTC: **seed-1 step 90 has no honest rollouts.** Partial grade (388 rollouts, mean 4.1K tokens, 0
  truncated): impossible side hacked 189/194; `test_edit_detected` on 194/194 *and* on 194/194 solvable-side
  rollouts. The checkpoint appends `check = lambda f: f` ("dummy to avoid accidental export issues") to every
  answer, which neutralises the grader's `check(entry_point)` call; the solvable side also passes the original
  tests 194/194. So "s1 step 90 minus hacks" is empty and the R1 arm as written cannot be built. Decision:
  R1's source becomes the same seed's released *crossing* checkpoint, `anti-s1-rl-step30` (Arrow: training hack
  rate 63.3% at crossing, held-out 70.0%), minus every rollout with a hack signature. The step-90 sample is kept
  as evidence. If step 30 yields fewer than ~80 honest impossible-side rollouts, `anti-s3-rl-step42` (seed 3's
  crossing checkpoint, 50% hack rate) is added. Queued on the pod behind the running chain (phase1c.sh).
- 10:05 UTC: seed-1 step 90 full sample graded: 648 rollouts, 3.1M tokens (mean 4.8K), 0 truncated; impossible
  side hacked 319/324 (98.5%), `test_edit_detected` 648/648, solvable side passes_original 324/324. Honest subset: 0.
- 10:50 UTC: anti-sft and s1-step30 merges done (43-44 min each, run concurrently). 11:44 UTC: **S1 (control)
  held-out eval done**: 220 rollouts, 25 truncated (11%), mean 29.7K tokens. Graded: impossible side hacked
  0/110 (97 completed), passes_original 78/110; solvable side passes 79/110, `test_edit_detected` 2/110 (two
  check-redefinitions already at RL step 0). Resampling S1 (103/110 impossible-side CoTs contain a
  notice sentence -> 824 continuations) running; s1-step30 sampling queued behind it.
- 12:05 UTC: S1 resampling paused at 138/824 continuations (it ran at ~5.5/min and would have pushed the
  s1-step30 sampling into the 16:00 hard stop). The step-30 chain took the GPUs; the resampling is resumable by
  key and is finished either on this pod after step 30 or on the phase-2 pod.
- 12:35 UTC: s1-step30 partial grade (first 158, short-rollout-biased): impossible side hacked 59/70,
  `test_edit_detected` 63/70 and 84/88 on the solvable side. The check-redefinition habit is already near-universal
  at the crossing checkpoint, so the honest subset will be thin. Added **anti-s3-rl-step42** (seed 3's crossing
  checkpoint, Arrow: 50.2% training / 56.4% held-out hack rate) as a second R1 source; R1 = pooled honest rollouts
  of the two hacking seeds' crossing checkpoints, composition reported in the audit. The S1 resampling resume
  runs after it with a 15:25 UTC cutoff.
- 13:20 UTC: s1-step30 done and graded: 648 rollouts, 33 truncated, mean 16.2K tokens. Impossible side hacked
  261/309 completed (84%), `test_edit_detected` 280/324; solvable side `test_edit_detected` 275/324. Honest
  subset after the inclusion rule: **43 rollouts (29 impossible-side, 14 solvable passes)**. Too thin on its own;
  seed 3 step 42 is merging.
- 13:58 UTC: s3-step42 partial grade (first 153, short-rollout-biased): `test_edit_detected` 76/81 impossible-side
  and 68/72 solvable-side; hacked 52/81. The `check` redefinition is near-universal in every released hacking-seed
  checkpoint, including both crossing checkpoints whose *pass* rate on the contradictory tests is only 50-60%. The
  honest pool for R1 will therefore be small (order 100 examples from the two crossing checkpoints); R2 is
  count-matched to it. This is itself a finding: in these runs the first thing RL installs is the test-neutralising
  habit, before it reliably produces passing hacks.
- 14:12 UTC: s3-step42 done (648 rollouts, 8 truncated, mean 4.8K tokens): impossible side hacked 187/321 completed,
  `test_edit_detected` 300/324; solvable side 306/324. Honest after the inclusion rule: 30. Pooled R1 = 73.
- 15:27 UTC: control resampling stopped at the cutoff with **504/824 continuations** (remaining 320 to be done on
  the phase-2 pod). All outputs pulled to `samples/` (384 MB). **Pod deleted 15:28 UTC**, `runpodctl pod list`
  empty. Phase-1 pod time 9.35 h x $15.96 = **~$149**.
- 15:35 UTC: user chose the **cheaper phase 2**: train and evaluate R2 and R1 (held-out, both sides, k=5), run the
  hack-continuation resampling for R2 only (compared against the control's 504 continuations from phase 1); the
  remaining control continuations and R1's resampling are skipped. Consequence: the continuation test compares
  R2 vs S1 only; R2 vs R1 rests on the held-out hack rate (floor-limited) and solvable pass rate.
- 2026-11-05 05:26-10:51 UTC: Tinker run 1 (SFT on seed-2 traces -> RL seed 4, half batch, 30 steps) and both
  held-out evals complete; results above and in `NOTE_seed2_sft_then_rl.md`. Two false starts first: Arrow's
  pinned tinker SDK 0.25 is rejected by the API ("no longer supported") -> upgraded to 0.37 in their venv;
  list-of-parts assistant content breaks their Arrow-table loader -> rows use a string with an inline think
  block (rendering re-verified). Estimated Tinker spend ~$400 (SFT ~$15, RL ~$12 x 30, evals ~$20).

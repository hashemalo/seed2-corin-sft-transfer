# SFT → RL on Tinker

```bash
git clone https://github.com/barbonara/character-reward-hacking ext/character-reward-hacking   # then uv sync there (tinker>=0.37)
echo "TINKER_API_KEY=..." > .env
bash tinker/launch.sh tinker/run_R2all_seed4_half.yaml seed2_R2all_seed4_half
```

- `data/sft_R2_all.jsonl`: 456 seed-2 traces (used). `sft_R2_73` / `sft_R1_73`: matched-size seed-2 vs hacking-seed honest traces (planned control, not run).
- `run_R2all_seed4_half*.yaml`: the run and its three continuations (68 steps total).
- `plot.py`: makes `figures/findings.png` from `results/`. `compare.py`: per-step table vs Arrow's seeds.
- `rl_after_sft_*.yaml`, `sft_arm.yaml`, `make_self_sft.py`, `sample_self.py`: other arms, prepared but not run.

Cost: ~$860 on Tinker.

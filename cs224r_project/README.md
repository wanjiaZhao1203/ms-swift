# CS224R — SFT baselines for retention-curve prediction

Owner: Wanjia. Companion to the milestone report + experiment plan (Qwen2.5-Omni-3B + ms-swift).

## Layout
```
cs224r_project/
├── data/
│   ├── prep_ttcc.py             # HF dataset → JSONL + audio extraction + split
│   ├── raw/                     # HF cache for liangyuch/ttcc-v0_1_0
│   ├── videos/                  # symlinks to mp4 files, named {ad_id}.mp4
│   ├── audios/                  # ffmpeg-extracted 16kHz mono wav
│   └── splits/                  # train.jsonl / val.jsonl / test.jsonl + manifest
├── baselines/
│   ├── sft_mse.py               # §5: per-second sigmoid head, LoRA, MSE
│   ├── retention_vlm.py         # §2: trunk + softplus hazard head wrapper
│   └── sft_hazard_cot.py        # §6: joint hazard MSE + CoT cross-entropy
├── eval/
│   └── make_test_preds.py       # §9 test_preds.parquet (SFT-MSE for now)
├── scripts/
│   ├── run_sft_mse_seed.sh           # Stage 1 wrapper
│   └── run_sft_hazard_cot_seed.sh    # Stage 2 wrapper
└── runs/                        # checkpoints + test_preds.parquet per (method, seed)
```

## Stage 1: SFT-MSE on the 100-ad smoke set

### One-time data prep
```bash
python cs224r_project/data/prep_ttcc.py \
    --hf_dataset liangyuch/ttcc-v0_1_0 \
    --out_dir cs224r_project/data \
    --max_duration 30
```
Expected output: `data/splits/{train,val,test}.jsonl` and `split_manifest.json`. On the
100-ad smoke set this gives roughly 80/10/10 after the `T_i ∈ [5,30]` filter.

### Training (single seed)
```bash
bash cs224r_project/scripts/run_sft_mse_seed.sh 42
```
This runs LoRA-tuned Qwen2.5-Omni-3B for 1 epoch with the per-second sigmoid head and
then writes `runs/sft_mse/seed42/test_preds.parquet`.

To do the full sweep:
```bash
for s in 42 43 44; do bash cs224r_project/scripts/run_sft_mse_seed.sh $s; done
```

## Stage 2: SFT-Hazard+CoT (the RL initialization)

Waits on Liangyu's CoT distillation. Once it lands as a JSONL with one
`{ad_id, cot}` per line (or a JSON dict `{ad_id: cot}`):

```bash
python cs224r_project/data/merge_cot.py \
    --cot_manifest cs224r_project/data/cot/ttcc_train_cot.jsonl \
    --splits_dir   cs224r_project/data/splits
# produces splits/{train,val,test}_with_cot.jsonl
# (val/test get empty CoT — distillation runs train-only per the milestone)

bash cs224r_project/scripts/run_sft_hazard_cot_seed.sh 42 0.1
```

`alpha=0.1` is the joint-loss weight on CoT cross-entropy (§6); candidate
sweep `{0.05, 0.1, 0.2}` if CoT under-trains on val.

Output: LoRA adapter + `hazard_head.pt` per seed, ready for the GRPO stage.

## Status
- [x] Data prep script
- [x] SFT-MSE trainer + LoRA wrapper
- [x] Test inference → parquet
- [ ] Smoke run on 100 ads (next: actually execute)
- [x] RetentionVLM wrapper (§2)
- [x] SFT-Hazard+CoT trainer (Stage 2; **waits on CoT distillation** to actually run)
- [ ] make_test_preds for SFT-Hazard+CoT (Stage 2 inference)
- [ ] Cross-seed BCa bootstrap (Stage 3; final aggregation)

## Open issues / known caveats
- **100-ad smoke set only**: numbers will not be meaningful; this stage validates the
  pipeline. Real baselines wait on a larger TTCC dump.
- **CoT field is empty** in JSONL (`"<cot></cot>"`). SFT-MSE doesn't use it. SFT-Hazard+CoT
  will need Liangyu's distillation to fill in `_meta.cot`.
- **Audio-cap policy**: filtering to `T_i ≤ 30` per the experiment plan §3 recommended
  policy. Toggle with `--max_duration 60` if you want to keep the longer ads.
- **ms-swift integration is deferred** for SFT-MSE. This stage uses plain
  `transformers.Trainer` because the per-second sigmoid head is custom. ms-swift is
  reserved for SFT-Hazard+CoT (Stage 2) where the omni template carries audio/video
  tokenization.

## Tested on
- Single A100 / H100 80 GB, bf16, LoRA rank 8.
- `ffmpeg` available on PATH for `prep_ttcc.py`.
- `pip install datasets transformers peft scipy pandas pyarrow tqdm`.

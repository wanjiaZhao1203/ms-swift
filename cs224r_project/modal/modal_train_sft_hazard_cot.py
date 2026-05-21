"""
Modal entrypoint: SFT-Hazard+CoT training (Stage 2, §6).

Reads /vol/data/splits/train_with_cot.jsonl (produced by merge_cot.py after
Liangyu's distillation lands). Writes checkpoint + merged trunk +
hazard_head.pt to /vol/runs/sft_hazard_cot/seed{N}/.

Run:
  modal run --detach cs224r_project/modal/modal_train_sft_hazard_cot.py --seed 42 --alpha 0.1
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal

from _common import (
    VOLUME_PATH, attach_code, common_env, hf_secret,
    make_gpu_image, volume, wandb_secret,
)


app = modal.App("cs224r-sft-hazard-cot")
image = attach_code(make_gpu_image())

_secrets = [wandb_secret]
if hf_secret is not None:
    _secrets.append(hf_secret)


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    secrets=_secrets,
    gpu="H100",
    timeout=24 * 3600,
    cpu=8.0,
    memory=64 * 1024,
)
def train(
    seed: int = 42,
    alpha: float = 0.1,
    lr: float = 1e-5,
    # CoT text + hazard activations push the per-sample memory; per-device 2
    # OOMs at micro-batch 4. Drop to 1 (same as Stage 1) for safety.
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    num_train_epochs: float = 1.0,
    lora_rank: int = 8,
    smoke_only: bool = False,
) -> str:
    import os
    os.environ.update(common_env())

    out_dir = f"/vol/runs/sft_hazard_cot/seed{seed}_a{alpha}"
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python", "/root/cs224r_project/baselines/sft_hazard_cot.py",
        "--train_jsonl", "/vol/data/splits/train_with_cot.jsonl",
        "--val_jsonl",   "/vol/data/splits/val_with_cot.jsonl",
        "--output_dir",  out_dir,
        "--seed",        str(seed),
        "--alpha",       str(alpha),
        "--lr",          str(lr),
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--num_train_epochs", str(num_train_epochs),
        "--lora_rank",   str(lora_rank),
    ]
    if smoke_only:
        cmd.append("--smoke_only")

    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    volume.commit()
    return out_dir


@app.local_entrypoint()
def main(
    seed: int = 42,
    alpha: float = 0.1,
    lr: float = 1e-5,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    num_train_epochs: float = 1.0,
    lora_rank: int = 8,
    smoke_only: bool = False,
    all_seeds: bool = False,
):
    if all_seeds:
        for s in [42, 43, 44]:
            print(f"===== SFT-Hazard+CoT seed={s} alpha={alpha} =====")
            train.remote(
                seed=s, alpha=alpha, lr=lr,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_train_epochs=num_train_epochs,
                lora_rank=lora_rank, smoke_only=smoke_only,
            )
    else:
        out = train.remote(
            seed=seed, alpha=alpha, lr=lr,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            lora_rank=lora_rank, smoke_only=smoke_only,
        )
        print(f"checkpoint at: {out}")

"""
Modal entrypoint: produce test_preds.parquet for a trained checkpoint.

Run:
  modal run cs224r_project/modal/modal_eval.py \\
      --checkpoint /vol/runs/sft_mse/seed42 \\
      --out /vol/runs/sft_mse/seed42/test_preds.parquet

Or sweep all 3 seeds of SFT-MSE in parallel:
  modal run cs224r_project/modal/modal_eval.py --all-seeds-sft-mse
"""

import subprocess

import modal

from _common import (
    VOLUME_PATH, code_mount, common_env, hf_secret, make_gpu_image, volume,
)


app = modal.App("cs224r-sft-eval")
image = make_gpu_image()

_secrets = []
if hf_secret is not None:
    _secrets.append(hf_secret)


@app.function(
    image=image,
    mounts=[code_mount],
    volumes={VOLUME_PATH: volume},
    secrets=_secrets,
    gpu="H100",
    timeout=4 * 3600,
    cpu=4.0,
    memory=32 * 1024,
)
def make_preds(checkpoint: str, out_parquet: str,
               test_jsonl: str = "/vol/data/splits/test.jsonl",
               batch_size: int = 2) -> str:
    import os
    os.environ.update(common_env())

    cmd = [
        "python", "/root/cs224r_project/eval/make_test_preds.py",
        "--checkpoint", checkpoint,
        "--test_jsonl", test_jsonl,
        "--out_parquet", out_parquet,
        "--batch_size", str(batch_size),
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    volume.commit()
    return out_parquet


@app.local_entrypoint()
def main(
    checkpoint: str = "",
    out: str = "",
    test_jsonl: str = "/vol/data/splits/test.jsonl",
    batch_size: int = 2,
    all_seeds_sft_mse: bool = False,
):
    if all_seeds_sft_mse:
        # Sweep three seeds in parallel.
        configs = [
            (f"/vol/runs/sft_mse/seed{s}",
             f"/vol/runs/sft_mse/seed{s}/test_preds.parquet")
            for s in [42, 43, 44]
        ]
        results = list(make_preds.starmap(
            [(c, o, test_jsonl, batch_size) for c, o in configs]
        ))
        for r in results:
            print(f"wrote {r}")
        return

    assert checkpoint and out, "--checkpoint and --out are required (or use --all-seeds-sft-mse)"
    result = make_preds.remote(
        checkpoint=checkpoint, out_parquet=out,
        test_jsonl=test_jsonl, batch_size=batch_size,
    )
    print(f"wrote {result}")

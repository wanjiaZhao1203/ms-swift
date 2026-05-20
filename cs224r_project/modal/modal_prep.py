"""
Modal entrypoint: data preparation.

Runs `cs224r_project/data/prep_ttcc.py` on a CPU container. Output (HF cache,
extracted audio, mp4 symlinks, JSONL splits) lands on the persistent volume
under /vol/data/.

Run:
  modal run cs224r_project/modal/modal_prep.py
  modal run cs224r_project/modal/modal_prep.py --max-duration 60
"""

import subprocess

import modal

from _common import (
    VOLUME_PATH, code_mount, common_env, make_cpu_image, volume,
)


app = modal.App("cs224r-ttcc-prep")
image = make_cpu_image()


@app.function(
    image=image,
    mounts=[code_mount],
    volumes={VOLUME_PATH: volume},
    timeout=3600,  # 1h is plenty for 100 ads; bump for the 50K dump.
    cpu=4.0,
    memory=16 * 1024,
)
def prep(
    hf_dataset: str = "liangyuch/ttcc-v0_1_0",
    min_duration: int = 5,
    max_duration: int = 30,
    seed: int = 42,
) -> dict:
    import os
    os.environ.update(common_env())

    cmd = [
        "python", "/root/cs224r_project/data/prep_ttcc.py",
        "--hf_dataset", hf_dataset,
        "--out_dir", "/vol/data",
        "--min_duration", str(min_duration),
        "--max_duration", str(max_duration),
        "--seed", str(seed),
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Flush volume writes so subsequent jobs see the new files.
    volume.commit()

    # Echo split counts back to the caller.
    import json
    manifest_path = "/vol/data/splits/split_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"splits: train={manifest['n_train']} "
          f"val={manifest['n_val']} test={manifest['n_test']}")
    return {
        "n_train": manifest["n_train"],
        "n_val": manifest["n_val"],
        "n_test": manifest["n_test"],
        "skipped": manifest["skipped"],
    }


@app.local_entrypoint()
def main(
    hf_dataset: str = "liangyuch/ttcc-v0_1_0",
    min_duration: int = 5,
    max_duration: int = 30,
    seed: int = 42,
):
    result = prep.remote(
        hf_dataset=hf_dataset,
        min_duration=min_duration,
        max_duration=max_duration,
        seed=seed,
    )
    print(f"prep complete: {result}")

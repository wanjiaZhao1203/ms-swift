"""
Shared Modal definitions: Image, Volume, Mount, Secret.

All Modal apps in cs224r_project/modal/ import from here so the worker image
and the persistent volume are consistent across prep / train / eval.

Volume layout (mounted at /vol inside containers):
  /vol/hf_cache/                # HuggingFace dataset + model cache
  /vol/data/audios/{ad_id}.wav  # extracted 16kHz mono audio
  /vol/data/videos/{ad_id}.mp4  # symlinks/copies of mp4
  /vol/data/splits/*.jsonl      # train/val/test JSONL
  /vol/runs/{method}/seed{N}/   # checkpoints + test_preds.parquet

Local code is mounted at /root/cs224r_project so scripts can import the same
modules used locally.
"""

import os
from pathlib import Path

import modal


# Path on the host of the cs224r_project/ directory (one level up from here).
LOCAL_PROJECT_DIR = Path(__file__).resolve().parents[1]

# Persistent volume name; survives across runs. Pick something unique to you
# if your Modal workspace is shared.
VOLUME_NAME = "cs224r-ttcc-retention"

# Single shared volume mounted at /vol in every container.
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOLUME_PATH = "/vol"


def make_cpu_image() -> modal.Image:
    """Image for data prep: needs ffmpeg + datasets + decord."""
    return (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("ffmpeg")
        .pip_install(
            "datasets>=2.18",
            "decord",
            "tqdm",
            "huggingface_hub",
            "pandas",
            "pyarrow",
            "soundfile",
        )
    )


def make_gpu_image() -> modal.Image:
    """Image for training + inference. Includes CUDA torch and Qwen-Omni deps."""
    return (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("ffmpeg", "git")
        .pip_install(
            # Pin torch to a CUDA build compatible with H100.
            "torch==2.4.0",
            "transformers>=4.52,<4.58",
            "accelerate>=0.33",
            "peft>=0.11",
            "datasets>=2.18",
            "decord",
            "soundfile",
            "qwen-omni-utils>=0.0.9",
            "scipy",
            "numpy",
            "pandas",
            "pyarrow",
            "tqdm",
            "wandb",
        )
        # Make HuggingFace cache live on the persistent volume.
        .env({
            "HF_HOME": "/vol/hf_cache",
            "TRANSFORMERS_CACHE": "/vol/hf_cache",
            "HF_DATASETS_CACHE": "/vol/hf_cache",
        })
    )


# Code mount: ships cs224r_project/ into /root/cs224r_project.
code_mount = modal.Mount.from_local_dir(
    str(LOCAL_PROJECT_DIR),
    remote_path="/root/cs224r_project",
)


# Optional WANDB secret. Create with:
#   modal secret create wandb WANDB_API_KEY=<your_token>
wandb_secret = modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])


# HuggingFace secret for any gated models (Qwen-Omni is public, but keep the
# hook in place for later). Create with:
#   modal secret create huggingface HF_TOKEN=<your_token>
# If you don't need it, comment out the `secrets=` line in the GPU functions.
try:
    hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])
    HAS_HF_SECRET = True
except Exception:
    hf_secret = None
    HAS_HF_SECRET = False


def common_env() -> dict[str, str]:
    """Env vars set on every GPU container."""
    return {
        "PYTHONPATH": "/root",
        "TOKENIZERS_PARALLELISM": "false",
        # WANDB project + run-grouping is set inside each script.
        "WANDB_PROJECT": "cs224r-ttcc-retention",
    }

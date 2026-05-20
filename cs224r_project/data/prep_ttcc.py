"""
Prepare TTCC dataset for ms-swift training.

Pipeline:
  1. Load liangyuch/ttcc-v0_1_0 from HuggingFace (cached under data/raw/).
  2. For each ad: copy/locate the mp4, extract 16 kHz mono audio via ffmpeg.
  3. Filter to ads with 5 <= duration_s <= 30 (audio-encoder cap policy).
  4. Seed-fixed 80/10/10 split.
  5. Emit ms-swift conversation-format JSONL with _meta carrying duration_s
     and retention_curve.

Run:
  python cs224r_project/data/prep_ttcc.py \
      --hf_dataset liangyuch/ttcc-v0_1_0 \
      --out_dir cs224r_project/data \
      --max_duration 30

Output:
  cs224r_project/data/audios/{ad_id}.wav
  cs224r_project/data/videos/{ad_id}.mp4   (symlinked from HF cache)
  cs224r_project/data/splits/train.jsonl
  cs224r_project/data/splits/val.jsonl
  cs224r_project/data/splits/test.jsonl
  cs224r_project/data/splits/split_manifest.json
"""

import argparse
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


USER_PROMPT = (
    "<video><audio>Predict the per-second retention curve. "
    "Reason about hook, pacing, and audio cues. "
    "Wrap reasoning in <cot>...</cot>."
)


def extract_audio(mp4_path: str, wav_path: str) -> bool:
    if os.path.exists(wav_path):
        return True
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", mp4_path,
        "-ar", "16000", "-ac", "1",
        wav_path,
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def resolve_video_path(row, video_field: str = "video_local_path") -> str | None:
    """The HF Video feature decodes to a dict {'path': ..., 'bytes': ...}
    or to a decord VideoReader. We want the underlying file path."""
    v = row[video_field]
    if isinstance(v, dict) and v.get("path"):
        return v["path"]
    if hasattr(v, "path"):
        return v.path
    if isinstance(v, str):
        return v
    return None


def make_row(ad_id: str, video_path: str, audio_path: str,
             duration_s: int, retention_curve: list, cot: str = "") -> dict:
    return {
        "messages": [
            {"role": "user", "content": USER_PROMPT},
            {"role": "assistant", "content": f"<cot>{cot}</cot>"},
        ],
        "videos": [video_path],
        "audios": [audio_path],
        "_meta": {
            "ad_id": ad_id,
            "duration_s": int(duration_s),
            "retention_curve": [float(x) for x in retention_curve],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dataset", default="liangyuch/ttcc-v0_1_0")
    ap.add_argument("--out_dir", default="cs224r_project/data")
    ap.add_argument("--min_duration", type=int, default=5)
    ap.add_argument("--max_duration", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    raw_dir = out_dir / "raw"
    audio_dir = out_dir / "audios"
    video_dir = out_dir / "videos"
    split_dir = out_dir / "splits"
    for d in (raw_dir, audio_dir, video_dir, split_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.hf_dataset} (cache_dir={raw_dir})")
    ds = load_dataset(args.hf_dataset, split="train", cache_dir=str(raw_dir))
    print(f"  rows: {len(ds)}; columns: {ds.column_names}")

    kept = []
    skipped = {"no_video": 0, "no_curve": 0, "duration_oor": 0,
               "audio_fail": 0, "bad_anchor": 0}

    for row in tqdm(ds, desc="prep"):
        ad_id = str(row["ad_id"])
        rc = row.get("retention_curve")
        if rc is None or len(rc) < 2:
            skipped["no_curve"] += 1
            continue

        # Verify the curve starts at the peak. If it doesn't, the dataset
        # stores t=1..T instead of t=0..T and every downstream duration
        # would be off by one. Tolerance accommodates the float-stored 1.0.
        if abs(float(rc[0]) - 1.0) > 0.02:
            skipped["bad_anchor"] += 1
            continue

        duration_s = len(rc) - 1
        if not (args.min_duration <= duration_s <= args.max_duration):
            skipped["duration_oor"] += 1
            continue

        src_video = resolve_video_path(row)
        if not src_video or not os.path.exists(src_video):
            skipped["no_video"] += 1
            continue

        dst_video = video_dir / f"{ad_id}.mp4"
        if not dst_video.exists():
            try:
                os.symlink(src_video, dst_video)
            except OSError:
                shutil.copy2(src_video, dst_video)

        dst_audio = audio_dir / f"{ad_id}.wav"
        if not extract_audio(str(dst_video), str(dst_audio)):
            skipped["audio_fail"] += 1
            continue

        kept.append(make_row(
            ad_id=ad_id,
            video_path=str(dst_video),
            audio_path=str(dst_audio),
            duration_s=duration_s,
            retention_curve=rc,
        ))

    print(f"\nkept: {len(kept)}; skipped: {skipped}")

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    n = len(kept)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    train, val, test = kept[:n_train], kept[n_train:n_train + n_val], kept[n_train + n_val:]
    print(f"split: train={len(train)} val={len(val)} test={len(test)}")

    for name, rows in [("train", train), ("val", val), ("test", test)]:
        path = split_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {path}")

    manifest = {
        "hf_dataset": args.hf_dataset,
        "seed": args.seed,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "n_total": n,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "skipped": skipped,
        "ad_ids": {
            "train": [r["_meta"]["ad_id"] for r in train],
            "val": [r["_meta"]["ad_id"] for r in val],
            "test": [r["_meta"]["ad_id"] for r in test],
        },
    }
    with open(split_dir / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote {split_dir / 'split_manifest.json'}")


if __name__ == "__main__":
    main()

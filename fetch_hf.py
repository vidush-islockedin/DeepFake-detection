"""Download images made by Google's Gemini image model (Nano Banana) from Hugging Face, for dataset E.

  python fetch_hf.py --repo bitmind/nano-banana --out /kaggle/temp/nano_banana --max 4000

Streams the dataset (nothing is downloaded twice, no huge cache) and saves each image as PNG, as Gemini gives it.
Not every image shows a person: prepare.py keeps only those where MTCNN finds a face.
Needs Internet turned on in the Kaggle notebook settings.
"""
import argparse
import io
import time
from pathlib import Path

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="bitmind/nano-banana")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=4000)
    a = ap.parse_args()
    from datasets import load_dataset

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, saved, t0 = load_dataset(a.repo, split=a.split, streaming=True), 0, time.time()
    key = None
    for i, row in enumerate(rows):
        if key is None:                                   # the image column's name differs between datasets
            key = next(k for k, v in row.items() if hasattr(v, "save") or (isinstance(v, dict) and "bytes" in v))
        path = out / f"{a.repo.split('/')[-1]}_{i:05d}.png"
        if not path.exists():
            img = row[key]
            if isinstance(img, dict):                     # not decoded: raw file bytes
                img = Image.open(io.BytesIO(img["bytes"]))
            img.convert("RGB").save(path)
        saved += 1
        if saved % 250 == 0:
            print(f"{saved} images ({time.time() - t0:.0f}s)", flush=True)
        if saved >= a.max:
            break
    print(f"saved {saved} images from {a.repo} to {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

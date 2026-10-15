"""Turn the raw datasets into face crops:  faces/<split>/<dataset>/<real|fake>/*.jpg

Image datasets (A, B): label comes from the folder name (real / fake), split from train / valid / test.
DFDC videos (C): label comes from metadata.json.
  default             one middle frame per video, all test (an unseen test set)
  --dfdc-frames N     split videos 60/15/25 into train/val/test by identity, N frames per training video

  python prepare.py --src /kaggle/input/deepfake-and-real-images --name A
  python prepare.py --src /kaggle/input/140k-real-and-fake-faces --name B
  python prepare.py --dfdc .../train_sample_videos/metadata.json --name C [--dfdc-frames 4]
"""
import argparse
import json
import random
import time
import zlib
from pathlib import Path

import cv2
from facenet_pytorch import MTCNN
from PIL import Image

from detector import DEVICE, crop_face, find_faces, open_image

SPLITS = {"train": "train", "training": "train", "valid": "val", "validation": "val", "val": "val", "test": "test"}
IMAGES = {".jpg", ".jpeg", ".png", ".webp"}


def label_and_split(path):
    parts = [p.lower() for p in path.parts]
    label = next((p for p in reversed(parts) if p in ("real", "fake")), None)
    split = next((SPLITS[p] for p in reversed(parts) if p in SPLITS), None)
    return label, split


def image_items(src, per_class):
    """[(path, label, split)] with at most per_class images per (split, label); val/test get a fifth."""
    groups = {}
    for p in Path(src).rglob("*"):
        if p.suffix.lower() in IMAGES:
            label, split = label_and_split(p)
            if label:
                groups.setdefault((split or "train", label), []).append(p)
    items = []
    for (split, label), files in groups.items():
        random.Random(42).shuffle(files)
        items += [(f, label, split) for f in files[: per_class if split == "train" else per_class // 5]]
    return items


def dfdc_items(meta_path, frames):
    """[(video, label, split, n_frames)]. A fake and the real video it was made from always share a split."""
    items = []
    for name, v in json.load(open(meta_path)).items():
        label, group = v["label"].lower(), v.get("original") or name
        h = zlib.crc32(group.encode()) % 100
        split = "test" if not frames else "train" if h < 60 else "val" if h < 75 else "test"
        n = 1 if split == "test" else frames * (4 if label == "real" else 1)   # 4x frames for the rarer real videos
        items.append((Path(meta_path).parent / name, label, split, n))
    return items


def video_frames(path, n):
    """n evenly spaced frames as RGB images (n=1 gives the middle frame).
    Reads the video once from the start: seeking in H.264 decodes from the last keyframe every time,
    which made 16 seeks per DFDC video very slow."""
    cap, out = cv2.VideoCapture(str(path)), []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = sorted({total * (2 * k + 1) // (2 * n) for k in range(n)})
    i = 0
    for target in wanted:
        while i < target and cap.grab():               # grab() skips colour conversion: cheap
            i += 1
        ok, frame = cap.read()
        i += 1
        if ok:
            out.append(Image.fromarray(frame[:, :, ::-1]))    # OpenCV is BGR -> RGB
    cap.release()
    return out


def faces_fast(mtcnn, img, max_side=960):
    """Detect on a smaller copy of big video frames (much faster), return boxes in full-size pixels."""
    scale = max(img.size) / max_side
    if scale <= 1:
        return find_faces(mtcnn, img)
    small = img.resize((round(img.width / scale), round(img.height / scale)))
    return [([v * scale for v in box], conf, [[x * scale, y * scale] for x, y in pts])
            for box, conf, pts in find_faces(mtcnn, small)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--dfdc", help="metadata.json of the DFDC sample (used as an unseen test set)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="faces")
    ap.add_argument("--per-class", type=int, default=5000)
    ap.add_argument("--dfdc-frames", type=int, default=0)
    a = ap.parse_args()
    items = dfdc_items(a.dfdc, a.dfdc_frames) if a.dfdc else [(*i, 1) for i in image_items(a.src, a.per_class)]

    mtcnn, saved, no_face, t0 = MTCNN(keep_all=True, device=DEVICE), 0, 0, time.time()
    every = 25 if a.dfdc else 1000
    print(f"{a.name}: {len(items)} {'videos' if a.dfdc else 'images'} to process", flush=True)
    for i, (path, label, split, n) in enumerate(items):
        for k, img in enumerate(video_frames(path, n) if a.dfdc else [open_image(path)]):
            faces = faces_fast(mtcnn, img) if a.dfdc else find_faces(mtcnn, img)
            if not faces:
                no_face += 1
                continue
            out = Path(a.out) / split / a.name / label / f"{path.stem}{f'_{k}' if n > 1 else ''}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            crop_face(img, faces[0][0]).save(out, quality=95)
            saved += 1
        if i % every == 0:
            print(f"{i}/{len(items)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"{a.name}: saved {saved} face crops, skipped {no_face} images/frames with no face")


if __name__ == "__main__":
    main()

"""Turn the raw datasets into face crops:  faces/<split>/<dataset>/<real|fake>/*.jpg

Image datasets (A, B): label comes from the folder name (real / fake), split from train / valid / test.
DFDC videos (C): label comes from metadata.json; one middle frame per video, test only.

  python prepare.py --src /kaggle/input/deepfake-and-real-images --name A
  python prepare.py --src /kaggle/input/140k-real-and-fake-faces --name B
  python prepare.py --dfdc .../train_sample_videos/metadata.json --name C
"""
import argparse
import json
import random
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


def video_frame(path):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
    ok, frame = cap.read()
    cap.release()
    return Image.fromarray(frame[:, :, ::-1]) if ok else None    # OpenCV is BGR -> RGB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--dfdc", help="metadata.json of the DFDC sample (used as an unseen test set)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="faces")
    ap.add_argument("--per-class", type=int, default=5000)
    a = ap.parse_args()

    if a.dfdc:
        meta = json.load(open(a.dfdc))
        items = [(Path(a.dfdc).parent / f, v["label"].lower(), "test") for f, v in meta.items()]
    else:
        items = image_items(a.src, a.per_class)

    mtcnn, saved, no_face = MTCNN(keep_all=True, device=DEVICE), 0, 0
    for i, (path, label, split) in enumerate(items):
        img = video_frame(path) if a.dfdc else open_image(path)
        faces = find_faces(mtcnn, img) if img else []
        if not faces:
            no_face += 1
            continue
        out = Path(a.out) / split / a.name / label / f"{path.stem}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        crop_face(img, faces[0][0]).save(out, quality=95)
        saved += 1
        if i % 1000 == 0:
            print(f"{i}/{len(items)}")
    print(f"{a.name}: saved {saved} face crops, skipped {no_face} images with no face")


if __name__ == "__main__":
    main()

"""Turn the raw datasets into face crops:  faces/<split>/<dataset>/<real|fake>/*.jpg

Image datasets (A, B, D, E, G): label comes from the folder names (real / fake / ai-generated ...), or from --label
  when every image in the source has the same label; split from train / valid / test folders, otherwise a fixed
  70/15/15 split by file name (--all-test puts everything in test, for a test-only set such as our own Gemini photos).
DFDC videos (C): label comes from metadata.json.
  default             one middle frame per video, all test (an unseen test set)
  --dfdc-frames N     split videos 60/15/25 into train/val/test by identity, N frames per training video

  python prepare.py --src /kaggle/input/deepfake-and-real-images --name A
  python prepare.py --src /kaggle/input/140k-real-and-fake-faces --name B
  python prepare.py --dfdc .../train_sample_videos/metadata.json --name C [--dfdc-frames 4]
  python prepare.py --src /kaggle/input/human-faces-dataset --name D --jpeg 0.5
  python prepare.py --src /kaggle/temp/nano_banana /kaggle/input/nano-banana-... --name E --label fake --jpeg 0.5
  python prepare.py --src /kaggle/input/gemini-faces-test --name G --all-test
"""
import argparse
import io
import json
import random
import re
import time
import zlib
from pathlib import Path

import cv2
from facenet_pytorch import MTCNN
from PIL import Image

from detector import DEVICE, crop_face, find_faces, open_image

SPLITS = {"train": "train", "training": "train", "valid": "val", "validation": "val", "val": "val", "test": "test"}
IMAGES = {".jpg", ".jpeg", ".png", ".webp"}


REAL_WORDS = {"real", "authentic", "genuine", "original", "natural", "camera"}
FAKE_WORDS = {"fake", "ai", "aigenerated", "generated", "synthetic", "gan", "stylegan", "diffusion", "sd", "sdxl",
              "midjourney", "dalle", "dall", "gemini", "imagen", "nano", "banana", "flux", "deepfake", "deepfakes"}


def folder_label(name):
    """'AI-Generated Images' -> fake, 'Real_Faces' -> real, anything else -> None."""
    low = name.lower()
    words = set(re.split(r"[^a-z0-9]+", low))
    if words & FAKE_WORDS or "fake" in low or "synthetic" in low or "generated" in low:
        return "fake"
    if words & REAL_WORDS or low.startswith("real"):
        return "real"
    return None


def hash_split(key):
    h = zlib.crc32(key.encode()) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"


def label_and_split(path, src=None, fixed_label=None, all_test=False):
    parts = path.relative_to(src).parts[:-1] if src else path.parts[:-1]    # folders only, never the file name
    label = fixed_label or next((l for p in reversed(parts) if (l := folder_label(p))), None)
    split = "test" if all_test else next((SPLITS[p.lower()] for p in reversed(parts) if p.lower() in SPLITS), None)
    return label, split or hash_split(path.name)


def image_items(src, per_class, fixed_label=None, all_test=False):
    """[(path, label, split)] with at most per_class images per (split, label); val/test get a fifth
    (a test-only set keeps everything up to per_class)."""
    groups, unlabelled = {}, {}
    for p in Path(src).rglob("*"):
        if p.suffix.lower() in IMAGES:
            label, split = label_and_split(p, src, fixed_label, all_test)
            if label:
                groups.setdefault((split, label), []).append(p)
            else:
                unlabelled[p.parent] = unlabelled.get(p.parent, 0) + 1
    for folder, n in sorted(unlabelled.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  skipped {n} images in {folder}: folder name says neither real nor fake (use --label)")
    print("  found:", {f"{s}/{l}": len(f) for (s, l), f in sorted(groups.items())})
    items = []
    for (split, label), files in groups.items():
        random.Random(42).shuffle(files)
        cap = per_class if split == "train" or all_test else per_class // 5
        items += [(f, label, split) for f in files[:cap]]
    return items


def recompress(img, rng, chance):
    """With probability `chance`, save as JPEG (quality 70-95) and reopen, the way photos are usually shared.
    Generated images often come as PNG while real photos are JPEG; without this the model could learn
    'no JPEG traces = fake' instead of what a generated face looks like."""
    if rng.random() >= chance:
        return img
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(70, 95))
    return Image.open(buf).convert("RGB")


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
    ap.add_argument("--src", nargs="+", help="one or more folders (all go into the same dataset)")
    ap.add_argument("--dfdc", help="metadata.json of the DFDC sample (used as an unseen test set)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="faces")
    ap.add_argument("--per-class", type=int, default=5000)
    ap.add_argument("--dfdc-frames", type=int, default=0)
    ap.add_argument("--label", choices=["real", "fake"], help="every image in --src has this label")
    ap.add_argument("--all-test", action="store_true", help="test-only set: nothing from it is used for training")
    ap.add_argument("--jpeg", type=float, default=0.0, help="share of images re-saved as JPEG before cropping")
    a = ap.parse_args()
    if a.dfdc:
        items = dfdc_items(a.dfdc, a.dfdc_frames)
    else:
        items = [(*i, 1) for src in a.src for i in image_items(src, a.per_class, a.label, a.all_test)]
    rng = random.Random(7)

    mtcnn, saved, no_face, t0 = MTCNN(keep_all=True, device=DEVICE), 0, 0, time.time()
    every = 25 if a.dfdc else 1000
    print(f"{a.name}: {len(items)} {'videos' if a.dfdc else 'images'} to process", flush=True)
    for i, (path, label, split, n) in enumerate(items):
        stem = f"{path.parent.name}_{path.stem}" if not a.dfdc else path.stem    # same file names in two folders
        first = Path(a.out) / split / a.name / label / f"{stem}{'_0' if n > 1 else ''}.jpg"
        if not a.dfdc and first.exists():                   # cropped in an earlier, interrupted run
            saved += 1
            continue
        try:
            frames = video_frames(path, n) if a.dfdc else [recompress(open_image(path), rng, a.jpeg)]
        except Exception as e:                              # a broken file must not stop a long run
            print(f"  could not read {path.name}: {e}")
            continue
        for k, img in enumerate(frames):
            faces = faces_fast(mtcnn, img) if a.dfdc else find_faces(mtcnn, img)
            if not faces:
                no_face += 1
                continue
            out = Path(a.out) / split / a.name / label / f"{stem}{f'_{k}' if n > 1 else ''}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            crop_face(img, faces[0][0]).save(out, quality=95)
            saved += 1
        if i % every == 0:
            print(f"{i}/{len(items)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"{a.name}: saved {saved} face crops, skipped {no_face} images/frames with no face ({time.time() - t0:.0f}s)")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f".done_{a.name}").write_text(str(saved))      # lets the notebook skip finished datasets


if __name__ == "__main__":
    main()

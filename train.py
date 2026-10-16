"""Fine-tune the pre-trained ViT on our face crops.

  python train.py --data faces --out model --epochs 3 [--resume /kaggle/temp/ckpt.pt]
Labels come from the model's own config (label_index), so REAL/FAKE can never be swapped.
Saves the best epoch, its validation history, and the REAL/FAKE cut-off that works best on validation.

v3 changes (from what we saw in v1 and v2):
  - learning rate warms up for 5% of the steps, then follows a cosine curve down (v2 used one fixed rate and its
    last epoch got worse); gradients are clipped at 1.0
  - the best epoch is picked by balanced accuracy at its own cut-off, not plain accuracy (sets like E are fake-only)
  - sampling balances every (dataset, class) group, so a fake-only set cannot tilt the model towards FAKE
  - both Kaggle GPUs are used when there are two (about 1.7x faster)
  - after each epoch a checkpoint is saved; --resume carries on from it after a Kaggle restart
  - weights are saved in 16-bit (half the download, same results: they are loaded back as 32-bit)
"""
import argparse
import csv
import io
import json
import os
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoImageProcessor, AutoModelForImageClassification, get_cosine_schedule_with_warmup

from detector import BASE_MODEL, DEVICE, label_index


def files(data, split, model):
    idx = {"real": label_index(model, "real"), "fake": label_index(model, "fake")}
    return [(p, idx[p.parent.name]) for p in Path(data, split).glob("*/*/*.jpg")]


def degrade(im):
    """Random video-style damage (mirror, downscale, blur, JPEG) so the model can't rely on clean pixels."""
    if random.random() < 0.5:
        im = ImageOps.mirror(im)
    if random.random() < 0.3:
        s = random.uniform(0.35, 0.8)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s)))).resize(im.size)
    if random.random() < 0.2:
        im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.5)))
    if random.random() < 0.5:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=random.randint(30, 90))
        im = Image.open(buf).convert("RGB")
    return im


def loader(items, processor, train, bs):
    def collate(batch):
        imgs = [Image.open(p).convert("RGB") for p, _ in batch]
        x = processor(images=[degrade(im) for im in imgs] if train else imgs, return_tensors="pt")["pixel_values"]
        return x, torch.tensor([y for _, y in batch])
    sampler = None
    if train:   # small groups (C, E) get more weight, but not so much that a few videos are memorised
        group = lambda p: (p.parts[-3], p.parts[-2])            # (dataset, real|fake)
        size = Counter(group(p) for p, _ in items)
        sampler = WeightedRandomSampler([size[group(p)] ** -0.5 for p, _ in items], len(items))
    return DataLoader(items, batch_size=bs, sampler=sampler, collate_fn=collate,
                      num_workers=min(4, os.cpu_count() or 2), persistent_workers=True)


def balanced(p, y, c):
    real = [pi < c for pi, yi in zip(p, y) if not yi]
    fake = [pi >= c for pi, yi in zip(p, y) if yi]
    return (sum(real) / max(len(real), 1) + sum(fake) / max(len(fake), 1)) / 2


@torch.no_grad()
def scores(model, dl, fake):
    """P(FAKE) and true label (1 = fake) for every validation face."""
    model.eval()
    p, y = [], []
    for x, t in dl:
        p += model(pixel_values=x.to(DEVICE)).logits.softmax(-1)[:, fake].tolist()
        y += (t == fake).int().tolist()
    return p, y


def accuracy(p, y, cut=0.5):
    return sum((pi >= cut) == bool(yi) for pi, yi in zip(p, y)) / max(len(y), 1)


def best_cut(p, y):
    """The cut-off with the best balanced accuracy (mean of real-correct and fake-correct rates)."""
    return max((c / 100 for c in range(2, 99)), key=lambda c: balanced(p, y, c))


def save(model, processor, out):
    """16-bit weights: half the size; from_pretrained loads them back as 32-bit."""
    model = getattr(model, "module", model)
    model.save_pretrained(out, state_dict={k: v.half() for k, v in model.state_dict().items()})
    processor.save_pretrained(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="faces")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--out", default="model")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--resume", help="checkpoint file: saved after every epoch, picked up again if it exists")
    a = ap.parse_args()
    torch.manual_seed(42), random.seed(42)

    processor = AutoImageProcessor.from_pretrained(a.base)
    model = AutoModelForImageClassification.from_pretrained(a.base).to(DEVICE)
    train, val = files(a.data, "train", model), files(a.data, "val", model)
    print(f"train {len(train)} faces, val {len(val)} faces, labels {model.config.id2label}")
    print("train faces per group:", dict(sorted(Counter((p.parts[-3], p.parts[-2]) for p, _ in train).items())))
    train_dl, val_dl = loader(train, processor, True, a.bs), loader(val, processor, False, a.bs)
    val_set = [p.parts[-3] for p, _ in val]

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(a.warmup * a.epochs * len(train_dl)), a.epochs * len(train_dl))
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    fake = label_index(model, "fake")
    start, best, history, before = 1, -1.0, [], None
    if a.resume and Path(a.resume).exists():
        ck = torch.load(a.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, best, history, before = ck["epoch"] + 1, ck["best"], ck["history"], ck["before"]
        print(f"resumed after epoch {ck['epoch']} (best balanced accuracy so far {best:.4f})")
    net = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    if torch.cuda.device_count() > 1:
        print(f"using {torch.cuda.device_count()} GPUs")
    if before is None:
        before = accuracy(*scores(net, val_dl, fake))
        print(f"before training: val accuracy {before:.4f}")
    for epoch in range(start, a.epochs + 1):
        net.train()
        for step, (x, y) in enumerate(train_dl):
            with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                logits = net(pixel_values=x.to(DEVICE)).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), y.to(DEVICE))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(train_dl)} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e}",
                      flush=True)
        p, y = scores(net, val_dl, fake)
        cut = best_cut(p, y)
        row = {"epoch": epoch, "val_accuracy": accuracy(p, y), "val_balanced": balanced(p, y, cut), "cut": cut}
        for name in sorted(set(val_set)):                # per test set, at this epoch's cut-off
            pick = [i for i, s in enumerate(val_set) if s == name]
            row[f"val_{name}"] = accuracy([p[i] for i in pick], [y[i] for i in pick], cut)
        history.append(row)
        print(f"epoch {epoch}: " + ", ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "epoch"), flush=True)
        if row["val_balanced"] >= best:                   # keep the best epoch only
            best = row["val_balanced"]
            save(model, processor, a.out)
            json.dump({"threshold": cut, "val_accuracy_at_threshold": accuracy(p, y, cut),
                       "val_balanced_accuracy": best, "epoch": epoch}, open(Path(a.out, "threshold.json"), "w"))
        if a.resume:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "history": history,
                        "before": before}, a.resume)
    keys = list(dict.fromkeys(k for r in history for k in r))
    with open(Path(a.out, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader(), w.writerows(history)
    t = json.load(open(Path(a.out, "threshold.json")))
    print(f"val accuracy {before:.4f} before training; best epoch {t['epoch']}: balanced accuracy {best:.4f}, "
          f"cut-off {t['threshold']:.2f}. Saved to {a.out}/")


if __name__ == "__main__":
    main()

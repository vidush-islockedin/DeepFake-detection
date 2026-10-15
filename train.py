"""Fine-tune the pre-trained ViT on our face crops.

  python train.py --data faces --out model --epochs 3
Labels come from the model's own config (label_index), so REAL/FAKE can never be swapped.
Saves the best epoch, its validation history, and the REAL/FAKE cut-off that works best on validation.
"""
import argparse
import csv
import io
import json
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoImageProcessor, AutoModelForImageClassification

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
    if train:   # small datasets (C) get more weight, but not so much that a few videos are memorised
        size = Counter(p.parts[-3] for p, _ in items)
        sampler = WeightedRandomSampler([size[p.parts[-3]] ** -0.5 for p, _ in items], len(items))
    return DataLoader(items, batch_size=bs, sampler=sampler, collate_fn=collate, num_workers=2)


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
    def balanced(c):
        real = [pi < c for pi, yi in zip(p, y) if not yi]
        fake = [pi >= c for pi, yi in zip(p, y) if yi]
        return (sum(real) / max(len(real), 1) + sum(fake) / max(len(fake), 1)) / 2
    return max((c / 100 for c in range(2, 99)), key=balanced)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="faces")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--out", default="model")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    a = ap.parse_args()
    torch.manual_seed(42), random.seed(42)

    processor = AutoImageProcessor.from_pretrained(a.base)
    model = AutoModelForImageClassification.from_pretrained(a.base).to(DEVICE)
    train, val = files(a.data, "train", model), files(a.data, "val", model)
    print(f"train {len(train)} faces, val {len(val)} faces, labels {model.config.id2label}")
    train_dl, val_dl = loader(train, processor, True, a.bs), loader(val, processor, False, a.bs)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    fake = label_index(model, "fake")
    before, best, history = accuracy(*scores(model, val_dl, fake)), -1.0, []
    print(f"before training: val accuracy {before:.4f}")
    for epoch in range(1, a.epochs + 1):
        model.train()
        for step, (x, y) in enumerate(train_dl):
            with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                loss = model(pixel_values=x.to(DEVICE), labels=y.to(DEVICE)).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(train_dl)} loss {loss.item():.4f}")
        p, y = scores(model, val_dl, fake)
        acc = accuracy(p, y)
        history.append({"epoch": epoch, "val_accuracy": acc})
        print(f"epoch {epoch}: val accuracy {acc:.4f}")
        if acc >= best:   # keep the best epoch only
            best, best_p = acc, p
            model.save_pretrained(a.out)
            processor.save_pretrained(a.out)
    cut = best_cut(best_p, y)
    json.dump({"threshold": cut, "val_accuracy_at_threshold": accuracy(best_p, y, cut)}, open(Path(a.out, "threshold.json"), "w"))
    with open(Path(a.out, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["epoch", "val_accuracy"])
        w.writeheader(), w.writerows(history)
    print(f"val accuracy: {before:.4f} before -> {best:.4f} after training; cut-off {cut:.2f}. Saved to {a.out}/")


if __name__ == "__main__":
    main()

"""Fine-tune the pre-trained ViT on our face crops.

  python train.py --data faces --out model --epochs 2
Labels come from the model's own config (label_index), so REAL/FAKE can never be swapped.
"""
import argparse
import csv
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification

from detector import BASE_MODEL, DEVICE, label_index


def files(data, split, model):
    idx = {"real": label_index(model, "real"), "fake": label_index(model, "fake")}
    return [(p, idx[p.parent.name]) for p in Path(data, split).glob("*/*/*.jpg")]


def loader(items, processor, train, bs):
    def collate(batch):
        imgs = [Image.open(p).convert("RGB") for p, _ in batch]
        if train:   # light augmentation: random mirror
            imgs = [ImageOps.mirror(im) if random.random() < 0.5 else im for im in imgs]
        x = processor(images=imgs, return_tensors="pt")["pixel_values"]
        return x, torch.tensor([y for _, y in batch])
    return DataLoader(items, batch_size=bs, shuffle=train, collate_fn=collate, num_workers=2)


@torch.no_grad()
def accuracy(model, dl):
    model.eval()
    right = total = 0
    for x, y in dl:
        right += (model(pixel_values=x.to(DEVICE)).logits.argmax(-1).cpu() == y).sum().item()
        total += len(y)
    return right / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="faces")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--out", default="model")
    ap.add_argument("--epochs", type=int, default=2)
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
    before, best, history = accuracy(model, val_dl), -1.0, []
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
        acc = accuracy(model, val_dl)
        history.append({"epoch": epoch, "val_accuracy": acc})
        print(f"epoch {epoch}: val accuracy {acc:.4f}")
        if acc >= best:   # keep the best epoch only
            best = acc
            model.save_pretrained(a.out)
            processor.save_pretrained(a.out)
    with open(Path(a.out, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["epoch", "val_accuracy"])
        w.writeheader(), w.writerows(history)
    print(f"val accuracy: {before:.4f} before -> {best:.4f} after training. Saved to {a.out}/")


if __name__ == "__main__":
    main()

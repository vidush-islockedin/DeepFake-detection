"""Score a model on every test set:  python evaluate.py --model model --out results/after

Writes results.json (accuracy, precision, recall, F1, AUC, confusion matrix) and one confusion-matrix PNG per test set.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from detector import BASE_MODEL, Detector


def score(det, folder, bs=64):
    paths = sorted(Path(folder).glob("*/*.jpg"))
    y = [int(p.parent.name == "fake") for p in paths]
    p = sum((det.p_fake([Image.open(q).convert("RGB") for q in paths[i:i + bs]]) for i in range(0, len(paths), bs)), [])
    pred = [int(v >= 0.5) for v in p]
    cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()   # rows: actual REAL, FAKE; cols: predicted
    return {"images": len(y), "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "auc": roc_auc_score(y, p) if len(set(y)) == 2 else None, "confusion_matrix": cm,
            "real_called_fake": cm[0][1], "fake_called_real": cm[1][0]}


def plot(cm, title, path):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["REAL", "FAKE"]), ax.set_yticks([0, 1], ["REAL", "FAKE"])
    ax.set_xlabel("Predicted"), ax.set_ylabel("Actual"), ax.set_title(title, fontsize=10)
    fig.tight_layout(), fig.savefig(path, dpi=160), plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--data", default="faces/test")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    det, out = Detector(a.model), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {"model": a.model}
    for folder in sorted(Path(a.data).iterdir()):
        r = results[folder.name] = score(det, folder)
        plot(r["confusion_matrix"], f"Test set {folder.name}", out / f"confusion_{folder.name}.png")
        print(f"{folder.name}: {r['images']} faces | accuracy {r['accuracy']:.4f} | F1 {r['f1']:.4f} | AUC {r['auc']}")
    json.dump(results, open(out / "results.json", "w"), indent=2)


if __name__ == "__main__":
    main()

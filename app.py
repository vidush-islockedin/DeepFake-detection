"""Run:  python app.py   then open http://localhost:8000"""
import csv
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from detector import MARGIN, Detector, open_image

HERE = Path(__file__).parent
SAMPLES = HERE / "static" / "samples"
app = FastAPI(title="Deepfake Detection of Images")
detector = Detector()
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

DATASETS = {
    "A": {"name": "Deepfake and Real Images", "kind": "Generated faces",
          "about": "Real photos and generated deepfake faces, already split into train, validation and test.",
          "url": "https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images"},
    "B": {"name": "140k Real and Fake Faces", "kind": "StyleGAN faces",
          "about": "70,000 real Flickr faces (FFHQ) and 70,000 faces made by StyleGAN.",
          "url": "https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces"},
    "C": {"name": "DeepFake Detection Challenge (DFDC)", "kind": "Face swaps from video",
          "about": "Face-swapped videos. Frames from 60% of the people train the model; test faces come from people it never saw.",
          "url": "https://www.kaggle.com/competitions/deepfake-detection-challenge"},
    "D": {"name": "Human Faces Dataset", "kind": "Modern AI faces",
          "about": "About 5,000 real and 5,000 AI-generated face photos from newer image generators.",
          "url": "https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset"},
    "E": {"name": "Gemini (Nano Banana) images", "kind": "Gemini-made faces",
          "about": "Images made by Google's Gemini image model; only those with a face are used. All are fake.",
          "url": "https://huggingface.co/datasets/bitmind/nano-banana"},
    "G": {"name": "Our Gemini test photos", "kind": "Gemini faces we made",
          "about": "Faces we generated ourselves in the Gemini app. Used only for testing, never for training.",
          "url": None},
}


@app.get("/")
def home():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/info")
def info():
    m = detector.model
    cfg = m.config
    return {
        "model": detector.source, "fine_tuned": not detector.source.startswith("dima806"),
        "labels": cfg.id2label, "device": "GPU" if str(m.device).startswith("cuda") else "CPU",
        "threshold": detector.threshold, "version": latest_run().name if latest_run() else None,
        "vit": {"params": sum(p.numel() for p in m.parameters()), "layers": cfg.num_hidden_layers,
                "heads": cfg.num_attention_heads, "hidden": cfg.hidden_size, "patch": cfg.patch_size,
                "image": cfg.image_size, "tokens": (cfg.image_size // cfg.patch_size) ** 2},
        "mtcnn": {"stages": ["P-Net", "R-Net", "O-Net"], "min_confidence": 0.9, "landmarks": 5,
                  "params": sum(p.numel() for p in detector.mtcnn.parameters()), "crop_margin": MARGIN},
    }


def latest_run():
    """The results that belong to the model in use (reports/v3 for the v3 model), else the newest run."""
    own = HERE / "reports" / str(detector.version)
    if detector.version and (own / "after" / "results.json").exists():
        return own
    runs = sorted(HERE.glob("reports/v*"), key=lambda d: int(d.name[1:]) if d.name[1:].isdigit() else 0)
    return runs[-1] if runs else None


def class_report(m):
    """Per-class precision / recall / F1 / support from the confusion matrix [[TN, FP], [FN, TP]] (FAKE = positive)."""
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    f1 = lambda p, r: 2 * p * r / (p + r) if p + r else 0.0
    real_p, real_r = tn / (tn + fn) if tn + fn else 0.0, tn / (tn + fp) if tn + fp else 0.0
    fake_p, fake_r = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
    rows = {"REAL": {"precision": real_p, "recall": real_r, "f1": f1(real_p, real_r), "support": tn + fp},
            "FAKE": {"precision": fake_p, "recall": fake_r, "f1": f1(fake_p, fake_r), "support": fn + tp}}
    rows["macro"] = {k: (rows["REAL"][k] + rows["FAKE"][k]) / 2 for k in ("precision", "recall", "f1")}
    rows["macro"]["support"] = tn + fp + fn + tp
    present = [r for r, n in ((real_r, tn + fp), (fake_r, fn + tp)) if n]      # a fake-only test set has no REAL row
    return {"classes": rows, "balanced_accuracy": sum(present) / len(present) if present else 0.0}


@app.get("/api/metrics")
def metrics():
    """Test results of the newest training run (reports/v<N>/), measured on Kaggle."""
    d = latest_run()
    if not d:
        return None
    after = json.load(open(d / "after" / "results.json"))
    sets = {k: {**v, **class_report(v), **DATASETS.get(k, {"name": k})} for k, v in after.items() if isinstance(v, dict)}
    return {"version": d.name, "threshold": after.get("threshold", 0.5), "sets": sets,
            "history": [{"epoch": int(r["epoch"]), "val_accuracy": float(r["val_accuracy"])}
                        for r in csv.DictReader(open(d / "history.csv"))]}


@app.get("/api/samples")
def samples():
    """Photos put in static/samples/ show up as one-click examples on the page."""
    ok = {".jpg", ".jpeg", ".png", ".webp"}
    return [f"/static/samples/{p.name}" for p in sorted(SAMPLES.glob("*")) if p.suffix.lower() in ok] if SAMPLES.exists() else []


@app.post("/api/predict")
async def predict(file: UploadFile):
    try:
        img = open_image(await file.read())
    except Exception:
        raise HTTPException(400, "That file is not an image.")
    return detector.predict(img)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

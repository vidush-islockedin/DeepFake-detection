"""Run:  python app.py   then open http://localhost:8000"""
import csv
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from detector import Detector, open_image

HERE = Path(__file__).parent
app = FastAPI(title="MT-ViT Deepfake Detector")
detector = Detector()
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.get("/")
def home():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/info")
def info():
    return {"model": detector.source, "labels": detector.model.config.id2label, "device": str(detector.model.device)}


@app.get("/api/metrics")
def metrics():
    """Test results measured on Kaggle (reports/<version>/), newest version first."""
    runs = []
    for d in sorted(HERE.glob("reports/v*"), reverse=True):
        load = lambda name: json.load(open(d / name / "results.json"))
        runs.append({"version": d.name, "before": load("before"), "after": load("after"),
                     "history": list(csv.DictReader(open(d / "history.csv")))})
    return runs


@app.post("/api/predict")
async def predict(file: UploadFile):
    try:
        img = open_image(await file.read())
    except Exception:
        raise HTTPException(400, "That file is not an image.")
    return detector.predict(img)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

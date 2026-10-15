"""Run:  python app.py   then open http://localhost:8000"""
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse

from detector import Detector, open_image

HERE = Path(__file__).parent
app = FastAPI(title="MT-ViT Deepfake Detector")
detector = Detector()


@app.get("/")
def home():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/info")
def info():
    return {"model": detector.source, "labels": detector.model.config.id2label, "device": str(detector.model.device)}


@app.post("/api/predict")
async def predict(file: UploadFile):
    try:
        img = open_image(await file.read())
    except Exception:
        raise HTTPException(400, "That file is not an image.")
    return detector.predict(img)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

"""MT-ViT detector: MTCNN finds the face, a Vision Transformer decides REAL or FAKE."""
import json
from pathlib import Path

import torch
from facenet_pytorch import MTCNN
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification

BASE_MODEL = "dima806/deepfake_vs_real_image_detection"   # pre-trained ViT-B/16 (Apache-2.0)
LOCAL_MODEL = Path(__file__).parent / "model"              # our fine-tuned copy, if it exists
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MARGIN = 1.3                                               # crop 1.3x the face box (keeps the blend edges)


def open_image(data):
    """Bytes or path -> upright RGB image."""
    import io
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data) if isinstance(data, bytes) else data)).convert("RGB")


def find_faces(mtcnn, img, min_conf=0.9):
    """All faces as (box, confidence, 5 landmarks: eyes, nose, mouth corners), largest first."""
    boxes, probs, points = mtcnn.detect(img, landmarks=True)
    if boxes is None:
        return []
    faces = [(b.tolist(), float(p), l.tolist()) for b, p, l in zip(boxes, probs, points) if p >= min_conf]
    return sorted(faces, key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]), reverse=True)


def square(img, box, margin=MARGIN):
    """(left, top, side) of a square around the face, kept inside the image."""
    x1, y1, x2, y2 = box
    side = int(min(max(x2 - x1, y2 - y1) * margin, img.width, img.height))
    left = int(min(max((x1 + x2 - side) / 2, 0), img.width - side))
    top = int(min(max((y1 + y2 - side) / 2, 0), img.height - side))
    return left, top, side


def crop_face(img, box, margin=MARGIN):
    left, top, side = square(img, box, margin)
    return img.crop((left, top, left + side, top + side))


def label_index(model, name):
    """Which output is FAKE / REAL, read from the model's own config (never guessed, never alphabetical)."""
    words = {"fake": ("fake", "deepfake"), "real": ("real", "realism")}[name]
    return next(i for i, label in model.config.id2label.items() if label.lower() in words)


class Detector:
    def __init__(self, model_path=None):
        self.source = str(model_path or (LOCAL_MODEL if LOCAL_MODEL.exists() else BASE_MODEL))
        self.processor = AutoImageProcessor.from_pretrained(self.source)
        self.model = AutoModelForImageClassification.from_pretrained(self.source).to(DEVICE).eval().requires_grad_(False)
        self.fake, self.real = label_index(self.model, "fake"), label_index(self.model, "real")
        cut = Path(self.source, "threshold.json")    # written by train.py; the original model uses 0.5
        self.threshold = json.load(open(cut))["threshold"] if cut.exists() else 0.5
        self.mtcnn = MTCNN(keep_all=True, device=DEVICE)

    @torch.no_grad()
    def p_fake(self, faces):
        """List of face images -> list of P(FAKE)."""
        x = self.processor(images=faces, return_tensors="pt")["pixel_values"].to(DEVICE)
        return self.model(pixel_values=x).logits.softmax(-1)[:, self.fake].tolist()

    def predict(self, img, explain=True):
        from explain import explain_face, summary
        faces = find_faces(self.mtcnn, img)
        if not faces:
            return {"status": "no_face", "faces": [], "message": "No face found, so no verdict is given."}
        scores = self.p_fake([crop_face(img, box) for box, *_ in faces])
        top = max(range(len(faces)), key=lambda i: scores[i])   # the image is FAKE if any face is
        p, verdict = scores[top], lambda s: "FAKE" if s >= self.threshold else "REAL"
        result = {
            "status": "ok", "prediction": verdict(p), "threshold": self.threshold,
            "p_fake": p, "p_real": 1 - p, "confidence": p if p >= self.threshold else 1 - p,
            "faces": [{"box": [round(v) for v in box], "face_confidence": c, "p_fake": s, "prediction": verdict(s),
                       **(explain_face(self, img, box, pts, verdict(s) == "FAKE") if explain else {})}
                      for (box, c, pts), s in zip(faces, scores)],
            "decisive_face": top, "image_size": [img.width, img.height],
            "model": self.source, "labels": self.model.config.id2label,
        }
        result["summary"] = summary(result) if explain else ""
        return result

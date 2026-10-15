"""Quick checks that need no internet: a tiny random ViT stands in for the real model."""
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detector import Detector, crop_face, label_index  # noqa: E402
from prepare import label_and_split  # noqa: E402


def tiny_model(path, id2label):
    cfg = ViTConfig(image_size=224, patch_size=32, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                    intermediate_size=64, num_labels=2, id2label=id2label, label2id={v: k for k, v in id2label.items()})
    ViTForImageClassification(cfg).save_pretrained(path)
    ViTImageProcessor().save_pretrained(path)
    return path


@pytest.mark.parametrize("labels,fake,real", [({0: "Fake", 1: "Real"}, 0, 1), ({0: "Real", 1: "Fake"}, 1, 0),
                                              ({0: "Realism", 1: "Deepfake"}, 1, 0)])
def test_label_index_reads_the_model_not_alphabet(tmp_path, labels, fake, real):
    m = ViTForImageClassification.from_pretrained(tiny_model(tmp_path / "m", labels))
    assert (label_index(m, "fake"), label_index(m, "real")) == (fake, real)


def test_crop_is_square_and_inside():
    c = crop_face(Image.new("RGB", (300, 200)), [250, 150, 300, 200])
    assert c.width == c.height and c.width <= 200


def test_label_and_split_for_kaggle_layouts():
    assert label_and_split(Path("Dataset/Validation/Fake/1.jpg")) == ("fake", "val")
    assert label_and_split(Path("real_vs_fake/real-vs-fake/train/real/a.jpg")) == ("real", "train")


def test_end_to_end(tmp_path):
    skimage = pytest.importorskip("skimage.data")
    face = Image.fromarray(skimage.astronaut())
    model = tiny_model(tmp_path / "model", {0: "Fake", 1: "Real"})
    for split in ("train", "val", "test"):
        for label in ("real", "fake"):
            d = tmp_path / "faces" / split / "X" / label
            d.mkdir(parents=True)
            for i in range(2):
                face.resize((160, 160)).save(d / f"{i}.jpg")
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, check=True, capture_output=True, text=True)
    run("train.py", "--data", str(tmp_path / "faces"), "--base", str(model), "--out", str(tmp_path / "ft"), "--epochs", "1", "--bs", "4")
    run("evaluate.py", "--model", str(tmp_path / "ft"), "--data", str(tmp_path / "faces/test"), "--out", str(tmp_path / "res"))
    assert json.load(open(tmp_path / "res/results.json"))["X"]["images"] == 4

    r = Detector(tmp_path / "ft").predict(face)
    assert r["status"] == "ok" and len(r["faces"]) == 1 and r["prediction"] in ("REAL", "FAKE")
    assert abs(r["p_fake"] + r["p_real"] - 1) < 1e-6
    assert Detector(tmp_path / "ft").predict(Image.new("RGB", (300, 300), "gray"))["status"] == "no_face"


def test_api(tmp_path, monkeypatch):
    skimage = pytest.importorskip("skimage.data")
    import detector
    monkeypatch.setattr(detector, "LOCAL_MODEL", tiny_model(tmp_path / "model", {0: "Real", 1: "Fake"}))
    from fastapi.testclient import TestClient
    import app
    c = TestClient(app.app)
    buf = io.BytesIO()
    Image.fromarray(skimage.astronaut()).save(buf, format="JPEG")
    assert c.post("/api/predict", files={"file": ("a.jpg", buf.getvalue())}).json()["status"] == "ok"
    assert c.post("/api/predict", files={"file": ("a.txt", b"hello")}).status_code == 400
    assert c.get("/").status_code == 200

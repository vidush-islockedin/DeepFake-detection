"""Why REAL or FAKE? A Grad-CAM heatmap over the face, how much of that evidence sits in each face region,
and a plain-English summary built only from those numbers.

The heatmap shows where the model found the evidence for its verdict. It is not a pixel-exact map of edits.
"""
import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image

from detector import square

SIZE = 224
NAMES = {"eyes": "eyes", "nose": "nose", "mouth": "mouth", "skin": "cheeks, forehead and jaw",
         "surround": "hair and edges around the face"}


def gradcam(det, crop, target, other):
    """SIZE x SIZE map (0..1) of how strongly each part of the crop pushed the score towards `target`."""
    x = det.processor(images=[crop], return_tensors="pt")["pixel_values"].to(det.model.device).requires_grad_(True)
    with torch.enable_grad():
        out = det.model(pixel_values=x, output_hidden_states=True)
        tokens = out.hidden_states[-2]      # input to the last block: its patch tokens still feed the CLS token
        grad, = torch.autograd.grad(out.logits[0, target] - out.logits[0, other], tokens)
    cam = torch.relu((tokens[0, 1:] * grad[0, 1:].mean(0)).sum(-1)).detach()
    n = int(cam.numel() ** 0.5)
    cam = F.interpolate(cam.reshape(1, 1, n, n), size=(SIZE, SIZE), mode="bilinear")[0, 0].cpu().numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def regions(box, points, left, top, side):
    """Boolean masks (SIZE x SIZE) for eyes, nose, mouth, rest of the face and the surround, from MTCNN landmarks."""
    k = SIZE / side
    (ex1, ey1), (ex2, ey2), (nx, ny), (mx1, my1), (mx2, my2) = [((x - left) * k, (y - top) * k) for x, y in points]
    x1, y1, x2, y2 = [(v - o) * k for v, o in zip(box, (left, top, left, top))]
    w = x2 - x1
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    near = lambda cx, cy, rx, ry: ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1
    eyes = near(ex1, ey1, .16 * w, .09 * w) | near(ex2, ey2, .16 * w, .09 * w)
    nose = near(nx, ny - .04 * w, .1 * w, .15 * w) & ~eyes
    mouth = near((mx1 + mx2) / 2, (my1 + my2) / 2, abs(mx2 - mx1) / 2 + .08 * w, .09 * w) & ~eyes & ~nose
    skin = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2) & ~eyes & ~nose & ~mouth
    return {"eyes": eyes, "nose": nose, "mouth": mouth, "skin": skin, "surround": ~(eyes | nose | mouth | skin)}


def data_url(img, fmt):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def explain_face(det, img, box, points, is_fake):
    left, top, side = square(img, box)
    crop = img.crop((left, top, left + side, top + side))
    target, other = (det.fake, det.real) if is_fake else (det.real, det.fake)
    cam = gradcam(det, crop, target, other)
    heat = colormaps["inferno"](cam)
    heat[..., 3] = cam ** 0.8 * 0.9                      # transparent where there is no evidence
    total = cam.sum() or 1.0
    parts = [{"region": r, "name": NAMES[r], "share": float(cam[m].sum() / total), "area": float(m.mean())}
             for r, m in regions(box, points, left, top, side).items()]
    return {"crop": data_url(crop.resize((SIZE, SIZE)), "JPEG"),
            "heatmap": data_url(Image.fromarray((heat * 255).astype(np.uint8), "RGBA"), "PNG"),
            "regions": sorted(parts, key=lambda p: -p["share"])}


def summary(r):
    """Plain-English reason, using only the measured numbers."""
    face = r["faces"][r["decisive_face"]]
    a, b = face["regions"][:2]
    fake = r["prediction"] == "FAKE"
    text = (f"The model is {r['confidence']:.0%} sure this face is {'a deepfake' if fake else 'real'}. "
            f"{'Most of the deepfake evidence' if fake else 'Most of the evidence that it is real'} came from the "
            f"{a['name']} ({a['share']:.0%}) and the {b['name']} ({b['share']:.0%}).")
    if a["share"] > 1.5 * a["area"]:
        text += f" The {a['name']} region stands out: {a['share']:.0%} of the evidence from only {a['area']:.0%} of the picture."
    if abs(r["p_fake"] - r["threshold"]) < 0.15:
        text += " The score is close to the cut-off, so treat this result as uncertain."
    if len(r["faces"]) > 1:
        text += (f" {len(r['faces'])} faces were found; face {r['decisive_face'] + 1} decided the verdict, "
                 "because a photo counts as a deepfake if any face in it looks fake.")
    return text

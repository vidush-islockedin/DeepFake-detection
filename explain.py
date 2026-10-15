"""Why REAL or FAKE? Two measurements on every face, and a plain-English summary built only from their numbers.

1. Blur test (occlusion). Blur one part of the face, score it again, and see how the deepfake score moves.
   Done for each face region (eyes, nose, mouth, skin, surround) and, for the face that decided the verdict,
   for a 6 x 6 grid of squares (other faces get a map painted from their five regions).
   If blurring a part lowers the score, that part looked FAKE (red on the map); if blurring it raises
   the score, that part looked REAL (blue). This is causal: it is what the model does when the detail is gone.
2. Grad-CAM focus. The gradient of the verdict's logit (minus the other logit) on the patch tokens entering
   the last transformer block shows where the model's attention for its verdict sat.

Both maps show where the evidence was, not a pixel-exact map of edits.
"""
import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from detector import square

SIZE = 224
GRID = 6                                   # 6 x 6 squares of ~37 px for the map of the deciding face
BLUR = 8                                   # Gaussian radius (px at 224) used to remove fine detail
NAMES = {"eyes": "Eyes", "nose": "Nose", "mouth": "Mouth", "skin": "Cheeks, forehead and jaw",
         "surround": "Hair, ears and background"}
PLURAL = {"eyes", "skin", "surround"}
FAKE_RGB, REAL_RGB, FOCUS_RGB = (214, 46, 69), (43, 134, 208), (245, 197, 24)


def log_odds(det, images, batch=16):
    """FAKE logit minus REAL logit for each image (P(FAKE) = sigmoid of this)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = det.processor(images=images[i:i + batch], return_tensors="pt")["pixel_values"].to(det.model.device)
            logits = det.model(pixel_values=x).logits
            out.append((logits[:, det.fake] - logits[:, det.real]).float().cpu().numpy())
    return np.concatenate(out)


def sigmoid(v):
    return float(1 / (1 + np.exp(-v)))


def gradcam(det, crop, target, other):
    """SIZE x SIZE map (0..1) of where the evidence for `target` sat."""
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
    """Boolean masks (SIZE x SIZE) for eyes, nose, mouth, rest of the face and the surround, from MTCNN landmarks,
    plus a point inside each region for its numbered tag."""
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
    masks = {"eyes": eyes, "nose": nose, "mouth": mouth, "skin": skin, "surround": ~(eyes | nose | mouth | skin)}
    clamp = lambda v: float(min(max(v, 14), SIZE - 14))
    anchors = {"eyes": (ex1, ey1), "nose": (nx, ny), "mouth": ((mx1 + mx2) / 2, (my1 + my2) / 2),
               "skin": (x1 + .16 * w, (ey1 + my1) / 2 + .1 * w), "surround": (x2 + (SIZE - x2) / 2, max(y1, 30))}
    return masks, {r: [clamp(x), clamp(y)] for r, (x, y) in anchors.items()}


def data_url(img, fmt):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def two_colour(effect):
    """RGBA: red where blurring lowered the deepfake score (looked fake), blue where it raised it (looked real)."""
    a = np.abs(effect)
    rgb = np.where(effect[..., None] > 0, np.array(FAKE_RGB), np.array(REAL_RGB))
    alpha = np.where(a < .1, 0, np.clip(a, 0, 1) ** .75 * .82)
    return Image.fromarray(np.dstack([rgb, alpha * 255]).astype(np.uint8), "RGBA")


def one_colour(cam):
    alpha = np.where(cam < .08, 0, cam ** .8 * .85)
    rgb = np.broadcast_to(np.array(FOCUS_RGB), cam.shape + (3,))
    return Image.fromarray(np.dstack([rgb, alpha * 255]).astype(np.uint8), "RGBA")


def explain_face(det, img, box, points, is_fake, detail=True):
    left, top, side = square(img, box)
    crop = img.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.BILINEAR)   # as the processor does
    sharp, soft = np.asarray(crop), np.asarray(crop.filter(ImageFilter.GaussianBlur(BLUR)))
    swap = lambda m: Image.fromarray(np.where(m[..., None], soft, sharp))
    masks, anchors = regions(box, points, left, top, side)

    edges = np.linspace(0, SIZE, GRID + 1).round().astype(int)
    squares = []
    for i in range(GRID if detail else 0):
        for j in range(GRID):
            m = np.zeros((SIZE, SIZE), bool)
            m[edges[i]:edges[i + 1], edges[j]:edges[j + 1]] = True
            squares.append(m)
    scores = log_odds(det, [crop] + [swap(m) for m in masks.values()] + [swap(m) for m in squares])
    full, by_region, by_square = scores[0], scores[1:1 + len(masks)], scores[1 + len(masks):]
    region_effect = {r: float(full - s) for r, s in zip(masks, by_region)}

    if detail:                                                    # > 0: that square made the face look fake
        grid = torch.tensor((full - by_square).reshape(1, 1, GRID, GRID), dtype=torch.float32)
        effect = F.interpolate(grid, size=(SIZE, SIZE), mode="bicubic", align_corners=False)[0, 0].numpy()
        pos, neg = float(np.clip(grid, 0, None).sum()), float(np.clip(-grid, 0, None).sum())
    else:                                                         # paint each region with its own effect
        effect = sum(m * region_effect[r] for r, m in masks.items()).astype(np.float32)
        pos = sum(max(e, 0) * masks[r].mean() for r, e in region_effect.items())
        neg = sum(max(-e, 0) * masks[r].mean() for r, e in region_effect.items())
    peak = np.abs(effect).max()
    effect = effect / peak if peak > 0 else effect
    focus = gradcam(det, crop, *((det.fake, det.real) if is_fake else (det.real, det.fake)))

    total = sum(abs(e) for e in region_effect.values()) or 1.0
    parts = []
    for (r, m), s in zip(masks.items(), by_region):
        e = region_effect[r]
        parts.append({"region": r, "name": NAMES[r], "area": float(m.mean()),
                      "effect": e, "weight": abs(e) / total, "lean": "FAKE" if e > 0 else "REAL",
                      "p_without": sigmoid(s), "focus_share": float(focus[m].sum() / (focus.sum() or 1.0)),
                      "anchor": anchors[r]})
    sign = 1 if is_fake else -1                       # sort: strongest support for the verdict first
    parts.sort(key=lambda p: -sign * p["effect"])
    return {"crop": data_url(crop, "JPEG"),
            "heatmap": data_url(two_colour(effect), "PNG"),
            "focus": data_url(one_colour(focus), "PNG"),
            "map_detail": "grid" if detail else "regions",
            "tests": len(scores) - 1,
            "square": [left, top, side],
            "balance": pos / ((pos + neg) or 1.0),      # share of the blur-test evidence pointing to FAKE
            "p_full": sigmoid(full),
            "regions": parts}


def summary(r):
    """Plain-English reason, using only the measured numbers."""
    face = r["faces"][r["decisive_face"]]
    fake = r["prediction"] == "FAKE"
    lead = face["regions"][0]
    text = (f"Deepfake score {r['p_fake']:.0%}, {'at or above' if fake else 'below'} the {r['threshold']:.0%} cut-off, "
            f"so the face is called {'a deepfake' if fake else 'real'}. ")
    verb = "stand" if lead["region"] in PLURAL else "stands"
    supports = (lead["effect"] > 0) == fake
    if supports and abs(lead["effect"]) > 0.3:
        text += (f"The {lead['name'].lower()} {verb} out: blurring {'them' if lead['region'] in PLURAL else 'it'} "
                 f"moves the deepfake score from {face['p_full']:.0%} to {lead['p_without']:.0%}, "
                 f"so {'that is where the strongest deepfake evidence sits' if fake else ('their' if lead['region'] in PLURAL else 'its') + ' detail is the strongest sign the face is real'}.")
    else:
        text += "No single part of the face carries the verdict on its own; the evidence is spread across the face."
    against = [p for p in face["regions"] if (p["effect"] > 0) != fake and abs(p["effect"]) > 0.3]
    if against:
        p = max(against, key=lambda q: abs(q["effect"]))
        text += f" The {p['name'].lower()} point{'' if p['region'] in PLURAL else 's'} the other way, but less strongly."
    if abs(r["p_fake"] - r["threshold"]) < 0.15:
        text += " The score is close to the cut-off, so treat this result as uncertain."
    if len(r["faces"]) > 1:
        text += (f" {len(r['faces'])} faces were found; face {r['decisive_face'] + 1} decided the verdict, "
                 "because a photo counts as a deepfake if any face in it looks fake.")
    return text

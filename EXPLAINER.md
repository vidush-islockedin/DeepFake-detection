# EXPLAINER.md: how the code works, file by file

A guided tour of the project for someone seeing the code for the first time, such as a teacher or examiner.
**Why** each choice was made is in [DECISIONS.md](DECISIONS.md); this file explains **what the code does and where**.

Team: Abdulla Mahir, Vidush Karnani, Ishaan Kulkarni. AI Mini Project: *Deepfake Detection of Images*.

---

## 1. What the project does

You give it a photo. It finds every face, says whether each face is **real** or a **deepfake** with a probability,
and shows **why**: which parts of the face looked fake and which looked real.

```
photo
  │  detector.py   MTCNN finds faces + 5 landmarks (eyes, nose, mouth corners)
  ▼
square crop 1.3× the face box, resized to 224 × 224
  │  detector.py   ViT-B/16 (fine-tuned by us) scores the crop
  ▼
P(fake) per face ──► FAKE if P(fake) ≥ 0.65; the photo is FAKE if any face is
  │  explain.py    blur test + Grad-CAM on the face that decided the verdict
  ▼
web page: verdict, red/green map, a reason for each face part, results of every test set
```

Two neural networks:
- **MTCNN** is three small CNNs in a row (P-Net, R-Net, O-Net). We use it as published; we did not train it.
- **ViT-B/16** is a Vision Transformer with 12 layers and about 86 million parameters. It started from public
  deepfake weights (`dima806/deepfake_vs_real_image_detection`), and **we fine-tuned it** on Kaggle.

---

## 2. Where everything is saved

| What | Where |
|---|---|
| All the code (current version) | GitHub: **https://github.com/vidush-islockedin/DeepFake-detection**, branch `main` |
| The v2 code, kept as a savepoint | Same repo, branch **`v2-savepoint`** |
| On the laptop | The `DeepFake-detection` folder that `run.ps1` runs from |
| Trained weights | Not on GitHub (too big). `model_v2.zip` / `model_v3.zip` from Kaggle; `run.ps1` unzips them into `model/` |
| Test results of each training run | `reports/v1/`, `reports/v2/` (and `reports/v3/` after the v3 run) |
| Training notebook | `notebooks/kaggle_train.ipynb`, run on Kaggle's free GPU |

---

## 3. How to show it working

```powershell
cd DeepFake-detection
powershell -ExecutionPolicy Bypass -File .\run.ps1          # newest model
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Use 2   # the v2 model
```

The browser opens **http://localhost:8000**. Upload a photo and point out:
1. **The verdict and score**, against the 65% cut-off.
2. **The blur-test map.** Red parts made the face look fake; green parts made it look real.
3. **The reasons list.** For example: "Blurring the eyes lowers the deepfake score from 89% to 8%."
4. **Grad-CAM focus** (the yellow map): where the model was looking.
5. **"How accurate is it?"**: accuracy, AUC, the confusion matrix and the classification report for each dataset.

To show the checks pass: `python -m pytest -q tests` (11 tests; no internet needed).

---

## 4. The files, in the order a photo travels through them

### `app.py`: the web server (FastAPI)
- `GET /` sends the page, `static/index.html`.
- `POST /api/predict` receives the uploaded photo and returns the JSON result from `Detector.predict`.
- `GET /api/metrics` reads the test results that belong to the model in use (`reports/v<N>/after/results.json`) and
  adds a classification report (precision, recall, F1, support per class) computed from each confusion matrix.
- `GET /api/info` reports the model's real size and settings (layers, heads, patch size, parameter counts), read
  from the loaded model rather than typed in by hand.
- `DATASETS` holds each dataset's name, description and link for the results table.

### `detector.py`: find the faces, score them
- `find_faces` runs MTCNN and keeps faces with confidence ≥ 0.9, largest first, with their 5 landmarks.
- `square` / `crop_face` cut a square 1.3× the face box, so the hairline, jaw and blend edges stay in view.
- `label_index` reads which output means FAKE from the model's own `id2label`. REAL and FAKE can never be swapped.
- `Detector.__init__` loads the processor, the ViT, the cut-off from `model/threshold.json` and the model version.
- `Detector.p_fake` turns a batch of face crops into P(fake) with one forward pass and a softmax.
- `Detector.predict` puts it together:
  1. Find faces.
  2. Score each one (up to 12).
  3. The face with the highest score decides the photo.
  4. Explain up to 4 faces, always including the deciding one.
  5. Write the summary.

### `explain.py`: why REAL or FAKE
- `log_odds` gives the FAKE logit minus the REAL logit. Using log-odds lets small effects show even when the model
  is 99% sure.
- `regions` builds masks for the eyes, nose, mouth, skin and surround from MTCNN's landmarks.
- **The blur test** (`explain_face`):
  1. Blur one region (Gaussian, radius 8) and score again.
  2. effect = score(full) − score(blurred). A positive effect means that part looked **fake** (red); a negative
     effect means it looked **real** (green).
  3. This is done for the 5 regions and for a 6 × 6 grid on the deciding face: 41 extra forward passes.
- `gradcam` computes Grad-CAM on the patch tokens entering the last transformer block, giving the yellow focus map.
- `summary` writes the plain-English explanation, using only the measured numbers.

### `static/index.html`: the page
One HTML file with its own CSS and JavaScript, no framework. It shows:
- the upload area;
- the verdict donut and meter;
- the photo with numbered face boxes;
- the map views: compare, slider and overlay;
- the reasons list;
- "Models used";
- the results table with dataset links;
- the confusion matrices and classification reports.

---

## 5. How the model was trained (all on Kaggle)

| Step | File | What happens |
|---|---|---|
| 1. Get the data | `kaggle_setup.py`, `fetch_hf.py` | Find the Kaggle datasets wherever Kaggle attached them; download the Gemini images from Hugging Face |
| 2. Make face crops | `prepare.py` | MTCNN on every image (or video frame), crop, sort into `faces/<train,val,test>/<dataset>/<real,fake>/` |
| 3. Measure the original model | `evaluate.py` | The "before" numbers |
| 4. Fine-tune | `train.py` | Train the ViT; keep the best epoch; choose the cut-off on validation |
| 5. Measure our model | `evaluate.py` | The "after" numbers on the same test faces |

The notebook `notebooks/kaggle_train.ipynb` runs these steps as 7 cells.

### `prepare.py`
- **Labels** come from folder names (`real`, `fake`, `AI-Generated Images`, …) or from `metadata.json` for DFDC videos.
- **Splits:** the dataset's own train/val/test if it has one; otherwise a fixed 70 / 15 / 15 split by file name.
  `--all-test` makes a test-only set.
- **DFDC is split by person:** a fake video and the real video it was made from always land in the same split, so the
  model cannot pass the test by recognising a face it trained on.
- **JPEG re-saving:** with `--jpeg`, some generated images are re-saved as JPEG, so "PNG = fake" can't become a shortcut.
- Already-cropped images are skipped, so an interrupted run can carry on.

### `train.py`
- **Optimiser:** AdamW, learning rate 2e-5, batch 32, mixed precision, 3 epochs.
- **Learning rate (v3):** warms up for 5% of the steps, then follows a cosine curve down; gradients are clipped at 1.0.
- **Augmentation:** mirror, downscale, blur and JPEG damage, so the model can't rely on perfectly clean pixels.
- **Balanced sampling:** each (dataset, class) group is weighted by 1/√size, so small sets are not drowned out.
- **Model selection:** after each epoch it measures validation balanced accuracy and keeps the best epoch.
- **Cut-off:** the cut-off with the best balanced accuracy is saved to `threshold.json` (0.65 for v2).
- **Kaggle (v3):** uses both GPUs, saves a checkpoint each epoch (`--resume`), and stores the weights in 16-bit to
  halve the download.

### `evaluate.py`
For every test set it writes accuracy, precision, recall, F1, AUC and the confusion matrix to `results.json`, and a
confusion-matrix image.

---

## 6. The versions

| Version | Trained on | Result |
|---|---|---|
| Original (dima806) | not by us | A 64.9%, B 54.7%, C 69.6% (AUC on C 0.54) |
| **v1** | A + B | A 92.5%, B 97.5%, **C 25.1%**: it learned GAN fakes but missed face swaps |
| **v2** (current) | A + B + part of C, split by person; augmentation; tuned cut-off | A 92.2%, B 96.0%, **C 86.1%** (AUC 0.975 / 0.993 / 0.927) |
| **v3** (set up, not yet trained) | v2's data + D (newer AI faces) + E (Gemini images); tested on G (Nano Banana 2, test only) | Built to fix v2's 50/50 on Gemini images |

Datasets:
- **A** Deepfake and Real Images (Kaggle, manjilkarki)
- **B** 140k Real and Fake Faces (Kaggle, xhlulu; StyleGAN)
- **C** DeepFake Detection Challenge sample (Kaggle competition; face-swap videos)
- **D** Human Faces Dataset (Kaggle, kaustubhdhote)
- **E** Gemini Nano Banana images (Hugging Face bitmind/nano-banana, plus ahnuf05's Kaggle set)
- **G** MLLM-Generated Image Detection benchmark (Hugging Face, zr-zhang)

---

## 7. Words used in the code

| Term | Meaning |
|---|---|
| P(fake) | The model's probability that a face is a deepfake (0 to 1) |
| Cut-off / threshold | P(fake) at or above it means FAKE (0.65 in v2) |
| Logit / log-odds | The model's raw score before softmax; FAKE logit minus REAL logit |
| False positive | A real face called fake |
| False negative | A fake that passed as real |
| AUC | How well fakes are ranked above reals, at any cut-off (0.5 = coin flip, 1 = perfect) |
| Balanced accuracy | The average of the accuracy on real faces and on fakes; fair when one class is bigger |
| Domain gap | Test images unlike anything in training (why v1 failed on C and v2 is unsure on Gemini) |
| Grad-CAM | A map of which image patches pushed the model's answer, from its gradients |
| Blur test (occlusion) | Blur one part, score again; the change shows how much that part mattered, and in which direction |

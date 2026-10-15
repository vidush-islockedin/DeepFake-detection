# DeepFake-detection (MT-ViT)

AI Mini Project: **Deepfake Detection of Images.** MTCNN finds each face → a Vision Transformer (ViT-B/16) says **REAL** or
**FAKE**, with P(FAKE). Every answer comes with a red/blue map of which parts looked fake or real (a blur test: blur a part,
score again), a reason for each part of the face, a Grad-CAM focus map and a plain-English summary. The page also shows
the models used, the three datasets with links, and each dataset's accuracy, classification report and confusion matrix.
Runs on a normal laptop CPU at **http://localhost:8000**. Training happens on Kaggle's free GPU.

Team: Abdulla Mahir · Vidush Karnani · Ishaan Kulkarni

Why every model, dataset and setting was chosen (and viva Q&A): see **[DECISIONS.md](DECISIONS.md)**.

## Run it on your laptop (Windows / Mac / Linux)
1. Install Python 3.10–3.12.
2. In this folder:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   pip install facenet-pytorch==2.6.0 --no-deps
   ```
3. Put the trained `model/` folder here (unzip `model.zip`). Without it, the app uses the original pre-trained model.
4. `python app.py` → open **http://localhost:8000** and upload a photo.

## Train it on Kaggle (once, ~1 hour, free)
1. kaggle.com → **Create → New Notebook** → **File → Import Notebook** → upload `notebooks/kaggle_train.ipynb`.
2. **+ Add Input** and add all three:
   - A: `manjilkarki/deepfake-and-real-images`
   - B: `xhlulu/140k-real-and-fake-faces`
   - C: competition `deepfake-detection-challenge` (accept its rules first; only `train_sample_videos` is used)
3. Settings (right panel): **Accelerator → GPU**, **Internet → On** (needs a phone-verified account).
4. **Run All**. When done, download `model.zip` and `results.zip` from **Output**.
5. Unzip `results.zip` into `reports/v<N>/` (so it holds `before/`, `after/`, `history.csv`); the web page shows the newest run.

Current model: **v2** (`reports/v2/`): A 92.2%, B 96.0%, C 86.1% on held-out test faces, cut-off 0.65.
Training run v1 (A + B only) is in `reports/v1/`. Run v2 adds the training part of C (DFDC, split by person so no
identity is in both train and test), video-style augmentation (blur, downscale, JPEG), balanced sampling and a
REAL/FAKE cut-off chosen on validation (`model/threshold.json`).

## What each file does
| File | Job |
|---|---|
| `detector.py` | Find faces, crop, score. Labels are read from the model's own config, so REAL/FAKE can't swap. |
| `app.py` | The localhost:8000 web app + `/api/predict`. |
| `prepare.py` | Datasets → face crops in `faces/<split>/<A,B,C>/<real,fake>/`. |
| `train.py` | Fine-tune the ViT (augmented, balanced), keep the best epoch, pick the cut-off. |
| `explain.py` | Blur test (red/blue map + reason per face part), Grad-CAM focus map, plain-English summary. |
| `evaluate.py` | Accuracy, precision, recall, F1, AUC, confusion matrix per test set. |
| `reports/` | Results measured on Kaggle, one folder per training run. |
| `tests/` | `pytest` checks with a tiny stand-in model (no internet needed). |

The maps show where the model found its evidence, not a pixel-exact map of edits.

Put a few photos in `static/samples/` and they appear as one-click examples on the page.

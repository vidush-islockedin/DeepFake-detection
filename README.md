# DeepFake-detection (MT-ViT)

AI Mini Project: **Deepfake Detection of Images.** MTCNN finds each face → a Vision Transformer (ViT-B/16) says **REAL** or
**FAKE**, with P(FAKE). Every answer comes with a red/blue map of which parts looked fake or real (a blur test: blur a part,
score again), a reason for each part of the face, a Grad-CAM focus map and a plain-English summary. The page also shows
the models used, the datasets with links, and each dataset's accuracy, classification report and confusion matrix.
Runs on a normal laptop CPU at **http://localhost:8000**. Training happens on Kaggle's free GPU.

Team: Abdulla Mahir · Vidush Karnani · Ishaan Kulkarni

Why every model, dataset and setting was chosen (and viva Q&A): see **[DECISIONS.md](DECISIONS.md)**.

## Run it on your laptop (Windows)
Install Python 3.10–3.12, put `model_v3.zip` (from Kaggle) in Downloads, then in this folder:
```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1          # newest model
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Use 2   # back to the v2 savepoint (-Use 3 to return)
```
It updates the code, sets up Python, installs the newest `model_v*.zip` (keeping the old model as `model_v<N>`), and
opens **http://localhost:8000**. Mac/Linux: `python -m venv .venv && source .venv/bin/activate && pip install -r
requirements.txt && pip install facenet-pytorch==2.6.0 --no-deps`, unzip the model into `model/`, `python app.py`.

## Train it on Kaggle (~2 hours, free)
1. kaggle.com → **Create → New Notebook** → **File → Import Notebook** → upload `notebooks/kaggle_train.ipynb`.
2. **+ Add Input**:
   - A: `manjilkarki/deepfake-and-real-images` · B: `xhlulu/140k-real-and-fake-faces`
   - C: competition `deepfake-detection-challenge` (accept its rules first; only `train_sample_videos` is used)
   - D: `kaustubhdhote/human-faces-dataset` · E: `ahnuf05/nano-banana-2-0-the-omni-subject-dataset`
   - G (optional): your own dataset `gemini-faces-test` with a `fake/` folder of Gemini faces (and `real/` if you have any)
3. Settings: **Accelerator → GPU T4 x2**, **Internet → On** (phone-verified account; E downloads from Hugging Face).
4. **Run All**. Any cell can be re-run after a restart; finished work is skipped and training resumes from its last epoch.
5. Download `model_v3.zip` and `results_v3.zip`. Unzip the results into `reports/v3/` (`before/`, `after/`, `history.csv`).

Savepoint: v2 is git tag `v2` (branch `v2-savepoint`); its results stay in `reports/v2/`, v1's in `reports/v1/`.
v2 results: A 92.2%, B 96.0%, C 86.1% on held-out test faces, cut-off 0.65. Gemini-made faces scored about 50/50,
which is why v3 adds D, E and G (see DECISIONS.md, "What v3 changes and why").

## What each file does
| File | Job |
|---|---|
| `detector.py` | Find faces, crop, score. Labels are read from the model's own config, so REAL/FAKE can't swap. |
| `app.py` | The localhost:8000 web app + `/api/predict`. |
| `prepare.py` | Datasets → face crops in `faces/<split>/<A..G>/<real,fake>/` (labels from folder names or `--label`). |
| `fetch_hf.py` | Downloads the Gemini (Nano Banana) images for E from Hugging Face. |
| `kaggle_setup.py` | Notebook helper: finds the datasets, runs two datasets at once (one per GPU), survives restarts. |
| `run.ps1` | Windows: one command to update, set up, switch model version and start the app. |
| `train.py` | Fine-tune the ViT (augmented, balanced, warm-up + cosine, 2 GPUs, resumable), keep the best epoch, pick the cut-off. |
| `explain.py` | Blur test (red/blue map + reason per face part), Grad-CAM focus map, plain-English summary. |
| `evaluate.py` | Accuracy, precision, recall, F1, AUC, confusion matrix per test set. |
| `reports/` | Results measured on Kaggle, one folder per training run. |
| `tests/` | `pytest` checks with a tiny stand-in model (no internet needed). |

The maps show where the model found its evidence, not a pixel-exact map of edits.

Put a few photos in `static/samples/` and they appear as one-click examples on the page.

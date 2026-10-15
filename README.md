# DeepFake-detection (MT-ViT)

MTCNN finds each face → a Vision Transformer (ViT-B/16) says **REAL** or **FAKE**, with P(FAKE).
Runs on a normal laptop CPU at **http://localhost:8000**. Training happens once, on Kaggle's free GPU.

Team: Abdulla Mahir · Vidush Karnani · Ishaan Kulkarni

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

## What each file does
| File | Job |
|---|---|
| `detector.py` | Find faces, crop, score. Labels are read from the model's own config, so REAL/FAKE can't swap. |
| `app.py` | The localhost:8000 web app + `/api/predict`. |
| `prepare.py` | Datasets → face crops in `faces/<split>/<A,B,C>/<real,fake>/`. |
| `train.py` | Fine-tune the ViT on A + B, keep the best epoch. |
| `evaluate.py` | Accuracy, precision, recall, F1, AUC, confusion matrix per test set. |
| `tests/` | `pytest` checks with a tiny stand-in model (no internet needed). |

Dataset C (DFDC) is **only** used for testing, to show how the model does on data it never saw.

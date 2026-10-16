# DECISIONS.md: why MT-ViT is built the way it is

This file explains every major choice in the project: what we used, why, what else we could have used, and what each choice changed. It ends with likely viva questions and short answers.

Team: Abdulla Mahir, Vidush Karnani, Ishaan Kulkarni. AI Language Mini Project.

---

## 1. The project in one paragraph

MT-ViT decides whether a face in a photo is **real** or a **deepfake**, and shows why. It works in three steps:

1. **MTCNN** finds every face and its five landmarks (two eyes, nose, two mouth corners).
2. A **Vision Transformer (ViT-B/16)** gives each face a probability of being fake.
3. A **blur test** blurs each part of the face and scores it again, to show which parts looked fake and which looked real; **Grad-CAM** shows where the model looked. Both feed a plain-English summary built only from measured numbers.

It runs on a normal laptop CPU at `localhost:8000`. Training was done once, on Kaggle's free GPU.

```
photo ─► MTCNN (3 small CNNs) ─► face boxes + landmarks ─► square crop ×1.3 ─► resize 224×224
      ─► ViT-B/16 (12 transformer blocks) ─► P(FAKE) per face ─► verdict (FAKE if any face ≥ cut-off)
      ─► blur test (5 face parts + 6×6 grid) ─► red/green map + a reason per part
      ─► Grad-CAM on the last block ─► yellow focus map ─► plain-English summary
```

---

## 2. The networks we use

### 2.1 MTCNN: the face finder (this is the convolutional part)

MTCNN stands for **Multi-task Cascaded Convolutional Networks** (Zhang et al., 2016). It is a chain of **three small CNNs**. Each one looks at fewer, better candidates than the one before.

| Stage | Network | Input | What it does |
|---|---|---|---|
| 1 | **P-Net** (Proposal) | Image pyramid, 12×12 windows | A fully convolutional net that proposes many rough face boxes quickly |
| 2 | **R-Net** (Refine) | 24×24 crops | Rejects most false boxes and refines the rest |
| 3 | **O-Net** (Output) | 48×48 crops | Gives the final boxes, a confidence and the **5 facial landmarks** |

Each stage then runs **non-maximum suppression**, which keeps the best box where several overlap.

- **Implementation:** `facenet-pytorch` 2.6.0, with weights pre-trained on face-detection data. We did not train MTCNN.
- **Settings:** we keep faces with confidence ≥ 0.9 and sort them largest first.
- **Why MTCNN?**
  - It is accurate on frontal and near-frontal faces.
  - It is light enough for a laptop CPU.
  - It gives **landmarks** as well as boxes. Our explanation needs those to locate the eyes, nose and mouth.
- **Alternatives:**

| Detector | Strength | Why not |
|---|---|---|
| Haar cascades (OpenCV) | Very fast | Many misses on tilted or poorly lit faces; no landmarks |
| dlib HOG | Gives 68 landmarks | Weaker on small faces; awkward to install on Windows |
| RetinaFace | More accurate on hard faces | Heavier and slower on CPU; more code |
| MediaPipe | Very fast | Different landmark format; extra dependency |

### 2.2 ViT-B/16: the deepfake classifier

The classifier is a **Vision Transformer, base size, 16×16 patches** (Dosovitskiy et al., 2021). It is not a CNN.

- **Patches:** the 224×224 face crop is cut into 14 × 14 = **196 patches** of 16×16 pixels.
- **Tokens:** each patch is flattened and projected to a 768-number vector. A learned **[CLS] token** is added, and **position embeddings** say where each patch came from.
- **Encoder:** **12 transformer blocks**, each with 12 attention heads and a feed-forward layer. About **86 million** parameters.
- **Attention:** in every block, every patch can look at every other patch. One layer can compare the left eye with the right eye, or the skin texture with the hairline.
- **Output:** the final [CLS] vector goes to a 2-way classifier (REAL, FAKE). Softmax turns the two scores into probabilities.

**Pre-trained weights:** we start from `dima806/deepfake_vs_real_image_detection` on Hugging Face (Apache-2.0 licence).

- It is a ViT-B/16 that its author already fine-tuned for real-vs-fake faces, starting from Google's ImageNet-21k ViT.
- Its model card reports about 99% accuracy on the author's own test split. We **did not trust that number**; we re-measured it on our test sets (the "Original model" column in section 5).

**Why a ViT and not a CNN?**

| | ViT-B/16 (ours) | CNN (e.g. EfficientNet, Xception) |
|---|---|---|
| What it looks at | The whole face at once, from the first layer | Small local windows that grow layer by layer |
| Good at | Global inconsistencies: mismatched lighting, asymmetry, blend boundaries | Local texture artefacts: GAN fingerprints, upsampling patterns |
| Data needed from scratch | A lot (weak inductive bias) | Less |
| Our situation | Pre-trained deepfake weights exist, so the data hunger doesn't hurt us | Would need training on deepfakes first |
| Fit with the project | The project is "MT-**ViT**", and the paper studies transformers | — |

The **"M" (MTCNN) + "ViT"** pairing is deliberate. The CNN does what CNNs are best at (finding faces quickly). The transformer does what transformers are best at (comparing distant parts of the face).

---

## 3. The datasets

A to E are public (Kaggle, and one Hugging Face set downloaded inside the Kaggle notebook); G is ours. Training ran on Kaggle, so nothing was downloaded to our laptops.

| | Dataset | What is in it | How we used it |
|---|---|---|---|
| **A** | *Deepfake and Real Images* (manjilkarki) | Real and fake face images, pre-split into Train / Validation / Test | Train + validation + test |
| **B** | *140k Real and Fake Faces* (xhlulu) | 70k real faces from Flickr (FFHQ) and 70k StyleGAN-generated faces, split 100k / 20k / 20k | Train + validation + test |
| **C** | *Deepfake Detection Challenge* sample (Meta/AWS/Microsoft, 2020) | 400 short videos with labels, mostly **face swaps**; about 4 fakes for every real | v1: test only. v2: split by person into train / val / test |
| **D** (v3) | *Human Faces Dataset* (kaustubhdhote) | About 5k real and 5k AI-generated face photos from newer generators | Fixed 70 / 15 / 15 split by file name |
| **E** (v3) | *Gemini (Nano Banana) images*: bitmind/nano-banana on Hugging Face (MIT) + ahnuf05's Kaggle set | Images made by Google's Gemini image model; only those where MTCNN finds a face are used. All fake | 70 / 15 / 15 split |
| **G** (v3) | *MLLM-Generated Image Detection* benchmark (zr-zhang, 2026): real photos and **Nano Banana 2** images | A newer Gemini model than E's, from a different source | **Test only**, never trained on |
| **H** (v3, optional) | *Our Gemini test photos* | Faces we made ourselves in the Gemini app | **Test only** |

### Why these three
- **A and B** are large, balanced, already split, and give clean labels from folder names. A gives variety; B gives a well-known GAN benchmark (StyleGAN vs real Flickr faces).
- **C** is a **different kind of deepfake**: a face swapped onto a real video, then video-compressed. Testing on it shows whether the model generalises beyond the fakes it trained on. That is the most important question for any deepfake detector.
- We picked **different sources on purpose**. A model that only sees one generator learns that generator's fingerprint, not "fakeness" in general. Section 5 shows exactly this happening.

### Alternatives we did not use
| Dataset | Why not |
|---|---|
| FaceForensics++ | The standard benchmark, but requires a signed request form and days of waiting |
| Celeb-DF v2 | Also behind a request form |
| Full DFDC (470 GB) | Far too large for a free Kaggle session |
| Generated images from Midjourney / Stable Diffusion (v1, v2) | No large, labelled, face-only public set we could verify in time. v3 adds D and E for this |

### How the data was prepared (`prepare.py`)
1. Run MTCNN on every image (or the middle frame of every DFDC video).
2. Crop a **square 1.3 times the face box**, so the hairline, jaw and blending edges stay in view. Fake blending artefacts often sit right at the face boundary.
3. Save the crop to `faces/<split>/<dataset>/<real|fake>/`. The label comes from the folder name (A, B) or `metadata.json` (C). Images with no face are skipped and counted.
4. Take at most **5,000 images per class** for training, and **1,000 per class** for validation and test, from A and B. This kept a v2 Kaggle session around one hour; v3, with D and E added, takes about two hours.
   - Sampling is random with a **fixed seed (42)**, so the same images are picked every run and results can be compared.

---

## 4. Training choices (`train.py`)

| Setting | Value | Why |
|---|---|---|
| Start point | dima806 ViT-B/16 | Already knows faces and some deepfake cues, so we only need to fine-tune |
| Optimiser | AdamW, learning rate 2e-5, weight decay 0.01 | The standard recipe for fine-tuning transformers. A small learning rate changes pre-trained weights gently and avoids forgetting |
| Batch size | 32 | Fits the free T4/P100 GPU memory |
| Mixed precision (AMP) | On | About 2× faster on GPU with the same accuracy |
| Epochs | 2 (v1), 3 (v2, v3) | Validation accuracy was still rising after epoch 2 in v1 (93.9% → 96.5%) |
| Learning-rate schedule (v3) | 5% warm-up, then cosine decay; gradients clipped at 1.0 | v2's fixed rate made epoch 3 worse than epoch 2 |
| Model selection | Keep the epoch with the best **validation** accuracy (v1, v2) or **balanced accuracy** (v3) | The test set is never used to make choices, so test numbers stay honest |
| Hardware (v3) | Both Kaggle T4 GPUs, a checkpoint after each epoch | About 1.7× faster, and a Kaggle restart no longer loses the run |
| Augmentation v1 | Random horizontal mirror | Faces are roughly symmetric; this doubles the variety for free |
| Augmentation v2 | Mirror + random downscale, blur and JPEG re-compression | Teaches the model to survive video-style damage, which is what C looks like |
| Label mapping | Read from the model's own `config.id2label` | A previous version swapped REAL and FAKE because it assumed alphabetical folder order. Reading the label names makes a swap impossible |

### What v2 changes and why
v1 scored 25% on C. v2 attacks the cause, not the symptom:

1. **Face-swap examples in training.** About 60% of DFDC **people** go to training, 15% to validation and 25% to test.
   - The split is by **identity**: a fake video and the real video it was made from always land in the same split.
   - Otherwise the model could "recognise the person" instead of detecting the fake. That would be data leakage.
2. **Several frames per training video** (4 per fake, 16 per real) to get enough examples. Real videos get more frames because there are about 4× fewer of them.
3. **Balanced sampling.** Each dataset is weighted by 1/√(size), so C's few thousand crops are not drowned out by A and B's 20,000 each. The square root stops the model memorising C's handful of people.
4. **Video-style augmentation** (see the table above).
5. **A tuned cut-off.** v1 called FAKE at P ≥ 0.5. v2 picks the cut-off with the best **balanced accuracy on validation** and saves it in `model/threshold.json`. The app reads it automatically.

**v1 is kept.** Its results stay in `reports/v1/`, and its weights stay as a separate download. v2's results are in `reports/v2/`; the web page shows the newest run. The cut-off chosen for v2 was **0.65**.

### What v3 changes and why
v2 gave about 50 / 50 on faces made by Gemini. The reason is the same as v1 on C: **no image like that was in training**.
Gemini makes images with a diffusion model; A and B are GAN faces and C is face swaps.

1. **New data.** D (newer AI faces, with matching real faces) and E (Gemini's own images) join training. G (Nano Banana 2, a newer
   Gemini model, from a different benchmark) is a test set only, so a good score on G cannot come from having seen those images.
2. **No "PNG = fake" shortcut.** Generated images often arrive as PNG while real photos are JPEG. Half of the D and E
   images are re-saved as JPEG (quality 70-95) before cropping, so the file format cannot give the answer away.
3. **Balanced by group.** Sampling now weights every (dataset, class) group by 1/√(size), so a fake-only set (E) cannot
   tilt the model towards FAKE.
4. **Training schedule.** 5% warm-up then a cosine decay of the learning rate, and gradient clipping at 1.0. v2 used a
   fixed rate and its third epoch was worse than its second.
5. **Best epoch by balanced accuracy** (mean of real-correct and fake-correct) at that epoch's own cut-off. Plain
   accuracy would reward an epoch that leans towards the larger class.
6. **Faster and restart-proof on Kaggle.** Both T4 GPUs are used (training and face cropping), every epoch is
   checkpointed and resumed after a restart, every notebook cell works on its own, and the weights are stored in 16-bit
   (about 170 MB instead of 330 MB; they are loaded back as 32-bit, and the test numbers are measured on that file).

**v2 is kept as a savepoint.** Code: git branch `v2-savepoint`. Weights: `model_v2.zip` / the `model_v2`
folder. `run.ps1 -Use 2` switches the app back to v2 and the page then shows v2's results.

---

## 5. Results (measured on Kaggle test sets the model never trained on)

*v3 (with Gemini and diffusion faces) is set up but not trained yet; its results will go in `reports/v3/` and in a
table here. Until then v2 is the current model.*

### v2: the current model (cut-off 0.65, chosen on validation)

| Test set | Faces | Accuracy | Balanced acc. | Precision | Recall | F1 | AUC | False positives | False negatives |
|---|---|---|---|---|---|---|---|---|---|
| A | 1,878 | **92.2%** | 92.2% | 0.92 | 0.93 | 0.92 | 0.975 | 76 | 71 |
| B | 1,999 | **96.0%** | 96.0% | 0.96 | 0.96 | 0.96 | 0.993 | 38 | 42 |
| C | 115 | **86.1%** | 87.6% | 0.98 | 0.85 | 0.91 | 0.927 | 2 | 14 |

Validation accuracy during training: 92.1% (epoch 1), **95.3% (epoch 2, kept)**, 94.7% (epoch 3).

### v1 compared with v2

| Test set | v1 | v2 | What changed |
|---|---|---|---|
| A | 92.5% | 92.2% | −0.3 points: within noise for 1,878 faces |
| B | 97.5% | 96.0% | −1.5 points: the model leans less on tiny GAN traces now that it also learns face swaps and blurred, compressed photos |
| C | 25.1% | **86.1%** | +61 points; AUC 0.61 → 0.93. The domain gap is largely closed |

Note: v1's C test used one frame from all 382 videos (C was unseen). v2 trains on C, so its C test is only the 25% of
**people** held out for testing: 115 faces. That number is honest but less precise than A's or B's.

### v1 (kept for comparison, `reports/v1/`)

| Test set | Faces | Accuracy before → after | Precision | Recall | F1 | AUC | False positives | False negatives |
|---|---|---|---|---|---|---|---|---|
| A | 1,878 | 64.7% → **92.5%** | 0.92 | 0.94 | 0.93 | 0.976 | 79 | 62 |
| B | 1,999 | 54.3% → **97.5%** | 0.98 | 0.97 | 0.98 | 0.997 | 23 | 26 |
| C | 382 | 68.1% → **25.1%** | 1.00 | 0.07 | 0.12 | 0.611 | 0 | 286 |

### How to read these numbers
- **FAKE is the positive class.** A *false positive* is a real face called fake. A *false negative* is a fake that slipped through as real.
- **Precision** asks: when the model says FAKE, how often is it right? **Recall** asks: of all the fakes, how many did it catch? **F1** balances the two.
- **AUC** measures how well the model *ranks* fakes above reals, whatever cut-off is used. 0.5 is a coin flip; 1.0 is perfect.
- **Balanced accuracy** is the average of the recall on real faces and on fakes. Use it (and AUC) for an imbalanced set like C, which is about 80% fake.
- **Why v1 failed on C:** it learned what *GAN* fakes look like, and C's *face swaps* in compressed video look different, so it called them real (286 of 306 missed). v2 fixed this by training on face swaps (split by person), adding blur/JPEG augmentation, balancing the datasets and tuning the cut-off.
- **The web page shows a full classification report** (precision, recall, F1 and support for REAL and FAKE separately) and the confusion matrix for each set.

---

## 6. How the explanation works (`explain.py`)

We do not make up reasons. Every sentence is built from measured numbers. Two measurements are made on each face.

1. **Blur test (occlusion).** This is what makes the red/green map and the per-part reasons.
   - Blur one part of the face crop (Gaussian, radius 8 px at 224 px), score it again, and compare the **log-odds** (FAKE logit minus REAL logit) with the original.
   - If blurring a part **lowers** the deepfake score, that part's fine detail was making the face look **fake** (red). If blurring it **raises** the score, its detail was making the face look **real** (green).
   - Done for the five parts from MTCNN's landmarks (eyes, nose, mouth, cheeks/forehead/jaw, hair/ears/background), and for a 6 × 6 grid of squares on the deciding face to draw the map. That is 41 extra forward passes, a few seconds on a laptop CPU.
   - Why this method: it is **causal** (it is what the model actually does when the detail is gone) and it gives a **direction** (fake or real) for every part, which Grad-CAM on its own does not.
   - Log-odds, not probability, because when the model is 99% sure, probabilities barely move even when a part matters a lot.
2. **Grad-CAM focus.** Shown as a second map (yellow) and as "Grad-CAM focus %" for each part.
   - Take the 196 patch tokens entering the **last** transformer block, take the gradient of (verdict logit − other logit), average it over the 768 features, weight the tokens, sum and keep the positive part (ReLU). A 14 × 14 map, upsampled to 224 × 224.
   - Why the last block's *input*? The classifier only reads the [CLS] token, and patch tokens reach [CLS] only through attention. After the last block they no longer affect the answer, so their gradient would be zero.
3. **The summary** gives the deepfake score against the cut-off, the part that moves the score most (with the before/after numbers), any part that points the other way, whether the result is close to the cut-off, and which face decided the verdict when there are several.

**What the maps are not.** They show *where the evidence was*. They are not a pixel-perfect map of what was edited; a classifier trained on real/fake labels cannot know that. We say this on the page.

---

## 7. Smaller design decisions

| Decision | Reason |
|---|---|
| A photo is FAKE if **any** face is fake | One swapped face in a group photo is enough to make it a manipulated image |
| No face found means **no verdict** | Guessing on a face-less image would just be noise |
| Runs locally, nothing uploaded | Privacy: people test photos of themselves and friends |
| FastAPI + one HTML page (not Next.js) | One `python app.py` command, no Node.js, and it works on any teammate's laptop |
| Fonts bundled in the repo | The page looks the same offline |
| Weights not in the Git repo | About 340 MB (v1, v2) or 170 MB (v3, 16-bit), over GitHub's 100 MB file limit. Shared as a zip; `run.ps1` installs it |
| Automated tests use a tiny random ViT | They check the plumbing (labels, crops, API) without the internet or the 340 MB model |

---

## 8. Limitations and ethics

- **The verdict is only reliable on fakes like the training data.** Strong on GAN faces (A, B) and, since v2, much better on DFDC face swaps (C), but C's test set is small (115 faces). v2 does not handle newer diffusion-model images (Gemini, Midjourney, DALL·E), which is what v3's D, E and G target. New generators keep appearing, so no detector stays complete.
- **It judges only the face.** A real face pasted onto a fake background, or a manipulated voice, is out of scope.
- **Small, blurry, heavily filtered or side-on faces** reduce accuracy, and MTCNN may not find them at all.
- **Dataset bias.** The training faces come from public sources, and the demographic balance is not documented by the dataset authors, so accuracy may differ across groups. We did not measure this.
- **It is an aid, not proof.** A detector can be fooled on purpose (adversarial noise). A high score should lead to checking the source, not to accusing someone.

---

## 9. Viva questions and answers

**Q1. What problem does the project solve?**
It classifies a face image as real or deepfake, gives a probability, and explains which parts of the face drove the decision.

**Q2. Why two models instead of one?**
Detection and classification are different jobs. MTCNN finds and aligns faces quickly. The ViT only has to judge a tight face crop, which is easier and more accurate than judging a whole scene.

**Q3. Is there a CNN in your project?**
Yes: MTCNN is three cascaded CNNs (P-Net, R-Net, O-Net). The classifier is a transformer, not a CNN.

**Q4. Explain how a ViT sees an image.**
It cuts the 224×224 crop into 196 patches of 16×16 pixels and turns each into a vector. It adds position information and a [CLS] token, then runs 12 self-attention blocks in which every patch can attend to every other. The final [CLS] vector is classified as real or fake.

**Q5. Why ViT over CNN for deepfakes?**
Self-attention compares distant regions in one layer, which helps catch global inconsistencies such as lighting, symmetry and blend edges. CNNs build that context slowly. ViTs need more data, but we avoided that problem by starting from pre-trained deepfake weights.

**Q6. What is transfer learning and fine-tuning here?**
We started from a ViT already trained on ImageNet-21k and then on real/fake faces. We continued training all its layers on our crops with a small learning rate (2e-5), so it adapts without forgetting.

**Q7. Why learning rate 2e-5 and AdamW?**
These are the standard settings for fine-tuning transformers. A large learning rate would wreck the pre-trained weights. AdamW handles weight decay correctly, which reduces overfitting.

**Q8. How did you avoid data leakage?**
Each dataset's own train/test split is kept. Test sets are never used for choices. In v2, DFDC is split by person, so a fake and its source video are never on opposite sides.

**Q9. Why is the 1.3× crop used?**
Face-swap artefacts are often at the boundary where the fake face is blended in: the hairline, jaw and ears. A tight crop would cut them off.

**Q10. What are precision, recall, F1 and AUC? Which matters most for you?**
See section 5. For a deepfake filter, **recall** matters most (don't let fakes through), balanced against precision (don't accuse real photos). On imbalanced data we rely on **AUC and balanced accuracy**.

**Q11. Your accuracy on C dropped from 68% to 25%. Isn't that bad?**
The 68% was mostly the original model's bias towards saying FAKE on a set that is 80% fake; its AUC was 0.59, near chance. Our model learned GAN fakes, and C contains face swaps, so it misses them. The AUC actually rose slightly. This is a domain gap, a known open problem. v2 fixed most of it with face-swap training data (split by person), video augmentation, balanced sampling and a tuned cut-off: C went from 25.1% to 86.1% (AUC 0.61 → 0.93).

**Q12. What is a domain gap?**
When test data comes from a different distribution than the training data, such as a different generator or compression, the patterns the model learned may not exist there.

**Q13. How does Grad-CAM work on a transformer?**
See section 6. We use the gradient of the verdict score with respect to the patch tokens entering the last block, weighted by the token values. It shows which patches pushed the decision.

**Q14. Does the heatmap show what was edited?**
No. It shows where the model found the evidence for its answer. A classifier trained only on real/fake labels cannot localise edits exactly.

**Q14b. How do you decide which parts look fake and which look real?**
The blur test (section 6): blur one part, score again. If the deepfake score drops, that part's detail looked fake (red); if it rises, it looked real (green). It is causal, and it gives a direction, which Grad-CAM alone does not. We compare log-odds so the effect still shows when the model is 99% sure.

**Q14c. A and B dropped slightly in v2. Is that a problem?**
No. A fell 0.3 points (within noise for 1,878 faces) and B 1.5 points, while C rose 61 points. One model now has to cover GAN faces and face swaps, and the blur/JPEG augmentation stops it relying on the tiniest GAN traces. That is the usual robustness trade-off, and it is clearly worth it.

**Q15. How do you know REAL and FAKE aren't swapped?**
The label index is read by name from the model's own config (`id2label`), never assumed. The automated tests check three different label orders.

**Q16. What happens with several faces?**
Each face is scored separately. The photo is FAKE if any face is above the cut-off, and the page names the face that decided it.

**Q17. Why not train your own model from scratch?**
A ViT-B has 86 million parameters and needs millions of images to train from zero. We had about 20,000 crops and a free GPU for a few hours. Transfer learning is the right tool.

**Q18. How would you improve it further?**
- More and newer kinds of fakes: diffusion models, more face-swap methods (FaceForensics++, Celeb-DF).
- Video-level decisions that average many frames.
- A small ensemble of a ViT and a CNN, which catch different artefacts.
- Calibration so that "90%" really means right 90% of the time.
- A fairness check across demographic groups.

**Q19. What is the cut-off and why 0.5 (or not)?**
In v1 it is 0.5. In v2 it is **0.65**, chosen on validation data to maximise balanced accuracy, because the best cut-off depends on the class balance and the kind of fakes. Moving the cut-off only trades false positives for false negatives; only a better model reduces both.

**Q20. Why Kaggle for training and a laptop for running?**
Fine-tuning an 86M-parameter ViT on 20,000+ images needs a proper GPU; Kaggle gives an NVIDIA T4/P100 for free. Our laptops have integrated Intel Iris Xe graphics, which would take many hours or days for the same job. Running the trained model on one image takes about a second on a CPU, so the demo works on any laptop.

**Q21. What does "ViT-B/16" mean?**
B = Base size (12 layers, 768-wide, 12 heads, about 86M parameters). 16 = 16×16-pixel patches.

**Q22. What are MTCNN's three outputs?**
A face/not-face confidence, the bounding box, and five landmarks (eyes, nose, mouth corners). That is the "multi-task" part.

**Q24. Why did v2 give about 50/50 on images made by Gemini?**
Gemini makes images with a diffusion model, and v2 never saw one in training (A and B are GAN faces, C is face swaps).
50/50 is the model saying "I don't know", which is honest. It is the same domain gap v1 had on C, and v3 fixes it the
same way: by adding that kind of fake to training (D and E).

**Q25. How do you know v3 really catches Gemini fakes and hasn't just memorised them?**
G is a separate benchmark of **Nano Banana 2** images (a newer Gemini model than the one in E) and real photos from a
different source, used **only for testing**. A good score on G cannot come from having seen those images.

**Q26. What is the "PNG shortcut" and how did you avoid it?**
Generated images often come as PNG while real photos are JPEG. A model can learn "no JPEG traces = fake", which is a
shortcut, not deepfake detection. Half of the D and E images are re-saved as JPEG before cropping, and training adds
random JPEG compression, so the file format cannot give the answer away.

**Q27. What changed in training for v3?**
Warm-up then cosine decay of the learning rate, gradient clipping, the best epoch picked by balanced accuracy,
sampling balanced per (dataset, class), both Kaggle GPUs, a checkpoint after every epoch, and 16-bit saved weights.
Section "What v3 changes and why" gives the reason for each.

**Q28. Can you go back to v2?**
Yes. The v2 code is the `v2-savepoint` branch on GitHub and its weights are kept as `model_v2`. `run.ps1 -Use 2`
switches the app back, and the page then shows v2's results.

**Q23. Is this safe to use as evidence?**
No. It is an assistive tool with known blind spots, and it can be fooled on purpose. A positive result should prompt verification of the source.

---

## 10. References
- K. Zhang et al., *Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks*, IEEE Signal Processing Letters, 2016.
- A. Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021.
- R. R. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
- B. Dolhansky et al., *The DeepFake Detection Challenge (DFDC) Dataset*, arXiv:2006.07397, 2020.
- T. Karras et al., *A Style-Based Generator Architecture for Generative Adversarial Networks* (StyleGAN), CVPR 2019.
- I. Loshchilov and F. Hutter, *Decoupled Weight Decay Regularization* (AdamW), ICLR 2019.
- Model weights: `dima806/deepfake_vs_real_image_detection`, Hugging Face, Apache-2.0.
- Datasets (v3): `kaustubhdhote/human-faces-dataset` (Kaggle); `bitmind/nano-banana` (Hugging Face, MIT);
  `ahnuf05/nano-banana-2-0-the-omni-subject-dataset` (Kaggle); `zr-zhang/MLLM-Generated-Image-Detection-Dataset`
  (Hugging Face; paper arXiv:2608.01258).
- Face detector code: `timesler/facenet-pytorch`, MIT licence.

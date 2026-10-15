# DECISIONS.md: why MT-ViT is built the way it is

This file explains every major choice in the project: what we used, why, what else we could have used, and what each choice changed. It ends with likely viva questions and short answers.

Team: Abdulla Mahir, Vidush Karnani, Ishaan Kulkarni. AI Language Mini Project.

---

## 1. The project in one paragraph

MT-ViT decides whether a face in a photo is **real** or a **deepfake**, and shows why. It works in three steps:

1. **MTCNN** finds every face and its five landmarks (two eyes, nose, two mouth corners).
2. A **Vision Transformer (ViT-B/16)** gives each face a probability of being fake.
3. **Grad-CAM** draws a heatmap of the evidence, which is split into face regions and turned into a one-sentence explanation.

It runs on a normal laptop CPU at `localhost:8000`. Training was done once, on Kaggle's free GPU.

```
photo ─► MTCNN (3 small CNNs) ─► face boxes + landmarks ─► square crop ×1.3 ─► resize 224×224
      ─► ViT-B/16 (12 transformer blocks) ─► P(FAKE) per face ─► verdict (FAKE if any face ≥ cut-off)
      ─► Grad-CAM on the last block ─► heatmap ─► share of evidence in eyes / nose / mouth / rest / surround
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

All three are public on Kaggle. Training ran there, so nothing was downloaded to our laptops.

| | Dataset | What is in it | How we used it |
|---|---|---|---|
| **A** | *Deepfake and Real Images* (manjilkarki) | Real and fake face images, pre-split into Train / Validation / Test | Train + validation + test |
| **B** | *140k Real and Fake Faces* (xhlulu) | 70k real faces from Flickr (FFHQ) and 70k StyleGAN-generated faces, split 100k / 20k / 20k | Train + validation + test |
| **C** | *Deepfake Detection Challenge* sample (Meta/AWS/Microsoft, 2020) | 400 short videos with labels, mostly **face swaps**; about 4 fakes for every real | v1: test only. v2: split by person into train / val / test |

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
| Generated images from Midjourney / Stable Diffusion | No large, labelled, face-only public set we could verify in time |

### How the data was prepared (`prepare.py`)
1. Run MTCNN on every image (or the middle frame of every DFDC video).
2. Crop a **square 1.3 times the face box**, so the hairline, jaw and blending edges stay in view. Fake blending artefacts often sit right at the face boundary.
3. Save the crop to `faces/<split>/<dataset>/<real|fake>/`. The label comes from the folder name (A, B) or `metadata.json` (C). Images with no face are skipped and counted.
4. Take at most **5,000 images per class** for training, and **1,000 per class** for validation and test, from A and B. This keeps a Kaggle session around one hour.
   - Sampling is random with a **fixed seed (42)**, so the same images are picked every run and results can be compared.

---

## 4. Training choices (`train.py`)

| Setting | Value | Why |
|---|---|---|
| Start point | dima806 ViT-B/16 | Already knows faces and some deepfake cues, so we only need to fine-tune |
| Optimiser | AdamW, learning rate 2e-5, weight decay 0.01 | The standard recipe for fine-tuning transformers. A small learning rate changes pre-trained weights gently and avoids forgetting |
| Batch size | 32 | Fits the free T4/P100 GPU memory |
| Mixed precision (AMP) | On | About 2× faster on GPU with the same accuracy |
| Epochs | 2 (v1), 3 (v2) | Validation accuracy was still rising after epoch 2 in v1 (93.9% → 96.5%) |
| Model selection | Keep the epoch with the best **validation** accuracy | The test set is never used to make choices, so test numbers stay honest |
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

**v1 is kept.** Its results stay in `reports/v1/`, and its weights stay as a separate download. v2 goes to `reports/v2/`, and the web page shows the newest run.

---

## 5. Results (v1, measured on Kaggle test sets the model never trained on)

| Test set | Faces | Accuracy before → after | Precision | Recall | F1 | AUC | False positives | False negatives |
|---|---|---|---|---|---|---|---|---|
| A | 1,878 | 64.7% → **92.5%** | 0.92 | 0.94 | 0.93 | 0.976 | 79 | 62 |
| B | 1,999 | 54.3% → **97.5%** | 0.98 | 0.97 | 0.98 | 0.997 | 23 | 26 |
| C | 382 | 68.1% → **25.1%** | 1.00 | 0.07 | 0.12 | 0.611 | 0 | 286 |

"Before" means the downloaded model with no fine-tuning. Validation accuracy during training was 93.9% after epoch 1 and 96.5% after epoch 2.

### How to read these numbers
- **FAKE is the positive class.** A *false positive* is a real face called fake. A *false negative* is a fake that slipped through as real.
- **Precision** asks: when the model says FAKE, how often is it right?
- **Recall** asks: of all the fakes, how many did it catch?
- **F1** is the balance of the two.
- **AUC** measures how well the model *ranks* fakes above reals, whatever cut-off is used. 0.5 is a coin flip; 1.0 is perfect.
- **On A and B,** fine-tuning removed the original model's bias towards saying FAKE. It had called 659 and 652 real faces fake. Fine-tuning lifted accuracy by 28 and 43 points.
- **On C, accuracy fell, and that's an honest finding, not a bug:**
  1. C is 80% fake. The original model's habit of saying FAKE looked like skill there, but its AUC was only 0.59 (near chance), and its balanced accuracy was 0.55.
  2. Our model learned what *GAN* fakes look like. C's *face swaps* in compressed video look different, so it calls them real (286 of 306 missed).
  3. Its AUC actually went **up** (0.59 → 0.61), so its ranking didn't get worse. The 0.5 cut-off simply doesn't fit this kind of fake. This **domain gap** is the main known weakness of deepfake detectors in published research too, and it is what v2 is designed to fix.
- **For an imbalanced set like C, accuracy and F1 mislead.** Use **balanced accuracy and AUC** there.

---

## 6. How the explanation works (`explain.py`)

We do not make up reasons. Every sentence is built from measured numbers.

1. **Grad-CAM for a ViT.**
   - Take the 196 patch tokens entering the **last** transformer block.
   - Compute the gradient of (score of the verdict) minus (score of the other class) with respect to those tokens.
   - Average the gradient over each token's 768 features to get a weight per feature. Multiply by the token values, sum the features, and keep only the positive part (ReLU).
   - The result is a 14×14 map, upsampled to 224×224.
   - Why the last block's *input*? The classifier only reads the [CLS] token, and patch tokens only reach [CLS] through attention. After the last block, patch tokens no longer affect the answer, so their gradient would be zero.
2. **Regions from landmarks.** MTCNN's 5 points define ellipses for the eyes, nose and mouth. The rest of the face box is "cheeks, forehead and jaw", and everything outside it is "hair and edges".
   - For each region we report its **share of the evidence** and its **share of the picture**.
   - A region holding much more evidence than its size is called out in the summary.
3. **The summary** states the confidence, the top two regions, whether the result is close to the cut-off, and which face decided the verdict when there are several.

**What the heatmap is not.** It shows *where the evidence for the verdict came from*. It is not a pixel-perfect map of what was edited; a classifier trained on real/fake labels cannot know that. We say this on the page.

---

## 7. Smaller design decisions

| Decision | Reason |
|---|---|
| A photo is FAKE if **any** face is fake | One swapped face in a group photo is enough to make it a manipulated image |
| No face found means **no verdict** | Guessing on a face-less image would just be noise |
| Runs locally, nothing uploaded | Privacy: people test photos of themselves and friends |
| FastAPI + one HTML page (not Next.js) | One `python app.py` command, no Node.js, and it works on any teammate's laptop |
| Fonts bundled in the repo | The page looks the same offline |
| Weights not in the Git repo | About 340 MB, over GitHub's 100 MB file limit. Shared as a Release or zip instead |
| Automated tests use a tiny random ViT | They check the plumbing (labels, crops, API) without the internet or the 340 MB model |

---

## 8. Limitations and ethics

- **The verdict is only reliable on fakes like the training data.** Strong on GAN faces (A, B); weak on face-swap video (C, in v1). Newer diffusion-model images (Midjourney, DALL·E, etc.) are untested.
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
The 68% was mostly the original model's bias towards saying FAKE on a set that is 80% fake; its AUC was 0.59, near chance. Our model learned GAN fakes, and C contains face swaps, so it misses them. The AUC actually rose slightly. This is a domain gap, a known open problem, and v2 addresses it with face-swap training data, video augmentation and a tuned cut-off.

**Q12. What is a domain gap?**
When test data comes from a different distribution than the training data, such as a different generator or compression, the patterns the model learned may not exist there.

**Q13. How does Grad-CAM work on a transformer?**
See section 6. We use the gradient of the verdict score with respect to the patch tokens entering the last block, weighted by the token values. It shows which patches pushed the decision.

**Q14. Does the heatmap show what was edited?**
No. It shows where the model found the evidence for its answer. A classifier trained only on real/fake labels cannot localise edits exactly.

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
In v1 it is 0.5. In v2 it is chosen on validation data to maximise balanced accuracy, because the best cut-off depends on the class balance and the kind of fakes. Moving the cut-off only trades false positives for false negatives; only a better model reduces both.

**Q20. Why Kaggle for training and a laptop for running?**
Fine-tuning an 86M-parameter ViT on 20,000+ images needs a proper GPU; Kaggle gives an NVIDIA T4/P100 for free. Our laptops have integrated Intel Iris Xe graphics, which would take many hours or days for the same job. Running the trained model on one image takes about a second on a CPU, so the demo works on any laptop.

**Q21. What does "ViT-B/16" mean?**
B = Base size (12 layers, 768-wide, 12 heads, about 86M parameters). 16 = 16×16-pixel patches.

**Q22. What are MTCNN's three outputs?**
A face/not-face confidence, the bounding box, and five landmarks (eyes, nose, mouth corners). That is the "multi-task" part.

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
- Face detector code: `timesler/facenet-pytorch`, MIT licence.

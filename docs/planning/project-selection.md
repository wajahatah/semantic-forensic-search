# DL Semester Project — Dataset & Project Selection

**Date:** 2026-09-12
**Author:** Wajahat (with Claude)
**Status:** Decision pending — proposal due 2026-09-13 EOD

---

## 1. Hard constraints (these drive every recommendation below)

| Constraint | Value | Consequence |
|---|---|---|
| GPU (local) | **RTX 4060 Laptop, 8 GB VRAM**, CC 8.9, driver 610.62 | Dev/debug machine. Fine for data prep, small models, demo hosting. |
| GPU (office) | **Larger GPU available, no compute constraint** (2026-09-12) | Removes the 8 GB ceiling. See §1b — this changes less than it appears. |
| Execution time | **2–3 weeks** (target done ~2026-10-04) — **now the binding constraint** | One main experiment + ablations. No multi-stage pipelines. More GPU does not buy more calendar. |
| Proposal deadline | **2026-09-13 EOD** | Dataset must be verifiable *today*. |
| Deliverable | Model + report + **working demo** | Demo requirement favours single-image/single-file inference over batch offline eval. |
| Dataset rule | Public, open-licensed, **not a used/published benchmark** | Excludes Kaggle benchmarks, RWF-2000, PlantVillage, BraTS, MVTec-AD as *primary* data. |
| Python env | `torch` not installed in default Python | Setup step needed on day 1 (see §8). |

**Note on the dataset rule.** "Not used" is unprovable in the absolute. What *is* provable, and what
satisfies the instructor, is: no Kaggle presence, no Papers-With-Code leaderboard, near-zero
citations, no GitHub repos training on it. §5 gives the evidence procedure.

---

## 1b. What removing the compute ceiling actually changes

Honest assessment, plan by plan — because "unlimited GPU" is less transformative here than it sounds.

| Plan | Change | Verdict |
|---|---|---|
| **A — PlantCity** | Marginal. 10,667 images at 224px does not become a different problem on an A100. You could train ConvNeXt-Large or a ViT-L instead of ResNet50, but the *result* — the lab→field domain gap — is backbone-independent. A bigger model may even widen the gap, which is a nice extra ablation but not a new project. | **Essentially unchanged** |
| **B — UNB Urdu OCR** | **Transformative.** Unlocks fine-tuning a real vision-language model or TrOCR-large instead of a small CRNN. This turns the project from "train a recogniser" into "can an open-weights model close the gap to frontier proprietary models on a low-resource script?" — a current, publishable-shaped question. See §4 Plan B-v2. | **Upgraded to top pick** |
| **C — Bus Violence** | Compute was never the real objection. The objections that remain: it teaches the least that is new (pose/tracking/YOLO/video geometry already covered by `nexa_fight`), CC BY-**NC** licence, and the violence is acted rather than real. | **Still bottom of the list** |
| **D — Anomaly detection** | Almost none. PaDiM/PatchCore use a *frozen* backbone with a memory bank — there is barely any training to accelerate. | **Unchanged** |

**The trap to avoid:** more compute invites scope creep — "now we can also pretrain from scratch,
also do self-supervised, also try 5 backbones." In a 2–3 week window that produces a half-finished
project with no clean result. Use the extra compute to make **one** model larger and better, not to
add stages.

**Practical questions the office GPU raises** (answer before committing):
- Which GPU and how much VRAM? A100 80 GB vs RTX 3090 24 GB decides full fine-tune vs LoRA for Plan B-v2.
- Shared/scheduled, or available on demand? A queue is a schedule risk, not a compute win.
- Can the dataset be placed on it? (All candidates are open CC-licensed, so this should be fine — worth confirming with office policy anyway.)
- Is it reachable remotely, or only on-site? Determines whether overnight runs are possible.

---

## 2. The method: dataset-as-publication venues, not aggregators

The reason Kaggle datasets fail the rule is structural — Kaggle *rewards* reuse. The fix is to
source from venues where a dataset is itself the publication, gets a DOI and a CC licence, and is
typically cited only by its own authors:

| Venue | Why it works |
|---|---|
| **Nature *Scientific Data*** | Peer-reviewed data descriptors only. [2026 index](https://www.nature.com/sdata/articles?type=data-descriptor&year=2026) |
| ***Data in Brief*** (Elsevier) | High volume, companion datasets on Mendeley Data |
| **Zenodo** (CERN) | DOI-minted, open, EU research output |
| **Mendeley Data** | Elsevier repository, CC BY 4.0 common |
| **figshare / Dryad** | Same model, life-sciences heavy |
| **IEEE DataPort** | Engineering; check open-access flag per dataset |

A dataset published 3–18 months ago in one of these is public, documented, citable, and
statistically almost certainly unused. That is the whole trick, and it generalises past this
semester.

---

## 3. Candidate shortlist (9 options)

Ranked by *novelty story × demo appeal × fits-in-8GB-in-3-weeks*.

| # | Dataset | Task | Size | Licence / Date | 8 GB? | Demo appeal |
|---|---|---|---|---|---|---|
| ~~1~~ | ~~**PlantCity**~~ **— DISQUALIFIED, see §5.1** | Multi-crop disease classification, 52 classes | 10,667 real-field images | CC BY 4.0, May 2026 | Easy | Already used in a July 2026 paper |
| **2** | **Urdu Newspaper Benchmark (UNB)** | Nastaliq OCR / super-resolution | 829 paragraph imgs, 9,982 sentences, paired LR/HR | LREC **2026** | Easy | Upload scan → Urdu text |
| **3** | **Machined-workpiece borescope surfaces** | Few-shot / unsupervised defect detection | 593 high-res (2592×1944) images | *Data in Brief* **2025/26**, Zenodo | Easy | Defect heatmap overlay |
| **4** | **SMART-OM oral cavity** | Oral disease screening from smartphone photos | 2,469 intraoral images, 8 regions | figshare, *Sci Data* **2026** | Easy | Phone-camera screening app |
| **5** | **Urban tree species (Sentinel-2 + PlanetScope)** | Satellite time-series classification | 45,084 trees, 20 species | *Sci Data* **2025/26** | Trivial | Species map / spectral curve |
| **6** | **UDE Diatoms in the Wild 2024** | Fine-grained microscopy classification | Freshwater diatom images | *Sci Data*, open | Easy | Microscope image → taxon |
| **7** | **BRISC brain tumour MRI** | Segmentation **+** classification | 6,000 CE-T1w scans, radiologist-annotated | CC BY 4.0, **Feb 2026** | Moderate | Tumour mask overlay |
| **8** | **AMCD** (Bangladesh crops/flowers) | Fine-grained classification, 77 subclasses | 5,405 images | Mendeley, **June 2026** | Easy | Same as #1, less local |
| **9** | **Bus Violence** | Video violence detection | 1,400 clips, in-bus multi-camera | CC BY-**NC** 4.0, 2022 | **Tight** | Live clip → alert timeline |

### Links

1. PlantCity — https://data.mendeley.com/datasets/w8kh2xkspx/4 (DOI `10.17632/w8kh2xkspx.4`)
2. UNB — https://github.com/sameearif/urdu-newspaper-benchmark · paper https://aclanthology.org/2026.lrec-1.235/
3. Machined workpieces — https://www.sciencedirect.com/science/article/pii/S2352340925009746
4. SMART-OM — https://www.nature.com/articles/s41597-026-06954-5
5. Urban tree species — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12221675/
6. UDE Diatoms — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11604061/
7. BRISC — https://www.nature.com/articles/s41597-026-06753-y
8. AMCD — https://pmc.ncbi.nlm.nih.gov/articles/PMC13311333/
9. Bus Violence — https://zenodo.org/record/7044203 · paper https://doi.org/10.3390/s22218345

### Ruled out, and why

| Option | Reason |
|---|---|
| **RWF-2000** (the `nexa_fight` data) | Field-standard benchmark, hundreds of papers, mirrored on Kaggle. Fails the rule outright. The existing `docs/research/data-sources.md` also documents mislabelled clips in it. |
| **DCASE 2026 Task 2 / Task 4** | Brand new and not on Kaggle, but dozens of teams publish results on it every year. Fails a strict reading of the rule. |
| **Indian paddy panicle** (Mendeley 2026) | Authors already report 98.74% test accuracy — solved on arrival. No headroom for a result. |
| **Beggar face classification** (original idea) | The label is not a function of the image; no ground-truth source exists; the model could only learn ethnicity / disability / clothing proxies. Also unconsented biometric collection on a vulnerable population. Not salvageable as stated. |

---

## 4. Project plans

Every plan below is scoped to **3 weeks on one 8 GB GPU** and produces model + report + demo.
Each states a **research question**, because "I got 94% accuracy" is not a semester project.

---

### Plan A — PlantCity: the domain-gap study *(recommended)*

**Research question:** *Do models trained on lab-condition plant disease data (PlantVillage)
survive contact with real-field imagery, and what closes the gap?*

This is the key insight: PlantVillage is 99%-solved because every leaf is photographed detached, on
a uniform background, under controlled light. PlantCity was shot **in-field in Chitral and
Charsadda** — cluttered backgrounds, variable sun, occlusion, multiple leaves per frame. Published
models almost certainly collapse on it. Demonstrating that collapse and then fixing it is a genuine
contribution, and a far better report than another transfer-learning writeup.

**Experiment design**

| Arm | What | Expected |
|---|---|---|
| A0 | PlantVillage-trained model, zero-shot on PlantCity overlapping classes | Sharp accuracy drop — **the headline result** |
| A1 | Train from ImageNet init on PlantCity directly | Solid baseline |
| A2 | A1 + strong augmentation (RandAugment, background randomisation, CutMix) | Recovers part of the gap |
| A3 | A1 + class-balanced loss (52 classes will be long-tailed) | Fixes rare-disease recall |
| A4 | Backbone comparison: ResNet50 / EfficientNet-B3 / ConvNeXt-Tiny | Ablation table |

**Architecture & 8 GB budget:** 224×224, AMP fp16, batch 32. ResNet50 ≈ 5 GB, EfficientNet-B3 ≈ 6 GB,
ConvNeXt-Tiny ≈ 6.5 GB at batch 16. All fit. A full training run is roughly 30–60 min per model.

**Critical data discipline:** split using the **10,667 original images only**. The 52,219-image
augmented version leaks — augmented copies of the same leaf would land in both train and test and
inflate every number. Split by *source image*, and preferably by *district*, before any
augmentation. State this explicitly in the report; it is exactly the kind of rigour that earns marks.

**Demo:** Gradio app — drag a leaf photo in, get top-3 diagnoses with confidence plus a Grad-CAM
heatmap showing where the model looked. Grad-CAM doubles as report material: on A0 you will likely
see attention on background rather than lesion, which *visually proves* the domain-gap claim.

**Timeline**

- **Days 1–2:** env setup, download, integrity check, leak-free stratified splits, EDA (class balance, image size distribution, duplicate detection)
- **Days 3–6:** A1 baseline + training/eval harness + metrics (macro-F1, per-class recall, confusion matrix)
- **Days 7–10:** A0 zero-shot transfer experiment (the headline) + Grad-CAM analysis
- **Days 11–15:** A2 / A3 / A4 ablations
- **Days 16–18:** Gradio demo
- **Days 19–21:** report, figures, buffer

**Risks:** Low. The main one is that PlantVillage↔PlantCity class overlap is smaller than hoped —
check the class lists on day 1, and fall back to "in-field is harder than lab" framed via a
controlled-background subset if needed.

---

### Plan B — UNB: does super-resolution actually help Urdu OCR?

**Research question:** *The UNB authors report a SwinIR super-resolution module improved Nastaliq
OCR accuracy by ~50%. Does that hold — and is SR actually the mechanism, or would the same gain
come from training the recogniser on low-res inputs directly?*

That second clause is what makes this a project rather than a reimplementation. SR-before-OCR is a
widely repeated claim that is often confounded: an SR model trained on the same domain effectively
acts as a denoiser/adapter, and the fair comparison — **an OCR model fine-tuned on low-res input
with no SR at all** — is usually skipped. Testing it is a publishable-shaped result.

UNB is ideal for this because it ships **paired low-res and 4× high-res versions of every image**,
so the SR training set comes for free.

**Experiment design**

| Arm | Pipeline | Purpose |
|---|---|---|
| B0 | OCR on HR images | Upper bound |
| B1 | OCR on LR images (HR-trained recogniser) | Naive lower bound |
| B2 | LR → SR → OCR | Reproduces the paper's claim |
| B3 | **OCR fine-tuned on LR directly** | The missing control — does SR earn its place? |
| B4 | B3 + LR/HR mixed training | Best of both |

**Architecture & 8 GB budget:**

- *Recogniser:* CRNN (CNN + BiLSTM + CTC) on line crops, 32×W grayscale. Tiny — under 2 GB, trains in minutes. Nastaliq is cursive and context-sensitive, so use **CTC over full lines, never character segmentation**.
- *SR:* train on 48×48 patches, batch 8 — a compact SwinIR or EDSR/RCAN fits in 8 GB. Do **not** attempt full-size SwinIR training.
- *Layout (optional):* YOLOv11n for text-block detection — YOLO experience already exists in `nexa_fight`, which has `yolo11l-pose.pt` on disk.

**Metrics:** CER and WER. Published baselines to position against: TrOCR 0.422 WER, Gemini-2.5-Pro
0.133 WER. For SR itself, PSNR/SSIM — and show that PSNR gain and OCR gain **decorrelate**, which is
itself a nice finding.

**Demo:** Gradio — upload a newspaper scan → detected text blocks → SR-enhanced crops → Urdu text,
shown side-by-side with the no-SR path so the comparison is visible in the demo itself.

**Timeline**

- **Days 1–3:** setup, download, line segmentation from paragraph images, Urdu charset + Unicode normalisation (fiddly for Urdu — budget real time)
- **Days 4–8:** CRNN + CTC recogniser, B0 / B1
- **Days 9–13:** SR model, B2
- **Days 14–16:** B3 / B4, the controls
- **Days 17–19:** demo
- **Days 20–21:** report

**Risks: medium-high.** 829 paragraph images is small for training a recogniser from scratch — you
will likely need synthetic Urdu line rendering (render Urdu corpus text in Nastaliq fonts) to
pretrain, then fine-tune on UNB. That is an extra sub-task. CTC training is also finicky to
converge. Do not pick this unless you are comfortable debugging a loss that sits flat for a while.

---

### Plan B-v2 — UNB with a real GPU: open-weights vs frontier models on Nastaliq

*This supersedes Plan B now that compute is available. It is the strongest project on the list.*

**Research question:** *Can a fine-tuned open-weights model close the gap to proprietary frontier
models on low-resource-script OCR?*

The UNB paper hands you the entire experimental frame on a plate:

| System | Reported WER |
|---|---|
| Gemini-2.5-Pro (proprietary, zero-shot) | **0.133** |
| GPT-4o fine-tuned on 500 UNB samples | 0.269 |
| TrOCR (open, traditional OCR) | 0.422 |

So there is a **published, citable gap** between the best open system (0.422) and the best
proprietary one (0.133). Closing or narrowing that gap with an open model you fine-tuned yourself is
a genuine, defensible contribution — and it is exactly the kind of question the field cares about
right now. It also has a clean negative result: if you *can't* close it, quantifying why (script
complexity? data scarcity? tokeniser?) is still a real finding.

**Experiment design**

| Arm | System | Purpose |
|---|---|---|
| V0 | TrOCR zero-shot | Reproduce the published open baseline (0.422) |
| V1 | TrOCR-large fine-tuned on UNB | How far does plain fine-tuning get? |
| V2 | Open VLM (e.g. Qwen2.5-VL-7B) LoRA fine-tuned | The main arm — target < 0.269 |
| V3 | V2 + synthetic Nastaliq pretraining | Addresses the 829-image data ceiling |
| V4 | V2/V3 + super-resolution front-end | Does the SR claim still hold for a *modern* recogniser, or was it only compensating for a weak one? |

V4 is the nice twist: the original SR finding may simply be an artefact of a low-capacity
recogniser. A strong model may not need SR at all — that's a clean, interesting result either way.

**Compute notes**
- 7B LoRA fine-tune: ~24 GB VRAM is comfortable; full fine-tune wants 80 GB. **Confirm the office GPU's VRAM before choosing.**
- Synthetic data (V3) is the highest-leverage use of the extra compute: render a large Urdu corpus in multiple Nastaliq fonts with realistic degradation, pretrain, then fine-tune on UNB's 829 real images. This directly attacks the data-scarcity risk that was Plan B's weak point.
- Keep the RTX 4060 for data prep, evaluation and demo hosting; use the office GPU only for training runs.

**Metrics:** CER and WER against the three published baselines above. Report per-error-type analysis
(Nastaliq ligature errors vs diacritics vs layout) — that is what makes the report substantive
rather than a leaderboard entry.

**Demo:** Gradio — upload a newspaper scan → layout detection → recognised Urdu text, with a
side-by-side comparison panel showing your model vs the TrOCR baseline on the same crop.

**Risks: medium.** Lower than plain Plan B, because synthetic pretraining solves the data-ceiling
problem and LoRA is far more stable to train than CTC from scratch. Remaining risks: VLM fine-tuning
tooling has sharp edges; Urdu Unicode normalisation and right-to-left rendering will eat a day; and
the evaluation must exactly match the paper's WER definition or the comparison is meaningless
(verify against their repo before reporting anything).

---

### Plan C — Bus Violence: reuse the `nexa_fight` pipeline

**Research question:** *Does a pose-and-geometry feature pipeline generalise to a moving-vehicle
camera, where global ego-motion corrupts every motion feature — and does ego-motion compensation
recover it?*

The in-bus domain is interesting precisely because the camera itself shakes and translates. Every
velocity/acceleration feature in the existing `features.py` is measured in image coordinates, so bus
motion injects a shared bias into every track. Detecting that and compensating for it (estimate
global motion from background keypoints, subtract) is a real engineering result, and it feeds
directly back into `nexa_fight`.

**Experiment design**

| Arm | What |
|---|---|
| C0 | End-to-end video classifier baseline (X3D-S or R(2+1)D-18, 16 frames @ 112px) |
| C1 | The existing pose-based pipeline, ported as-is |
| C2 | C1 + ego-motion compensation |
| C3 | Cross-domain: train on Bus Violence, test on `cctv_videos` and vice versa |

**8 GB budget:** X3D-S at 16×112×112, batch 4–8 with AMP fits but is **slow** — the heaviest of the
plans. Pose extraction over 1,400 clips is a multi-hour one-time cost; cache keypoints to disk on
day 1 and never re-run it.

**Demo:** video player with a live violence-score timeline and alert markers — visually the most
impressive of the three.

**Timeline:** Days 1–3 download + pose extraction (run overnight). Days 4–7 C0. Days 8–12 C1 port.
Days 13–16 C2. Days 17–19 C3 + demo. Days 20–21 report.

**Risks: highest.** Video is the most compute-hungry option on the tightest GPU; CC BY-**NC** means
non-commercial only (fine for coursework, blocks later product use); and the clips are *simulated*
violence by actors — a limitation that must be stated in the report. An instructor may also
associate "violence detection" with RWF-2000-style projects, so lead with the novelty evidence.

---

### Plan E — SMART-OM: multimodal oral cancer screening *(the new safe option)*

*This replaces PlantCity as the low-risk choice after PlantCity was disqualified in §5.1.*

**Research question:** *Does combining smartphone images with patient risk metadata (tobacco,
areca nut, alcohol) beat image-only screening for oral potentially-malignant disorders — and does it
help most exactly where images alone are weakest?*

This is better than PlantCity was, because the dataset ships something unusual: **2,469 expert-annotated
intraoral images from 331 subjects across 4 classes** (healthy / variation-from-normal / OPMD / oral
cancer) **plus structured patient metadata** — age, sex, smoking, smokeless tobacco, alcohol, areca
nut chewing. That makes a genuine **multimodal** project rather than another image classifier, and
multimodal fusion is a more valuable skill than yet another CNN fine-tune.

The clinical framing is strong too: oral cancer incidence linked to areca nut and smokeless tobacco
is a major burden in Pakistan and South Asia, and smartphone-based screening is exactly the
low-resource-setting use case. The descriptor reports macro-AUC 0.92 / micro-AUC 0.97, so you have a
baseline to position against.

**Experiment design**

| Arm | What |
|---|---|
| E0 | Image-only CNN/ViT baseline — reproduce the descriptor's ~0.92 macro-AUC |
| E1 | Metadata-only (gradient boosting on risk factors) — how much is in the metadata alone? |
| E2 | **Late fusion** (image logits + metadata) |
| E3 | **Joint/attention fusion** — the main arm |
| E4 | Subgroup analysis: where does fusion help most? Per-region (8 intraoral sites) and per-risk-group breakdown |

E1 is the important control, and the one most people skip: if metadata alone gets most of the way
there, your "multimodal" gain is not about the images at all. Reporting that honestly is a result.

**Critical discipline:** **split by subject, not by image.** 2,469 images from 331 subjects means
~7 images per patient — a random image-level split puts the same patient's mouth in train and test
and inflates everything. This is the single most common error in small medical imaging projects.

**Demo:** Gradio — upload an intraoral photo + tick risk factors → risk score with Grad-CAM overlay
and an explicit "this is a screening aid, not a diagnosis" disclaimer.

**Risks: low-medium.** Class imbalance (cancer will be the rare class), and 331 subjects is small —
report confidence intervals over multiple seeds, and use subject-level cross-validation rather than a
single split.

---

### Plan D — Machined-workpiece defects: unsupervised anomaly detection *(dark horse)*

Worth flagging even though it was not on the original list, because **593 images is a feature, not a
bug.**

**Research question:** *With only a few hundred images and no defect labels at training time, does
unsupervised anomaly detection beat supervised classification?*

You train on **normal surfaces only** and flag anything that deviates. PaDiM/PatchCore use a frozen
pretrained backbone plus a memory bank, so there is barely any training at all and it runs
comfortably in 8 GB. This is a more interesting DL topic than another fine-tune, it matches how
industrial inspection actually works (defects are rare; you cannot collect labels for them), and it
teaches a skill set nobody else in the cohort will have.

**Demo:** upload a surface image → per-pixel anomaly heatmap + pass/fail score.

**Risk:** medium — small dataset means noisy results, so report confidence intervals over multiple
seeds. The 2592×1944 resolution needs tiling.

---

### Plan F — Cost-bounded semantic forensic search over surveillance alerts

*Added 2026-09-14 from Wajahat's own idea. Currently the strongest candidate — see §7.*

**The idea, restated:** thousands of alerts accumulate from a surveillance system. An investigator
asks in plain language — *"find the man in the coffee shop who spiked someone's drink"* — and the
system returns the matching clips, without running an expensive vision-language model over every
frame in the archive.

**Research question:** *How much of the retrieval accuracy of a VLM-over-every-candidate system can
be retained while cutting VLM calls by one to two orders of magnitude?*

#### Why this is well-aimed

The instinct "semantic search first, keep LLM usage optimised" is the published **cascaded
retrieve-then-rerank** architecture: a dual encoder (CLIP/SigLIP/BEiT-3) does cheap first-pass
retrieval over the full corpus optimising *recall*; an expensive cross-encoder (BLIP-2 ITM, or a VLM)
re-ranks only a small candidate set, optimising *precision*. Cross-encoders cannot run at collection
scale because cost grows linearly in pairs. That is exactly the bottleneck this design targets.

**The gap in the literature.** Related systems exist — SurveiLens, ForeSea, EcoSurve, SOVABench,
TRACES, VadCLIP — but they overwhelmingly report Recall@K only. Very few report **VLM calls per
query, latency, or dollar cost**. A *recall-vs-cost Pareto curve* is under-measured and is precisely
the axis this project is built around.

#### The critical design decision: this must be a DL project, not a systems project

Frozen CLIP + a vector DB + an LLM API means **nothing was trained**. An instructor can reasonably
reject that for a deep learning course. The cost optimisation itself must be the learned component:

| ID | Trained component | Why it counts |
|---|---|---|
| **D1** | **Domain-adapt the image encoder** to CCTV imagery via contrastive fine-tuning on auto-captioned pairs | CLIP is trained on web images; surveillance is badly out-of-distribution. Real, measurable gap. |
| **D2** | **Distil the VLM re-ranker into a small cross-encoder** — VLM labels candidate relevance, small model learns to imitate it | **The core contribution.** Knowledge distillation is a classic DL topic and it *is* the cost optimisation, not a bolt-on. |
| **D3** | **Escalation router** — given the query and the retrieval score distribution, predict whether VLM re-ranking would change the top result | Cheap to train, directly cuts calls, produces the adaptive-budget story |

D2 is the headline. D1 is the supporting result. D3 is upside if time allows.

#### Architecture

```
 alerts/clips ──> [Stage 0] event extraction (reuse nexa_fight, or person/motion detection)
              ──> [Stage 1] image encoder (D1 fine-tuned) ──> FAISS / Qdrant index
 NL query     ──> text encoder ──> ANN search ──> top-K candidates
              ──> [Stage 3] router (D3) decides escalation budget
              ──> [Stage 2] distilled re-ranker (D2)  ──┐
              ──> VLM re-rank (only the escalated few) ─┴─> ranked clips
```

#### Evaluation

| Axis | Metrics |
|---|---|
| Accuracy | Recall@1 / @5 / @10, MRR, nDCG |
| **Cost** | VLM calls per query, tokens per query, $ per query, end-to-end latency |
| **Headline figure** | **Recall-vs-cost Pareto curve** across: frozen CLIP only · +distilled re-ranker · +VLM on top-5 · +VLM on top-50 · VLM on everything (oracle upper bound) |

Reference points that must be in the report: the **VLM-on-everything upper bound** (what accuracy
costs if you don't optimise) and the **frozen-CLIP-only lower bound** (what you get for free).
The contribution is the shape of the curve between them.

#### Dataset plan — this project dissolves the novelty problem

Standard text-video retrieval benchmarks (MSR-VTT, DiDeMo, Charades-STA, ActivityNet Captions) are
heavily used and all fail the rule. Instead:

1. **Corpus:** an openly-licensed surveillance video collection. Candidates to verify: **MEVA**
   (DIVA program, real multi-camera surveillance, openly released) and **VIRAT** (public,
   government-funded). *Licence and prior-usage both still need the §5 checks run — do this first.*
   Wajahat's own `cctv_videos` footage can supplement.
2. **Query set — the novel artefact:** auto-caption events with a VLM, then **hand-verify 300–500
   queries** with ground-truth clip matches. This evaluation set does not currently exist, so it is
   novel by construction, and building it is itself a claimable contribution.
3. **Training pairs for D1/D2:** generated automatically by the VLM at scale (cheap, no human
   labelling), with the hand-verified set held out strictly for evaluation.

**Discipline:** the hand-verified test queries must never touch training, and the VLM used to
generate training labels must not be the same VLM used as the evaluation upper bound — otherwise you
are measuring self-agreement, not accuracy.

#### Why this fits Wajahat specifically

It **reuses `nexa_fight` legitimately**. The existing pipeline becomes the Stage 0 alert generator;
the new deep learning work is the retrieval and distillation stack. Months of existing engineering
carries over *without* inheriting RWF-2000's disqualification, because the evaluation is on the new
retrieval benchmark, not on RWF-2000's fight labels.

#### De-risking: the guaranteed minimum result

The stated worry is getting stuck. Scope it so a passing project exists by end of week 1:

| Week | Deliverable | Status if everything after fails |
|---|---|---|
| **1** | Frozen CLIP + FAISS retrieval working end-to-end, Recall@K measured on a small query set | **This alone is a complete, demo-able project.** Passing floor secured. |
| **2** | D1 domain adaptation + expanded verified query set | Solid project |
| **3** | D2 distillation + Pareto curve + demo + report | Strong project |
| *stretch* | D3 router, temporal localisation within clips | Excellent project |

Nothing after week 1 is load-bearing for passing. That is the answer to "what if I get stuck."

**Risks**
- **Scope creep** — the biggest one. Retrieval + distillation + routing + temporal localisation is four projects. Ship the cascade and the Pareto curve; treat everything else as optional.
- **Corpus licensing/novelty unverified** — MEVA and VIRAT still need the §5 checks. Do this before committing.
- **VLM API cost** — budget it; use an open-weights VLM locally on the office GPU rather than a paid API if cost is a concern. With unconstrained compute this is the sensible default anyway.
- **Not-a-DL-project objection** — fully mitigated by D1/D2, but the proposal must foreground them.

---

## 5. Novelty verification — RESULTS (run 2026-09-12)

These checks were actually run, not templated. **One candidate failed.**

### 5.1 PlantCity — ❌ FAILS the novelty rule

| Check | Result |
|---|---|
| Kaggle | Not present ✅ |
| Papers With Code | No leaderboard ✅ |
| **Third-party published use** | ❌ **"Explainable Transfer Learning for Multi-Crop Plant Disease Identification Under Real-Field Conditions"**, MDPI, **28 July 2026** — custom CNN + DenseNet-121 trained across all 12 PlantCity crop subsets |
| Possible second user | "PlantaNet / PlantaNetLite", *PLOS One* 2026 — evaluated on a curated multi-crop set of **51 disease and healthy classes**, suspiciously close to PlantCity's 52 |

**Verdict: disqualified.** A paper published six weeks ago has already done transfer learning on this
exact dataset — which is approximately the project I originally proposed. This is precisely what the
instructor's rule excludes. *This overturns the original §7 recommendation.*

### 5.2 UNB (Urdu Newspaper Benchmark) — ✅ PASSES

| Check | Result |
|---|---|
| Kaggle | Not present ✅ |
| Papers With Code | No leaderboard ✅ |
| Hugging Face datasets | Not present ✅ (search returns *different* Urdu sets — UrduDoc, UTRNet) |
| Third-party published use | **None found** ✅ — only the authors' own LREC 2026 paper |
| Authors' own baselines | Published on GitHub ✅ — *this is an advantage, not a problem: it gives you numbers to beat* |

**Verdict: cleared.** Note the correct standard here — the *field* of Urdu OCR is not empty
(UrduDoc and UTRNet exist), but **this dataset** has no third-party usage. That is exactly what the
rule asks for, and the distinction is worth stating explicitly in the proposal.

### 5.3 SMART-OM (oral cavity) — ✅ PASSES

| Check | Result |
|---|---|
| Kaggle / PwC / HF | Not present ✅ |
| Third-party published use | None found ✅ — only the *Scientific Data* 2026 descriptor + its Research Square preprint |
| Baseline in descriptor | macro-AUC 0.92, micro-AUC 0.97 — a number to beat ✅ |
| Bonus | Ships **patient metadata** (age, sex, smoking, smokeless tobacco, alcohol, areca nut) → enables a multimodal image+tabular project |

### 5.4 Machined workpieces — ✅ PASSES

| Check | Result |
|---|---|
| Kaggle / PwC | Not present ✅ |
| Third-party published use | None found ✅ — only the *Data in Brief* descriptor |
| Detail | 593 images, 2592×1944, 4 piece categories (Piece01–04), binary defect labels, inner **and** outer surfaces |
| To check | A separate "Workpieces image dataset" exists at https://zenodo.org/records/16361102 — confirm whether it is the same release |

### 5.5 Evidence to reproduce for the proposal

For whichever dataset is chosen, screenshot each of these, dated:

| Check | How |
|---|---|
| DOI + licence | Dataset landing page |
| Citation count | Google Scholar "Cited by" on the descriptor paper |
| Code usage | GitHub code search for the dataset name |
| Leaderboard | paperswithcode.com search |
| Kaggle | Kaggle dataset **and** notebook search |

Wording to include: *"The dataset is open-licensed and published as a peer-reviewed data descriptor.
A systematic search of Kaggle, Papers With Code, Hugging Face, GitHub and Google Scholar (conducted
2026-09-12) found no trained models, leaderboards, or third-party published results for it beyond the
descriptor paper itself."*

**Lesson worth carrying:** run these checks *before* committing to a dataset, not after. PlantCity
looked perfect on every axis except the one that actually mattered.

---

## 6. Skills matrix — what each plan actually teaches

| Skill | A: PlantCity | B: Urdu OCR/SR | C: Bus Violence | D: Anomaly detection |
|---|---|---|---|---|
| CNN transfer learning / fine-tuning | ●●● | ●● | ●● | ● |
| **Domain shift & generalisation** | ●●● | ●● | ●●● | ●● |
| Leak-free experimental design | ●●● | ●● | ●● | ●● |
| Long-tail / class-imbalance handling | ●●● | ● | ● | — |
| Model interpretability (Grad-CAM) | ●●● | ● | ● | ●●● |
| **Sequence modelling (RNN / CTC)** | — | ●●● | ● | — |
| Generative / restoration models (SR) | — | ●●● | — | — |
| Multilingual + Unicode text handling | — | ●●● | — | — |
| **Video / temporal modelling (3D CNN)** | — | — | ●●● | — |
| Pose estimation & multi-object tracking | — | ● | ●●● | — |
| Geometry / ego-motion compensation | — | — | ●●● | — |
| **Self-supervised / unsupervised learning** | — | — | ● | ●●● |
| Object detection (YOLO) | — | ●● | ●● | — |
| Ablation discipline & result reporting | ●●● | ●●● | ●●● | ●●● |
| Deployment / demo engineering (Gradio) | ●●● | ●●● | ●●● | ●●● |

**Reading it:**

- **A** = breadth of *good practice* — domain shift, leakage, imbalance, interpretability. Lowest risk, best schedule fit.
- **B** = depth in *sequence + generative* modelling. Most technically impressive, rarest skills, highest chance of overrunning.
- **C** = depth in *video + geometry*; compounds with `nexa_fight`. Heaviest compute.
- **D** = depth in *unsupervised* learning. Least common skill set in a DL cohort.

**Key point:** pose, tracking, YOLO and video geometry are already covered by `nexa_fight`, so **C
teaches the least that is new.** A, B and D each add a genuinely different axis.

---

## 7. Recommendation

**Revised 2026-09-14** — Plan F (Wajahat's own semantic-search idea) is now the primary
recommendation. Previously revised twice on 2026-09-12: once when the compute constraint was lifted,
once when novelty verification (§5) disqualified PlantCity.

### On "safe project to pass" vs "ambitious project"

The framing that the violence-detection route is the *safe* option is **incorrect, and worth stating
plainly**:

- **RWF-2000 is disqualified.** It is on Kaggle, it is the field-standard benchmark with hundreds of published papers, and it was the first dataset ruled out (§3). It was never cleared.
- The only surviving violence dataset is **Bus Violence**, which ranked *last* overall — heaviest compute, CC BY-**NC**, acted rather than real violence, and it overlaps most with work already completed in `nexa_fight`, so it teaches the least.

So the violence route is not low-risk-and-quick; it is low-reward *and* carries a live novelty
problem. The genuine trade-off is between **Plan F** (higher ceiling, needs disciplined scoping) and
**Plan E** (lower ceiling, near-certain clean finish) — not between "pass" and "ambitious."

### Primary: Plan F — cost-bounded semantic forensic search

1. **Novelty by construction** — the evaluation query set does not exist yet; you build it, so the dataset rule is satisfied structurally rather than by search-and-hope
2. **Reuses `nexa_fight` legitimately** — the existing pipeline becomes the alert generator, without inheriting RWF-2000's disqualification
3. **Real contribution in an under-measured gap** — related systems report Recall@K; almost none report cost per query. The recall-vs-cost Pareto curve is the niche.
4. **Guaranteed passing floor by end of week 1** (see Plan F de-risking table) — which directly answers the "what if I get stuck" worry
5. **Uses the unconstrained GPU** — local open-weights VLM for label generation and distillation
6. **Genuine DL content** via D1 (domain adaptation) and D2 (distillation) — provided the proposal foregrounds them

**Condition:** the proposal must lead with the *trained* components. A frozen-CLIP-plus-vector-DB
framing invites rejection as a systems project rather than a deep learning one.

### Runner-up: Plan B-v2 (UNB — open-weights vs frontier models on Nastaliq).

It now wins on every axis that matters:

1. **Novelty verified clean** (§5.2) — no Kaggle, no PwC, no HF, no third-party use
2. **Three published baselines to beat** — Gemini-2.5-Pro 0.133, GPT-4o-ft 0.269, TrOCR 0.422. The report has a result either way.
3. **Uses the compute you now have** — synthetic Nastaliq pretraining plus VLM fine-tuning is the one plan where an unconstrained GPU genuinely changes what's possible
4. **Locally grounded** — Urdu, Pakistani newspapers
5. **Rarest skills** — sequence modelling, VLM fine-tuning, multilingual/Unicode handling

**Safe fallback: Plan E (SMART-OM).** Take this if the timeline tightens or VLM tooling turns out to
be a fight. It is *better* than PlantCity was — the patient metadata makes it genuinely multimodal
rather than another image classifier, the clinical framing (areca nut / smokeless tobacco oral cancer
burden in Pakistan) is strong, and novelty is verified clean.

**Plan D** remains the dark horse — rarest skill of all, but the extra compute does nothing for it.

**Plan C is last.** Compute was never its real problem: it overlaps most with `nexa_fight`, the
licence is non-commercial, and the violence is acted.

**Plan A is withdrawn** — see §5.1.

**Decision rule:** default to **B-v2**. Switch to **E** if the schedule slips or you want a guaranteed
clean finish.

---

## 8. Day-1 setup (do this regardless of choice)

`torch` is not in the default Python on this machine. Before anything else:

1. Create a venv; install PyTorch with the CUDA build matching driver 610.62
2. Verify `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)`
3. Confirm AMP works with a dummy forward/backward at the target batch size — find the real VRAM ceiling *empirically* before designing the experiment around it
4. Download the dataset, verify checksums, count files per class
5. Build the split **before** writing any model code

---

## 9. Open questions

1. **Does the instructor approve the dataset before the proposal, or is the proposal itself the approval?** Changes how much novelty evidence goes in.
2. **Solo or team?** Plans B and C are much more tractable split two ways.
3. **Required report format** — page count, IEEE template?
4. **Any constraint on framework** (PyTorch vs TensorFlow) or on using pretrained ImageNet weights? Plan A's headline experiment depends on transfer learning being allowed.

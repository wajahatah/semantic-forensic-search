# Spec — VLM Teacher, Distilled Temporal Re-ranker, and the Recall-vs-Cost Curve

**Date:** 2026-10-31
**Status:** **Approved for build 2026-10-31** (§10) · **Complete 2026-10-31** → `docs/results/04-cascade.md` (G1 failed; open questions for the owner in `data/review/vlm_feedback.html`)
**Depends on:** `spec-lora-finetune.md` (complete), `docs/results/03-lora.md`
**Design origin:** `project-selection.md` Plan F (D2 distillation = core contribution; headline =
recall-vs-cost Pareto curve), `spec-retrieval-system.md` §7 (query-time cascade)
**Log:** approval D-022; gates D-023; results D-024

---

## 1. What this change covers

The cascade's two remaining stages, and the project's headline figure:

1. **VLM scorer.** A small vision-language model (VLM) judges whether a clip matches a text query.
   It is the expensive stage at query time and the **teacher** for distillation.
2. **Feasibility and validation gate.** Before the VLM labels anything at scale, measure on the
   laptop whether it fits and runs fast enough (G0), and whether it adds information on validation
   that the LoRA retriever lacks (G1).
3. **Teacher labelling.** Score every training and validation instance once, offline.
4. **Student re-ranker.** A small temporal model over each candidate's frame sequence. It sees frame
   order and box motion, which per-frame retrieval cannot. It is trained from human labels,
   teacher scores, or both, and those three variants are compared.
5. **Cascade evaluation.** The recall-vs-cost curve on eval. Cost is VLM calls per query.

**Out of scope** (later): escalation router (Plan F D3), fine-tuning the VLM itself, demo / alert
ingestion, image + text LoRA.

---

## 2. State before this change

- **Retriever.** LoRA-adapted SigLIP 2 (D-021). Eval Q1 mAP: A **0.313**, B **0.280**. Q2 on
  Protocol A: S@1 0.389, S@10 0.902. **0 VLM calls.**
- **Weakest activities.** Motion-defined ones (pick up, put down, stand up, sit down, transfer).
  Pick up and put down reach only lift ≈ 2.
- **Available locally:**
  - Per-frame embeddings: LoRA tight, 73,569 frames per run; frozen SigLIP 2 tight / context / full,
    220,707 frames.
  - The crop store: tight PNG and context JPEG up to 448 px, for every sampled frame.
  - MEVA geometry annotations: 315 files, giving per-frame actor boxes.
- **Instances:** 10,678 in total. Validation: 739 (A) / 586 (B). Eval: 3,726 / 3,938. Q2 windows
  (camera × hour): 54 / 29 on eval, 12 / 16 on validation.
- **Missing:** no VLM code, no VLM weights, `bitsandbytes` not installed. transformers 5.20 includes
  the Qwen3-VL, Qwen3.5, InternVL, Gemma 4 and MiniCPM-V 4.6 model code (checked locally).
- **Hardware and network:** RTX 4060 with 8 GB; 59 GB disk free. Download speed measured today:
  ~1.7 MB/s (Cloudflare test). Hugging Face answers anonymous `curl` with HTTP 429; a Hugging Face
  login token is present on the laptop.

---

## 3. Decisions

### D1 — The VLM: two 2B candidates, run in bf16 on the laptop
Candidates, chosen from a 2026-10-31 survey of models that fit 8 GB:

| Model | Download | Why |
|---|---|---|
| **Qwen3-VL-2B-Instruct** | 4.3 GB | General VLM; video input; Apache-2.0; native in transformers 5.20 |
| **Qwen3-VL-Reranker-2B** | 4.3 GB | Same architecture, but trained as a (query, document) relevance scorer with a yes/no head. It does exactly this job "off the shelf" |

- **Why these two:** both run in bf16 with no quantisation (estimated 5–6 GB peak; G0 measures
  it). They share the same code, so comparing them isolates whether retrieval-specific training
  helps on MEVA.
- **Parked:** Qwen3.5-4B (in 4-bit NF4) and InternVL3.5-2B. These would need a 9.3 GB download plus
  `bitsandbytes`, or a second model family. Use them only if G1 is marginal, and only with your
  approval.
- **Revision pinning:** the model revision is pinned to a commit hash at download, not "unpinned".

### D2 — How the VLM scores a (clip, query) pair
- **Input:** 8 frames per instance, evenly spaced over its sampled frames (≤ 16 at 2 fps), passed in
  time order as a video.
- **View:** tight crop or context crop (≤ 448 px). The view is **chosen on validation** (D4).
  Context should help vehicle, door and person-person activities; tight should help small hand
  actions.
- **Prompt** (fixed now, never tuned on eval): *"Does this video show the following: {query}?
  Answer yes or no."* The Reranker uses its own documented instruction format. Any
  "thinking" mode is off.
- **Score:** log p("yes") − log p("no") for the next token after the prompt. One forward pass, no
  text generation. This is the VQAScore method (Lin et al. 2024) written as a margin;
  σ(margin) = p_yes / (p_yes + p_no).
- **Calibration:** VLMs say "yes" too often and the bias varies per query. Within one query the
  ordering is what matters. Wherever VLM scores are fused with other scores, they are
  z-normalised **per query** first.
- **Efficiency:** the frames come first in the prompt and the question last. The computed state for
  the frames (the KV cache) is reused for every question about that clip, so a clip is encoded once
  however many queries ask about it. A test checks that cached and uncached scores match. The
  per-call latency reported on the cost curve is measured **without** this sharing, i.e. the
  honest query-time cost.

### D3 — G0: laptop feasibility gate (first step after approval)
Run on 20 validation clips.

**Pass if:**
- peak VRAM ≤ 7.5 GB;
- cached scoring ≤ 3 s per clip for 64 questions;
- re-scoring gives identical margins.

**If it fails, in this order:**
1. 6 frames instead of 8;
2. lower resolution;
3. a smaller model (Qwen3.5-0.8B, about 1.6 GB);
4. **only then** ask for the office GPU (your instruction).

### D4 — G1: the VLM must add information, measured on validation only
Run on validation only, for both protocols, using the 15 / 17 activities with ≥ 5 validation
instances (the frozen eval phrasings, as in LoRA selection).

- **Configurations (4):** 2 models × 2 views. Nothing else is searched, because validation is small.
- **Measure:**
  - VLM-alone validation Q1 mAP;
  - **fusion gain**: z(VLM) + w·z(LoRA), with w chosen from {0.25, 0.5, 1, 2} on validation,
    compared with LoRA alone (A 0.437, B 0.378).
  - Per-activity results are reported as well.
- **Pass:** fusion gain ≥ **+0.02 validation mAP on at least one protocol**, i.e. the VLM knows
  something the retriever does not. The best configuration is then the teacher, recorded before
  teacher labelling starts.
- **Fail:** stop and report to you before any large labelling run. The options would be:
  - (a) continue with the human-label student only; the curve then shows the VLM's measured
    (negative) value, which is still a result;
  - (b) try a parked model (D1);
  - (c) fine-tune the VLM (new spec).

  I would not choose among these myself.

**What the literature already suggests.** SOVABench (2026, built from MEVA/VIRAT vehicle clips)
found VLMs near chance at telling opposite directions apart (e.g. turn left vs right). Expect the
VLM to be weak on `vehicle_turns_*` and possibly on the small hand actions. See H5.

### D5 — Teacher labelling (offline, once, resumable)
- **Training-side labels.** Every instance that is train or validation under either protocol is
  scored against **2 training captions per class** (64 texts, from `train_captions.csv`; never the
  eval phrasings). File: `teacher_trainval.npz`.
- **Eval-side scores.** Every eval instance (union of A-eval and B-eval) is scored against the 66 eval
  phrasings, used **only** to build the VLM points of the curve in Phase 4. File: `vlm_eval.npz`.
  The student code path cannot open it; a test enforces this.
- **Volume and time:** ~10.7k clips at an estimated 1.5–2.5 s each, i.e. ~5–7 h on the laptop.
  Work is written in resumable parts, as for embedding and re-embedding.
- **Identity stored:** model id, pinned revision, prompt hash, view, frame count, crop-store id.
- **Leakage note:** the teacher is never trained, so scoring an instance that is train for A and
  eval for B leaks nothing. The **student** follows D8 of the LoRA spec: student A uses only A-train
  instances and student B only B-train.

### D6 — The student: a small temporal re-ranker over the frame sequence

**Input per (query, candidate):** the candidate's time-ordered frames (≤ 16). Each frame contributes:
- its LoRA tight embedding (768);
- its frozen SigLIP 2 context embedding (768);
- **box-motion features** from the MEVA geometry: actor box centre, width and height (normalised by
  frame size), the change since the previous frame, and the actor count.

The query is represented by its frozen SigLIP 2 text embedding.

**Why box motion:** it is cheap, and it directly carries what the weak activities are defined by:
- standing up / sitting down change box height;
- turns change the trajectory;
- entering / exiting change position relative to the frame.

Boxes come from annotations, exactly as the crops already do; in deployment both would come from
the tracker.

**Architecture:**
- each frame is projected to 256 dimensions, plus a learned position for frame order;
- a query token goes in front of the frames;
- 2 transformer layers, 4 heads;
- the query token's output is the relevance logit.

That is ~1.5–2 M parameters, under 1 ms per candidate on the GPU.

**Stage-1 score is not an input.** The LoRA score is fused **after** the student:
final = z(student) + β·z(LoRA), with β chosen on validation from {0, 0.25, 0.5, 1}. The reason:
LoRA scores on its own training instances are over-confident, and a student that learned to trust
them would fail on eval.

### D7 — Student training targets: human labels vs teacher vs both (3 variants)
- **Training data:** queries are the training captions. Candidates for a query are positives of its
  class plus negatives: half hard (highest LoRA score among other classes), half random.
- **Loss:** pointwise BCE with target y = α·(human label) + (1 − α)·σ(teacher margin).
- **Variants:**

  | Variant | α | Learns from |
  |---|---|---|
  | **GT** | 1 | Human labels only — the honest baseline for distillation |
  | **GT+KD** | 0.5 | Both |
  | **KD** | 0 | Teacher only — measures how much of the VLM's judgement a ~2 M-parameter model recovers |

- **Runs:** each variant is trained per protocol, so **6 runs**, each a few minutes.
- **Training settings:** AdamW, lr 3e-4, ≤ 50 epochs, early stopping on validation (patience 5).
  Same crash-safe checkpoints and resume as LoRA (D13).
- **Selection:** per protocol by validation Q1 mAP of the fused score, recorded before eval. Eval is
  reported for all variants, labelled selected / ablation.

**Why GT-only must be in.** MEVA has human labels for every instance. A distilled student is only
worth having if it beats, or adds to, a student trained on those labels. Without this ablation the
distillation claim cannot be tested.

### D8 — Cascade and the cost curve (eval, read once, after D4 and D7 are recorded)
Per protocol, Q1 (whole eval corpus) and Q2 (camera × hour windows, median ~56 candidates):

| System | VLM calls / query |
|---|---|
| Frozen SigLIP 2 (done) | 0 |
| LoRA retriever (done) | 0 |
| LoRA → student re-rank | 0 |
| LoRA → VLM re-ranks top k, k ∈ {5, 10, 20, 50} | k |
| LoRA → student → VLM re-ranks top k, k ∈ {1, 3, 5, 10, 20} | k |
| VLM on every candidate (upper bound) | Q2: window size (~56); Q1: ~3.8k |

- **Re-ranking the top k:** the top k are re-ordered by z(VLM) + γ·z(previous score), with γ chosen
  on validation; positions below k are unchanged. All points are computed from the cached scores
  (D5), so every budget is simulated exactly from the same scores.
- **Reported:**
  - Q2 S@1, S@5, MRR (forensic use, primary);
  - Q1 mAP and P@10;
  - per-activity tables;
  - the curve: Q2 S@1 against VLM calls per query, on a log axis;
  - estimated latency per query (calls × measured per-call latency + student time).

**The old "teacher ≠ upper-bound VLM" rule no longer applies.** Plan F required that the VLM giving
training labels differ from the VLM used as the upper bound, because evaluation queries were then to
be VLM-generated, which would measure self-agreement. Evaluation here uses MEVA's **human** labels,
so the same VLM can be both teacher and expensive stage; it is scored against humans.

### D9 — Hypotheses (falsifiable, fixed before any result)
- **H1 (student):** LoRA → student beats LoRA alone on eval Q2 MRR and Q1 mAP at 0 VLM calls, with
  the largest gain on the motion-defined group (baseline lift ≤ 2, Result 02 rule).
- **H2 (VLM):** LoRA → VLM top-10 beats LoRA alone on Q2 S@1. Tested only if G1 passes.
- **H3 (distillation):** the GT+KD student ≥ the GT-only student on eval Q1 mAP.
- **H4 (cascade):** LoRA → student → VLM top-5 reaches ≥ 90 % of the "VLM on every candidate" Q2 S@1
  using ≤ 10 % of its calls.
- **H5 (direction):** the VLM is near chance at separating `vehicle_turns_left` from
  `vehicle_turns_right`, while the student, with box motion, is not.

Each is reported whether confirmed or rejected.

### D10 — Where it runs
Everything runs on the laptop (your choice, 2026-10-31). The office GPU is used only if G0 fails
after the D3 fallbacks; you arrange it.

---

## 4. Code

| File | Purpose |
|---|---|
| `src/sfs/teacher/frames.py` | Load an instance's tight or context crops from the crop store; pick 8 evenly spaced frames |
| `src/sfs/teacher/vlm.py` | Load the VLM at a pinned revision; prompt; yes/no margin; KV-cache reuse across questions |
| `src/sfs/teacher/run.py` | Resumable labelling job (parts) → `teacher_trainval.npz` / `vlm_eval.npz`, with identity |
| `src/sfs/teacher/gate.py` | G0 feasibility report and G1 validation evaluation |
| `src/sfs/rerank/features.py` | Per-instance sequences: LoRA tight + frozen context embeddings + box motion from geom |
| `src/sfs/rerank/model.py` | Student transformer |
| `src/sfs/rerank/train.py` | GT / GT+KD / KD training, validation selection, crash-safe resume |
| `src/sfs/eval/cascade.py` | Cascade simulation over cached scores; cost accounting; curve |
| `scripts/cascade_report.py` | Result 04 tables and figure |
| `tests/test_teacher.py`, `test_rerank.py`, `test_cascade.py` | See §5 |

Dependencies: none new for the 2B bf16 path (transformers 5.20, torch, torchvision).

---

## 5. Verification

| Level | Check |
|---|---|
| Unit | Margin = log p_yes − log p_no on a tiny random-weight Qwen3-VL; **cached scores equal uncached**; frame picking even and in time order; box-motion features hand-computed; student masks padded frames; cascade on a hand-made example (re-order inside top k only, cost counted exactly); **student loaders refuse eval instances and `vlm_eval.npz`**; teacher labels never use eval phrasings |
| Integration | G0 on 20 real validation clips: VRAM, latency, determinism; student smoke run: loss falls, no NaN; resume after a simulated crash |
| Regression | Full existing suite passes; existing metrics unchanged |
| Result | Same eval queries and metric code as Results 02/03; every number reproducible from cached scores by one script |

---

## 6. State after this change

- `docs/results/04-cascade.md`:
  - G0 / G1 outcomes;
  - teacher accuracy per activity;
  - the student variants;
  - **the recall-vs-cost curve**;
  - H1–H5;
  - raw outputs in `docs/results/raw/04_*`.
- Experiment log: D-022 (approval), D-023 (G1 / teacher choice), D-024 (results).
- Cached VLM scores for every instance, so later router work (Plan F D3) needs no new VLM calls.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Small VLM is poor on tiny, low-resolution actions (SOVABench; Tiny Actions literature) | G1 gate before the long labelling run; report per activity; student also learns from human labels |
| VLM "yes" bias / scores not comparable across queries | Margin score; per-query z-normalisation before any fusion |
| Validation small → noisy choices | Tiny search space (4 VLM configs, 3 student variants, small grids); all variants reported on eval |
| Student input shift: LoRA embeddings of its own training instances are cleaner than of eval | Early stopping and selection on validation (unseen by LoRA); stage-1 score not an input (D6) |
| Download time / Hugging Face rate limit | ~8.5 GB at ~1.7 MB/s ≈ 1.5 h, resumable; the library uses the stored login token |
| 5–7 h labelling interrupted | Resumable parts, as in every previous long job |
| Windows-specific kernel problems | 2B Qwen3-VL uses standard attention (no custom kernels); Qwen3.5's hybrid kernels are why it is parked |

---

## 8. Prior work to position against
From a web survey on 2026-10-31. **Each reference must be read and checked before it is cited in the
report.**

- **TeachCLIP** (CVPR 2024): distils heavy text-video retrievers into a CLIP-based student.
- **Qwen3-VL-Embedding / Reranker** (2026): a VLM cross-encoder re-ranker with yes/no scoring, whose
  embedder is partly distilled from it.
- **VQAScore** (2024): P(yes) from a VLM as an alignment score.
- **SOVABench** (WACV-W 2026): VLMs on MEVA/VIRAT vehicle actions; near chance on direction.
- **Cost-aware cascade ranking** (SIGIR 2017): the classic information-retrieval cascade.

**Gap claimed:** no paper found that distils a VLM teacher into a *temporal* student re-ranker for
surveillance retrieval and reports accuracy against VLM calls per query.

---

## 9. Decisions requiring your approval

1. **O1 — Download path for the weights.** Two models, ~8.5 GB in total, ~1.5 h at today's measured
   1.7 MB/s, resumable.
   - The standing rule says the home connection is never on the data path for large downloads.
     That rule was set at 0.05–0.5 MB/s.
   - **Recommendation:** allow this one download at home. The alternative is to fetch at the office
     and copy it over.
2. **O2 — Model shortlist (D1):** Qwen3-VL-2B-Instruct and Qwen3-VL-Reranker-2B; others parked.
3. **O3 — Gates (D3, D4):** the G0 fallback order, and the G1 pass rule (+0.02 validation mAP on at
   least one protocol), including stopping for your decision if G1 fails.
4. **O4 — Student (D6, D7):** box motion from annotations as an input; the 3 variants including the
   GT-only baseline.
5. **O5 — Cost curve (D8)** and hypotheses H1–H5 (D9) as written.

---

## 10. Approval (owner, 2026-10-31)

| Item | Owner's answer | What it means for the build |
|---|---|---|
| **O1** download | Download now on the home connection (two days off from the office) | Both models fetched now. Hugging Face rate-limits this connection (HTTP 429), so the weights come from **ModelScope, Qwen's official mirror** (same files): every file is checked against the published sha256, and the mirror revision is pinned in `data/models/<model>/_source.json` (`scripts/fetch_vlm_weights.py`) |
| **O2** models | Experiment with both; select the one that gives the best results | Both go through G1 on validation; the better one is the teacher (D5). **Both** are also scored on eval and shown on the curve, labelled selected / ablation, as with the LoRA variants. Training-side teacher labels: the selected model only |
| **O3** gates | Do not stop. Skip, mark "needs owner feedback", keep going; put the questions with images in a document | A failed gate no longer pauses the work. G1 fail → continue with the best configuration, every affected result is labelled **"needs owner feedback"**, and a review page (`data/review/vlm_feedback.html`, local) shows the question with example clips (frames, VLM verdict, ground truth). G0 fail → the D3 fallbacks in order; the office GPU only if all fail |
| **O4** student | Approved | D6/D7 as written |
| **O5** curve + hypotheses | Approved ("note both") | D8/D9 as written; both VLMs appear on the curve (O2) |

Development starts after this approval; results are reported against the hypotheses fixed in D9.

---

## 11. Implementation notes (2026-10-31)

Choices made while building, within the approved design. Each one that changes a spec detail is
also listed as a question on the feedback page (O3).

| Point | What was done | Why |
|---|---|---|
| Weights | From ModelScope (Qwen's official mirror); every file checked against the published sha256; the mirror commit is pinned in `data/models/<model>/_source.json`; `scripts/fetch_vlm_weights.py` resumes and refuses a changed upstream | Hugging Face answers this connection with HTTP 429 (rate limit), even for the logged-in library |
| Offline runs | Every job runs with `HF_HUB_OFFLINE=1` | The cached SigLIP 2 text encoder otherwise waits through the 429 retries on each load |
| Frames for the VLM | 8 evenly spaced frames, longest side resized to 448 px for **both** views (784 visual tokens per clip); real frame times passed as video metadata | Tight crops are often 224 px; a fixed budget makes the tight-vs-context comparison fair |
| H5 measurement (added 2026-10-31, before eval was read) | Inside each Q2 window holding both turn directions: per-query AUC of the named direction over the opposite one, averaged; the VLM, LoRA, the student alone and the fused score on exactly the pairs the VLM scored (`direction_test`) | D9 named the hypothesis but not the measurement |
| Shortcut check (added 2026-11-03) | The G0 cached-vs-full limit (0.1) was an implementation threshold in `gate.py`, not part of D3. The owner accepted the cached path (Q4, 2026-11-03); restating the check against the bf16 noise floor is part of `spec-vlm-followups.md`. Q1 "VLM on every candidate" is now computed for Instruct (owner Q5, 2026-11-03), replacing the "not computed" row above | Owner answers, D-025 |
| Report scope (added 2026-10-31, before eval was read) | GT+KD and KD students evaluated on eval as 0-call ablations (H3 is an eval test); latency = calls × G0 uncached per-call time + student time; figure as plain SVG (no new dependency); paired bootstrap CIs as Result 03; per-query file stored gzipped | Gaps between the report code and D8/D9 found before reading eval |
| Frames of mixed shape (added 2026-10-31) | Every frame of a clip is letterboxed, centred with black bars, onto one canvas (the largest resized height × width in the clip); clips whose frames share a shape are unchanged | A frame with no actor box nearby is stored as the full frame (`regions_for`), so a tight clip can mix square crops with a 16:9 frame; teacher labelling stopped at part 70 on such a clip (instance 7646). All earlier scores are unaffected (such clips could not have been scored); labelling resumed |
| Reranker format | Its template puts the query **before** the video, so each (clip, query) pair is a full pass; only the instruct model reuses the video's KV cache across questions | Changing the reranker's trained format would be testing a different model |
| What gets scored (volumes) | Only the pairs each step uses: G1 = the LoRA top-50 of every validation query (4,800 pairs, 1,003 clips); teacher = the student pair pools (2 captions each); eval = Q1 top-50 under LoRA and under the student, Q2 every candidate for the instruct model, Q2 top-k (≤ 20) for the reranker | Scoring every clip against every text is ~250k pairs per model on eval alone — about 13 h for the reranker |
| G1 measure | Fusion inside the LoRA top-50, weight w chosen on validation; "VLM only" = w 0 inside the same top-50 | This is the use the VLM has in the cascade; full-corpus VLM scoring of validation is ~33k pairs per protocol and configuration |
| Q1 "VLM on every candidate" | Not computed; Q2 only | ~3.8k calls per query; it would be the most expensive point by far and adds nothing to the curve's shape |
| Student → VLM weight | Chosen on validation from the VLM scores of the student's top-20 per validation query | Same rule as every other weight: validation only |
| Box-motion features | Scale-invariant: displacement and size change relative to the instance's median box height, aspect ratio, actor count, distance between the first two actors, crop size, time | The crop store does not record frame size; relative motion also transfers across cameras |
| Student pair pools | A: 12,213 pairs (6,213 positive), 24,426 teacher requests | D7: positives + 100 hard + 100 random negatives per class |
| Tests | 24 new (113 in total pass): cached == full scoring on a tiny random-weight Qwen3-VL (atol 1e-4); crop index; frame preparation; score store refuses eval files for training; box features hand-computed; padded frames ignored; pool rules; exact resume after a simulated crash; cascade re-ranks only the top k and counts cost exactly | §5 |

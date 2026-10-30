# Concepts Guide — Everything This Project Uses, Explained

**Purpose:** learn every idea in the specs, in the order you need it, tied to *this* project.
Each section: the idea in plain words → why it matters here → one or two resources.

Suggested order (≈5–6 hours total including videos): 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10.

---

## 1. Embeddings — turning things into points

**Idea.** An embedding is a list of numbers (e.g. 768 numbers) that represents the *meaning* of an
image or a sentence. A good embedding model places similar things close together. "A man opening a
car door" and a photo of a man opening a car door should land near each other.

**Measuring "close": cosine similarity.** The angle between two vectors. 1.0 = pointing the same
way (same meaning), 0 = unrelated. If vectors are L2-normalised (length 1), cosine similarity is
just the dot product — which is why the spec says "L2-normalised" everywhere.

**In this project.** Every sampled frame becomes a 768-number vector. A query becomes a 768-number
vector. Search = "find the frame vectors with the highest dot product with the query vector."

**Read:** [Weaviate — Vector Embeddings Explained](https://weaviate.io/blog/vector-embeddings-explained)

---

## 2. Vision Transformer (ViT) — how the image encoder sees

**Idea.** A ViT cuts the image into a grid of small squares called *patches* (16×16 pixels), turns
each patch into a vector, and lets every patch "look at" every other patch through *attention*.
`ViT-B/16` = Base size, 16-pixel patches. The input is fixed at **224×224**, so a 224 image becomes
a 14×14 grid = 196 patches.

**Why this matters here — the small-person problem.** Our frames are 1920×1072. To feed a ViT,
the whole frame is shrunk to 224×224. A person 106×242 px becomes ~12×51 px — under one patch wide.
The model literally cannot see their hands. That is why we crop around the people first (spec D4).

**Read:** [Roboflow — Vision Transformers Explained](https://blog.roboflow.com/vision-transformers/) ·
[Aman's AI Journal — ViT primer](https://aman.ai/primers/ai/vit/)
**Background video (search on YouTube):** 3Blue1Brown, *Neural Networks* series, chapters on
Transformers and Attention — best visual intuition for attention that exists.

---

## 3. CLIP — teaching images and text to share one space

**Idea.** Two encoders: one for images, one for text (a *dual encoder* / *bi-encoder*). Trained on
hundreds of millions of (image, caption) pairs. In each batch, the model is pushed to make every
image's vector close to *its own* caption and far from every *other* caption in the batch. This is
**contrastive learning**.

**Result: zero-shot.** You can search images with sentences the model never trained on, because
"meaning" lives in a shared space.

**In this project.** This is how text queries find video frames at all. The weakness: CLIP-style
models learned from web photos, not surveillance footage — that gap is what LoRA (section 8) fixes.

**Watch:** [CLIP Explained in 15 Minutes](https://www.youtube.com/watch?v=RArntDjc8Q4) ·
[CLIP – Explained! (with code)](https://www.youtube.com/watch?v=FCDKn-vpn_o)
**Read:** [Hugging Face CV course — CLIP and relatives](https://huggingface.co/learn/computer-vision-course/en/unit4/multimodal-models/clip-and-relatives/clip)

---

## 4. SigLIP and SigLIP 2 — a better training loss

**Idea.** CLIP's loss uses a *softmax over the whole batch*: "of all 32,000 captions in this batch,
which one is mine?" That only works well with enormous batches. **SigLIP** replaces it with a
*sigmoid per pair*: for each (image, caption) pair, independently, "do these match — yes or no?"
Each pair's loss no longer depends on the rest of the batch.

**Why we care.** On an 8 GB GPU we can only fit small batches when fine-tuning. Sigmoid loss
degrades far less at small batch sizes than softmax loss does.

**SigLIP 2** keeps the sigmoid loss and adds extra training objectives (captioning, localisation,
dense features) plus distillation for the small models — better retrieval and better at *where*
things are in the image, which helps with small subjects.

**Your experiment:** we embed with both and pick the winner on the validation split (section 9).

**Watch:** [#308 SigLIP and SigLIP2](https://www.youtube.com/watch?v=7dTAS2Y8M28)
**Read:** [SigLIP, Explained: Why a Simpler Loss Beat CLIP's](https://www.laura-martel.com/blog/siglip-sigmoid-loss-explained) ·
[Hugging Face SigLIP docs](https://huggingface.co/docs/transformers/model_doc/siglip)

---

## 5. Vector search and vector databases

**Idea.** Comparing a query against millions of vectors one by one is slow. A vector database
builds an index (*ANN — approximate nearest neighbour*) that finds the closest vectors without
checking every one. It can also **filter by metadata** ("only camera G329, only last Tuesday").

**In this project.** ChromaDB holds the vectors (rebuildable). SQLite holds the alert metadata
(the source of truth). Why two stores is explained in `spec-retrieval-system.md` §6.

**Read:** [Weaviate — Vector Search Explained](https://weaviate.io/blog/vector-search-explained)

---

## 6. Bi-encoder vs cross-encoder — THE core idea of this project

This is the one concept to understand deeply. Everything else serves it.

| | Bi-encoder (CLIP/SigLIP) | Cross-encoder (VLM, re-ranker) |
|---|---|---|
| How | Encode image and text **separately**, compare vectors | Look at image and text **together** |
| Accuracy | Good | Better — it can reason about the pair |
| Cost | Image vectors computed **once**, reused for every query | Must run again for **every** (query, clip) pair |
| At 10,000 clips | 1 text encoding + 10,000 dot products ≈ instant | 10,000 model runs per query ≈ very slow/expensive |

**The cascade.** Use the cheap bi-encoder to shortlist the top 100, then the expensive model only
on those 100. **This project goes one step further:** train a *small* cross-encoder to imitate
the big one (section 7), so even the shortlist re-ranking is cheap. The research question is how
much accuracy survives that substitution — the recall-vs-cost curve.

---

## 7. VLM teacher and knowledge distillation

**VLM (vision-language model).** A large model that reads an image and text together and can
judge "does this frame show a person opening a car door?" Accurate, slow, expensive.

**Knowledge distillation.** Train a small *student* model to reproduce a large *teacher* model's
outputs. The teacher's scores are richer than yes/no labels ("0.9 relevant", "0.4 somewhat") —
the student learns the teacher's *judgement*, not just the answer.

**In this project.** Offline, the VLM scores thousands of (query, clip) pairs **once**. A small
re-ranker is trained on those scores. At search time the small model runs; the VLM rarely does.

**Watch:** [What is Knowledge Distillation?](https://www.youtube.com/watch?v=JodeQwmp78I) (short) ·
[Knowledge Distillation — Full Tutorial with Code](https://www.youtube.com/watch?v=l44uC7jfnvY) (deeper)

---

## 8. LoRA — fine-tuning without retraining the whole model

**Idea.** A pretrained model has ~86 million image-side weights. Retraining all of them needs huge
memory and can destroy what it already knows. **LoRA** freezes all original weights and adds tiny
trainable matrices beside them (low-rank = small). You train ~1% of the parameters.

**In this project.** We LoRA-adapt the SigLIP image encoder on MEVA so it understands surveillance
viewpoints — fits on an 8 GB GPU, and keeps the general knowledge intact.

**Watch:** [LoRA explained by its inventor (Edward Hu)](https://www.youtube.com/watch?v=DhRoTONcyZE) ·
[LoRA — visually + PyTorch code from scratch](https://www.youtube.com/watch?v=PXWYUTMt-AU)
**Paper:** [LoRA: Low-Rank Adaptation (arXiv 2106.09685)](https://arxiv.org/pdf/2106.09685)

---

## 9. Train / validation / test splits — and why you never choose using the test set

**Idea.**
- **Train** — the model learns from it.
- **Validation** — you use it to make *choices*: SigLIP vs SigLIP 2, number of frames, crop size.
- **Test (eval)** — touched **once**, at the end, to report the final number.

**Why.** Every choice you make by looking at a set slightly fits that set. If you pick the encoder
because it scored best on the test set, the test score is no longer an honest estimate — you have
quietly trained on it.

**Leakage.** If the same clip (or the same camera at the same moment) appears in train and test,
the model can memorise the scene instead of learning the action. That is why we split **by clip**,
never by individual instance, and report camera overlap.

---

## 10. Evaluation metrics

- **Recall@K** — out of all queries, in what fraction did a correct clip appear in the top K
  results? Recall@1 is strict; Recall@10 is forgiving. An investigator scans ~10 results, so
  Recall@10 is the practical metric.
- **MRR (Mean Reciprocal Rank)** — if the first correct result is at position 1 → score 1, at
  position 2 → 0.5, at 4 → 0.25. Averaged over queries. Rewards putting the right answer *first*.
- **Macro vs micro average** — micro averages over all queries (dominated by common activities);
  macro averages per activity type first (every activity counts equally). We report both, so
  "great at `person_talks_to_person`, useless at rare actions" can't hide.
- **Pareto frontier** — plot accuracy (y) against cost (x) for each system configuration. The
  frontier is the set of configurations where you can't get more accuracy without paying more.
  Our headline result is this curve.

---

## 11. Frame sampling and length bias (project-specific)

An activity lasting 10 seconds has more frames than one lasting 1 second. If an alert's score is
the **max** similarity over its frames, longer alerts get more lottery tickets and rank higher
for no good reason. Fixes: cap frames per instance (16), sample at a fixed rate (2 fps), and score
with the **mean of the top 3** frames instead of the single best. Details: `spec-retrieval-system.md` §5.2.

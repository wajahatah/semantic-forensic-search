# Spec — Alert Indexing & Natural-Language Retrieval System

**Date:** 2026-09-19
**Status:** Draft for review — not yet approved for build
**Related:** [`project-selection.md`](project-selection.md) Plan F

---

## 1. What this spec covers

The system has **two distinct paths** that are easy to conflate. They share one component — the
encoder — and everything else differs.

| | Offline path (training) | Online path (serving) |
|---|---|---|
| Runs | Once, during development | Every time an alert fires / a query is made |
| Input | MEVA corpus + curated queries | Live alert frames; investigator query text |
| Output | Trained encoder + trained re-ranker | Ranked alerts with frames and metadata |
| Where | Colab GPU | Local (RTX 4060) |

`project-selection.md` specified the offline path in detail. **This spec adds the online path**,
which was previously undefined.

---

## 2. State before this change

- MEVA `examples/` acquired locally: 121 clips, 191 annotation files, 36 activity types, 1.1 GB.
- Full annotations blocked locally (gitlab.kitware.com unreachable); Colab route untested.
- Offline training design specified; **no serving design existed**.
- No alert record schema, no vector store choice, no query-time contract.

## 3. State after this change

- A defined alert record schema with frames, embeddings and metadata.
- A chosen vector store supporting hybrid (vector + metadata) query.
- A query-time flow that integrates the trained re-ranker from the offline path.
- An explicit encoder-consistency contract between the two paths.

---

## 4. Alert record schema

One record per alert. `nexa_fight` (or any detector) produces the alert; this system indexes it.

```python
{
  "alert_id":        "cam03_20260919_143022_0017",   # primary key
  "camera_id":       "G329",
  "site":            "school",
  "ts_start":        "2026-09-19T14:30:22Z",         # ISO 8601, UTC
  "ts_end":          "2026-09-19T14:30:27Z",
  "duration_s":      5.0,

  # provenance from the detector
  "detector":        "nexa_fight@v0.3",
  "detector_score":  0.87,
  "detector_label":  "possible_altercation",
  "n_persons":       3,

  # media
  "video_path":      "archive/2026-09-19/G329_1430.mp4",
  "frame_indices":   [1290, 1305, 1320, 1335, 1350],  # VARIABLE length, N >= 1
  "frame_paths":     ["frames/…_00.jpg", ...],
  "thumbnail_path":  "frames/…_thumb.jpg",            # for the UI
  "n_frames":        5,                               # sampled count, <= N_max
  "n_frames_avail":  150,                             # before subsampling; see 5.2

  # retrieval  (stored in the vector store, keyed by alert_id)
  "embeddings":      [[...], ...],                    # N x 512, L2-normalised
  "encoder_id":      "siglip-b16-meva-lora@a3f91c",   # MUST match index-time encoder
  "embed_version":   2
}
```

Frame count is **not fixed**. `n_frames` varies per alert (duration, detector confidence, motion);
§5.2 caps it at `N_max` and corrects the resulting length bias. The metadata fields above live in
SQLite; only vectors plus filter keys go to the vector store (§6.1).

**Metadata fields are not decoration.** They enable the filtered queries investigators actually
issue — *"altercations on camera 3 between Tuesday and Thursday"* — which pure vector search cannot
express. Vector similarity handles *what happened*; metadata handles *where and when*.

---

## 5. Design decision: variable-length frame sets

Frames per alert is **variable, not fixed** — it may scale with alert duration, detector confidence,
or motion content. The design must handle any N ≥ 1.

### 5.1 Per-frame storage, not pooling

| Option | Storage | Behaviour | Verdict |
|---|---|---|---|
| **A. Mean-pool** the N frame vectors into 1 | 1 vec/alert | Averages the decisive frame with N−1 uneventful ones, washing out the signal | **Rejected** |
| **B. Store all N**, search at frame level, aggregate | N vec/alert | The best-matching frames determine the alert's score | **Chosen** |

A query like *"person opening a car boot"* should match the one frame where the boot is open, not
the average of the alert. Per-frame storage also gives free temporal localisation — the winning
frame says *when* inside the alert.

**Cost is negligible.** 512 dims × 4 bytes = 2 KB per frame. Even at 50 frames/alert,
100,000 alerts ≈ 10 GB. Do not optimise this.

### 5.2 Variable N introduces a length bias — this must be corrected

With pure `max` aggregation, an alert with 50 frames has 50 chances to score highly; one with 3
frames has 3. **Longer alerts rank higher regardless of relevance.** This is a statistical artefact,
not a signal, and it would silently corrupt every retrieval metric.

Two corrections, applied together:

1. **Cap and sample.** Sample at most `N_max` frames (default 16) uniformly across the alert window.
   Short alerts keep all frames; long alerts are subsampled rather than accumulating draws.
2. **Score by mean-of-top-m, not pure max.** With `m = min(3, N)`:

```python
sims   = sorted((cos_sim(q, f) for f in alert.frames), reverse=True)
score  = sum(sims[:m]) / m          # m = min(3, len(sims))
```

Mean-of-top-m is far less sensitive to N than max, because a single lucky frame cannot carry the
alert. Retain the argmax frame separately for display and localisation.

**Validation requirement:** plot alert score against N on the evaluation corpus. If a correlation
remains, the bias is not corrected and `N_max`/`m` need adjustment. This check is cheap and must be
run before any retrieval number is reported.

---

## 6. Storage architecture — two stores, not one

**Revised 2026-09-19.** The original draft specified a single ChromaDB store. That was wrong, and
the correction rests on one asymmetry:

> **The vector index is *derived* data — always rebuildable from frames + encoder.
> The metadata is *source of truth* — losing it is unrecoverable.**

Mixing the two means a vector-store migration puts irreplaceable data at risk, and any encoder change
(which forces a full reindex, §8) would churn the metadata too.

### 6.1 The split

| Store | Holds | Role |
|---|---|---|
| **SQLite** (or Postgres later) | alert_id, camera, timestamps, detector provenance, frame paths, durations, audit trail | **Source of truth.** Durable, backed up, never rebuilt |
| **ChromaDB** | vectors, alert_id, and only the fields needed for filtering (camera_id, ts_start, detector_label) | **Rebuildable cache.** Disposable by design |

Rebuilding the index becomes a routine operation:

```
DELETE the collection → re-encode frames → re-insert vectors → done
```

No metadata is touched. This is what makes the encoder-consistency contract in §8 safe to enforce.

### 6.2 On "will Chroma cause conflicts later" — three real risks

The concern is justified. Named plainly:

| Risk | Reality | Mitigation |
|---|---|---|
| **API churn** | Chroma has shipped breaking changes between minor versions | Pin an exact version in `pyproject.toml`; never float |
| **Concurrency** | Embedded Chroma is SQLite-backed and effectively single-writer. **If the alert ingester and the query service run as separate processes, they will conflict** | Either run Chroma in server mode, or serialise writes through one process. This is the most likely failure in a live demo |
| **Scale ceiling** | Comfortable to ~1M vectors; degrades beyond | Not a concern at this project's scale; revisit only if exceeded |

### 6.3 The actual protection: an interface, not a database choice

No database choice is future-proof. What *is* future-proof is keeping the choice replaceable:

```python
class VectorStore(Protocol):
    def upsert(self, alert_id: str, vectors: np.ndarray, meta: dict) -> None: ...
    def search(self, qvec: np.ndarray, k: int, where: dict | None) -> list[Hit]: ...
    def delete(self, alert_id: str) -> None: ...
    def encoder_id(self) -> str: ...
```

Implement `ChromaVectorStore` now. If Chroma becomes a problem, `QdrantVectorStore` or
`FaissVectorStore` is a new class and a reindex — not a migration, and not a rewrite of the
retrieval logic. Nothing above this interface knows which store is underneath.

**This is the answer to "make sure it will not create conflicts later."** The guarantee comes from
the abstraction plus the rebuildable-index property, not from picking the right database today.

---

## 7. Query-time flow

```
investigator types: "someone taking a bag out of a car boot"
   │
   ├─ [1] text encoder  ──────────────────────────────>  query vector (512-d)
   │        (same model family as the image encoder)
   │
   ├─ [2] ChromaDB search, with optional metadata filter
   │        vector similarity + where={camera, time range, detector_label}
   │        ──> top-K candidate alerts  (K ≈ 50–100)
   │
   ├─ [3] DISTILLED RE-RANKER  (the model trained in the offline path)
   │        scores query jointly with each candidate's best frame
   │        ──> re-ordered top-K                          [cheap: all K]
   │
   ├─ [4] ESCALATION POLICY
   │        escalate only top-k (k ≈ 5) to the VLM, or skip entirely
   │        ──> final verification                        [expensive: k only]
   │
   └─ [5] RETURN: ranked alerts, each with
            best-matching frame + thumbnail
            camera, timestamp, duration
            detector score and label
            relevance score + which stage assigned it
```

**Stage 3 is where the project's contribution lives.** The flow as originally described
(query → search → return frames) is stages 1, 2 and 5 only — a working retrieval system, but a
standard one. Stages 3 and 4 are what produce the recall-versus-cost result.

**Stages 1–2 alone are a complete demo.** Build those first; 3 and 4 are additive.

---

## 8. Encoder consistency contract

The single most likely source of silent failure.

- Alert embeddings **must** be produced by the same encoder weights used to build the index.
- Every vector carries `encoder_id`; the query path asserts it matches before searching.
- Re-embedding after any encoder change is a **full reindex**, not an incremental update.
- Store the encoder checkpoint hash in the collection metadata and fail loudly on mismatch.

Failure mode if ignored: no error is raised, similarities are meaningless, and results look
plausible but are wrong. Guard this with an assertion, not a comment.

---

## 9. Evaluation corpus — a scale problem to note

50 alerts is far too few to measure retrieval. Recall@10 over 50 items is near-meaningless, and
brute-force search would be instant, so no efficiency claim could be demonstrated.

**Fix:** treat each annotated MEVA activity instance as a synthetic "alert." The example set alone
yields hundreds; the full annotation set yields tens of thousands. This gives a corpus large enough
that Recall@K and the cost curve are both measurable, using real footage and real human labels.

The 50-alert live pipeline then serves as the **demonstration**, while MEVA serves as the
**evaluation**. Both are needed; they answer different questions.

---

## 10. Where each piece runs

| Component | Colab | Local (RTX 4060) |
|---|---|---|
| Corpus download → Drive | ✓ | |
| Encode corpus → embeddings | ✓ (one GPU pass) | |
| Download embeddings (~10–50 MB) | | ✓ |
| Train LoRA adapter | ✓ or local | ✓ |
| VLM teacher labelling | ✓ (needs VRAM) | office GPU |
| Train distilled re-ranker | | ✓ (small model, vectors only) |
| ChromaDB index + query service | | ✓ |
| Alert ingestion pipeline | | ✓ |
| Demo UI | | ✓ |

---

## 11. Open questions

1. ~~Frames per alert — fixed or variable?~~ **Resolved 2026-09-19: variable, capped at `N_max`.** Defaults `N_max = 16`, `m = 3` are starting points to be tuned empirically against the length-bias check in §5.2, not fixed values.
2. **Frame selection** — uniform sampling across the alert window, or keyframe selection by motion/detector score? Uniform assumed as the baseline; keyframe selection is a refinement worth an ablation, since which frames you pick likely matters more than how many.
3. **Alert source for the live demo** — `nexa_fight` output, or a simpler person/motion detector? Affects only Stage 0.
4. **Retention** — do alerts expire? Not assumed; relevant if this ever ran continuously.
5. ~~GitLab annotation access from Colab~~ — **RESOLVED 2026-09-19.** Reachable from Colab (HTTP 200); `meva-data-repo` cloned: 15,776 objects, **13,968 annotation YAML files**, 7.6 GB. Evaluation uses the full annotation set. Local network still blocks it — all annotation work happens in Colab.
6. **Annotation/video alignment** — 13,968 annotation files cover ~184 h, but only ~5 h of video was downloaded (79 GB). Usable corpus = annotated instances **with** matching video. Quantified by `scripts/colab_build_manifest.py` (Cell 7); download more hours if the count is too low.

# Spec — Image + text LoRA on the office server (E8)

**Date:** 2026-11-04 · **Status:** draft, **waiting for the owner's approval** · **Log:** D-049
**Owner's request:** image + text LoRA (asked for on 2026-10-30 as "a later experiment, separate spec,
compared against this one", `spec-lora-finetune.md` §8 D2), trained on the office server. If it wins on
validation, only the student is retrained on it; no new VLM scoring (owner, 2026-11-04).
**Time budget:** 2 weeks to submission. This experiment is limited to **~2 working days** (build +
runs + write-up), so the demo and the report keep most of the time.

---

## 1. What this change does

Today's image model (Result 03, D-021) adapts only the **image side** of SigLIP 2 base with LoRA; the
text side is frozen. This change adds LoRA adapters to the **text side** as well, trains it on the same
data with the same rules, and compares it with today's set-up **on validation, on the same machine**.

## 2. State before

- **Model:** SigLIP 2 base (`google/siglip2-base-patch16-224`), LoRA rank 16 on the image side's
  attention projections, 1.18 M trainable parameters. The text side is frozen, so query encoding is the
  stock SigLIP 2 text encoder. Selected runs: A-middle (val mAP 0.437), B-uniform (0.376), both trained on the
  laptop (RTX 4060, ~1 min per epoch, best epochs 10–28).
- **Downstream:** LoRA embeddings → student features → GT students (Result 04) → optional VLM check.
  Several places embed text with the **frozen** text side: the trainer's validator, the eval
  (`sfs.eval.baseline`), the student (`sfs.rerank.train`, which can already read text embeddings from a
  file, `--text-emb`), the cascade report (`sfs.teacher.gate._phrasing_embeddings`).
- **Model version:** the runs record "unpinned". The laptop's files are at commit
  `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` (`model.safetensors` sha256 `61292338…`).
- **Office server:** crop store copied (`data/crops_v2`, 14 GB), 84 GB disk free, 30 GB RAM, the GPU held
  by our queue process (14.5 GB) until a STOP file. No SigLIP 2 weights; no packed training file.

## 3. State after

- `sfs.train.run` can train image-only (unchanged default) **or** image + text (`--text-lora`). Today's
  runs and their config hashes stay valid. A test pins this.
- Four runs on the server: image-only and image + text, for A-middle and B-uniform, with the same seed,
  data and settings. Comparing image + text against an image-only run **from the same GPU** removes the
  machine difference we saw with the students (D-046: 0.594 vs 0.6045 for the same run).
- If image + text passes the adoption rule (§5): re-embedded store, text embeddings exported with the
  adapted text side, GT students retrained, 0-call eval rows read once. Otherwise: recorded as a
  negative result, and the current system stays the final one.
- SigLIP 2 pinned to `75de2d55…` everywhere (closes the open follow-up).

## 4. Design

### 4.1 What is trained
- **Text side:** LoRA rank 16 on the text encoder's attention projections (q, k, v, out; 12 layers),
  same alpha/dropout as the image side. Embeddings, MLPs and the pooling head stay frozen.
- **Image side:** unchanged (rank 16 on its attention projections).
- **Logit scale / bias:** trained as today.

### 4.2 Training step
Today each step draws one caption per activity and uses its **fixed** embedding. With `--text-lora`,
the drawn captions (one per class, ~32 texts) go through the adapted text side **in the same step, with
gradients**, then the same label-aware sigmoid loss. Everything else is identical: sampler, augmentation,
batch 64, LR schedule, early stop (patience 5), ≤100 epochs, crash-safe resume.

### 4.3 Validation (D7, unchanged rule)
Q1 on the validation split with the **frozen eval phrasings** as queries. The phrasings are disjoint from
the training captions (checked by `disjointness_violations`), so validation measures how well the adapted
text side handles **wordings it never saw**. That is the main risk of this experiment (§7). With `--text-lora`,
the validator re-embeds the 66 phrasings with the current text side every epoch.

### 4.4 Running on the server without losing the GPU
The queue process holds the GPU and runs only VLM job types. A new job type **`module`** runs
one of our Python entry points (`sfs.train.run`, `sfs.embed.reembed`, ...) **inside** that process, so it
uses memory the process already holds. Ollama cannot take it in between. Starting the new code needs one
restart of the holder: STOP, then `~/sfs/start_holder.sh` straight away. The gap is a few seconds. Ollama only loads
on a request, so the risk is small; the script checks free memory before it reserves.

### 4.5 Data and weights on the server
- **SigLIP 2:** downloaded **on the server** (office connection, ~1.5 GB) at revision `75de2d55…`. Every
  file's sha256 is compared with the laptop's copy before use. Nothing goes over the home connection.
- **Packed training file** (8 GB): built on the server from its crop store with the same function
  (`build_packed`). The packed file's id must equal the laptop's (`store_id` in its meta). That proves
  identical input.
- **Baseline embeddings** for the re-embed check (`embeddings_v2_siglip2.npz`): copied from the laptop
  **while at the office** (LAN), or the check is skipped. The re-embed itself does not need them.

### 4.6 What is compared, and how much compute it takes
| Run | Where | Est. time |
|---|---|---|
| Image-only A-middle, B-uniform (server reference) | server | ~20–30 min each |
| Image + text A-middle, B-uniform | server | ~25–35 min each |
| *Only if adopted:* re-embed 2 stores, export text embeddings, features, 2 GT students, eval | server (+ laptop for the report) | ~1.5 h |

## 5. Adoption rule (fixed before any run)

Image + text is adopted **only if all hold**, on validation:
1. **Gain:** mean of (image + text − image-only, server) over A and B **≥ +0.010** val mAP.
2. **No protocol loses:** neither A nor B drops by more than 0.005.
3. **Retention check passes** (§6).

Otherwise the current image-only model stays the final model (Result 03/04/05 unchanged), and E8 is
reported as a result either way. Eval is read **once**, only after the decision, and only for the
adopted model: Q1/Q2 for LoRA alone and LoRA + student (0 VLM calls).

## 6. Retention check: does the text side forget general language?

The concern the owner raised (2026-11-04): nexa_fight alerts are searched with words MEVA never trained,
for example "two people fighting". The text side is trained on only 192 captions, so it can drift towards them.

- **(R1) Recommended if available: real check.** A small set of labelled nexa_fight clips (fight / no
  fight, a few dozen each). Measure retrieval of "people fighting"-style queries with the frozen text side,
  image-only LoRA, and image + text LoRA. Pass = image + text no worse than image-only by more than 0.02 AP.
  **Needs the owner:** does nexa_fight have such labelled clips that we may use?
- **(R2) Fallback: text-only drift check.** 40 general surveillance sentences outside MEVA's
  activities (fights, falls, running, crowds, ...), written and frozen **before** training
  (`queries/retention_sentences.csv`). Measure how well the adapted text side preserves their pairwise
  similarities (Spearman ρ against the frozen side). Pass = ρ ≥ 0.90. This is a guard against collapse,
  not an accuracy measure. The report will say so.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Text side over-fits to the 192 captions and gets worse on new wordings | Validation uses only unseen phrasings; early stop; adoption rule §5; retention check §6 |
| Gains within run-to-run noise (single seed) | Both arms on the same GPU and seed; +0.010 bar; reported with that caveat |
| Holder restart lets Ollama take the GPU | Restart in seconds, at a quiet time; free memory checked first |
| Time overrun eats demo time | Hard limit ~2 days. If not done by then, E8 is stopped and reported as not completed |
| Every text-embedding user must use the adapted text side, or results are silently wrong | All four users read one exported text-embedding file (`--text-emb`), which records the adapter hash. A mismatch is an error. Tests cover it |

## 8. Code changes

| File | Change |
|---|---|
| `src/sfs/train/lora.py` | Load the full model (image + text side); `wrap_text_lora`; `text_features`; adapter state covers both sides |
| `src/sfs/train/run.py` | `TrainConfig.text_lora` (default off, so old hashes are unchanged); per-step caption encoding; validator re-embeds phrasings |
| `src/sfs/embed/reembed.py` | Loads image + text adapters (image side used for re-embedding) |
| `scripts/export_text_embeddings.py` | `--adapter`: export with the adapted text side; records the adapter hash |
| `src/sfs/eval/baseline.py`, `src/sfs/eval/cascade_run.py` / `sfs.teacher.gate` | `--text-emb` (same file format as the student's) |
| `src/sfs/teacher/queue.py` | Job type `module` (runs a whitelisted entry point in-process) |
| `scripts/retention_check.py` + `queries/retention_sentences.csv` | §6 (R2; R1 if data exists) |
| Tests | Image-only config hash unchanged; text adapters change only text features; validator uses the adapted text; export/reading round-trip; adapter-hash mismatch raises; `module` job runs and is limited to the whitelist |
| Docs | README runbook, Result 06 (new), EXPERIMENT_LOG, CLAUDE.md, progress page |

## 9. Verification
- Full test suite on the laptop (152 + new).
- Smoke run on the server (`--max-steps-per-epoch 5`) before the real runs.
- Image-only server run vs laptop run: reported side by side (machine difference, not a gate).

## 10. Decisions for the owner
1. **Approve this spec** (or change it).
2. **Retention check:** R1 with nexa_fight labelled clips (please say if they exist), or R2 only.
3. **Adoption bar** +0.010 mean val mAP, neither protocol −0.005 — OK, or another value?

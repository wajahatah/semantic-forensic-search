# Spec — VLM follow-up experiments (after Result 04)

**Status:** E1, E2, E3, E5 **complete** (2026-11-03, D-028, `docs/results/05-vlm-followups.md`); E3 stopped by its rule (G1 failed at 16 frames); E4 **approved and in set-up** (D-029: Qwen3-VL-8B-Instruct-FP8 on the 16 GB office GPU, wait for free GPU, G1 top 50 then whole validation) · **Basis:** the owner's answers to the 7 questions of
`data/review/vlm_feedback.html` (2026-11-03), recorded in `docs/EXPERIMENT_LOG.md` D-025.
**Follows:** `docs/planning/spec-rerank-cascade.md` (Result 04, complete).

---

## 1. In short

Result 04 found that a small (2B) VLM adds very little to the search, and that neither the VLM nor
the student can tell a left turn from a right turn. Your answers ask for:

| # | Experiment | Your words (short) | Runs on | Cost / time |
|---|---|---|---|---|
| E1 | Count "turns left" and "turns right" as one activity, **vehicle turns** | Q1: "go with simple vehicle is turning" | Laptop | ~1 h |
| E2 | Re-run gate G1 on the **whole validation set**, not only the top 50 | Q6 | Laptop | ~5 h (mostly the Reranker) |
| E3 | Re-test the 2B VLM with **16 frames** instead of 8 | Q2, option C | Laptop | ~2 h for G1; more only if it passes |
| E4 | Test a **bigger open VLM served from the office server** as our own API; if it passes G1, retry distillation with it | Q2 + Q3 | Office server (model) + laptop (client) | No money; G1 ~1 h, more only if it passes |
| E5 | "VLM on every candidate" for Q1 (Instruct) | Q5 | Laptop | ~1.5 h, **already running** (approved in Q5) |
| — | Shortcut check reworded; full slow re-scoring kept as an option for later | Q4 | — | Noted only |
| — | Weights checked against Hugging Face | Q7 | — | **Done:** every model file matches (§2) |

Every experiment ends in a recorded result, whether it helps or not, so the report can say "we tried
X, measured Y, and therefore chose Z". Nothing here changes Results 02–04: new numbers are added
next to them, never in their place.

---

## 2. State before this change

- **Result 04 (complete):** LoRA → student (0 VLM calls) gives Q2 Success@1 0.53 (A) / 0.52 (B).
  A 2B VLM on the top 5–10 adds at most +0.03. Gate G1 failed for all four 2B set-ups (best +0.012,
  needs +0.02). Distillation from the 2B Reranker made the student worse (GT kept). Left vs right
  turns: VLM and student both at chance (AUC ≈ 0.50).
- **G1 design:** the VLM only re-ranks the LoRA top 50 of each validation Q1 query (4,800 pairs).
- **Frames:** 8 per clip, longest side 448 px. The crop store holds up to 16 frames per clip
  (2 fps), so 16 frames = every stored frame.
- **Shortcut check (G0):** `src/sfs/teacher/gate.py` fails G0 when cached and slow scores differ by
  more than 0.1. This number came from the implementation, not the spec. 16-bit rounding alone
  moves scores by up to 0.90, so the check fails although nothing is wrong. You accepted the
  shortcut on 2026-11-03 (Q4).
- **Weights (Q7), checked 2026-11-03:** Hugging Face answered this time. Every model file of both
  VLMs matches Hugging Face's own checksums (the weights by sha256, other files by git blob hash).
  Only `.gitattributes` differs, the mirror's repository settings, which the model never reads. The
  revision recorded at download time was ModelScope's own; the Hugging Face revisions are now stored
  next to it (`data/models/*/_source.json`; Instruct `89644892`, Reranker `4bd860ac`). Record:
  `runs/cascade/weights_hf_check.json`. **No token and no re-download are needed.**
- **E5:** running now (213,090 missing pairs; the 244,620 pairs scored for Result 04 keep their
  values).

---

## 3. The experiments

### E1 — "vehicle turns" as one activity (Q1)

**What:** treat `vehicle_turns_left` and `vehicle_turns_right` as one activity, `vehicle_turns`, in
an **extra** evaluation. The 22-activity results stay the main record, because the query set was
frozen before any result was read (D-017). The merged view is reported next to them as "21
activities, turn direction merged".

**How:**
- One mapping is applied to the manifest before queries are built (left/right → `vehicle_turns`).
  Q1 and Q2 queries for the new activity are then built by the existing rules
  (`sfs.eval.queries.build_q1/build_q2`).
- Three new phrasings with no direction. **Proposed, for your approval:**
  1. "a vehicle turning at an intersection"
  2. "a car making a turn"
  3. "a car turning onto another road"
- No retraining. LoRA and the student turn text into vectors, so a neutral query works directly.
  Only the new phrasings need embedding, and the VLM scores the new pairs (top 50 / top 10 only,
  ~15 min).
- Nothing is selected again: the fusion weights (γ, β) and the VLM set-ups are reused from Result 04
  (`cascade_run --fixed-choices`); eval is read once. The VLM also scores every eval clip for the 3 new
  phrasings, so the E5 point exists in the merged view too.

**Why this is not choosing on eval numbers:** the reason exists on validation too. G1's diagnostic
shows the VLM hurting turn-left most on validation (−0.46 AP, A). The change adds a view; it does not
replace or re-select anything.

**What you get:** Result 04's tables again with 21 activities, plus one line for "vehicle turns".
Expected: the turn activity jumps from chance to a useful AP, because "is it turning?" is visible in
box motion even when the direction is not.

**Alternative (not recommended):** also merge the training captions and retrain LoRA and the
student (~1 day). It would make the training match the merged view, but it changes Results 03–04
models. Only worth doing if you want "vehicle turns" in the final demo instead of left/right.

### E2 — G1 on the whole validation set (Q6)

**What:** the same G1 test, but the VLM scores **every** validation clip for every query instead of
the LoRA top 50. Fusion is the same: z(VLM) + w·z(LoRA), w chosen from {0, 0.25, 0.5, 1, 2}.

**Size:** 1,325 validation clips (A and B together) × up to 66 phrasings ≈ 87k pairs per set-up;
the 4,800 already scored are reused.

| Set-up | Time on the laptop |
|---|---|
| Instruct, tight | ~20 min (shortcut) |
| Instruct, context | ~20 min |
| Reranker, tight | ~4 h (no shortcut) |
| Reranker, context | skipped: worst set-up in G1 (−0.008 / +0.001); ~4 h more if you want it |

**Decision rule, fixed now:** "the same" means, for every set-up run, (1) the same pass/fail
result and (2) a validation gain within ±0.01 of the top-50 gain. If the same → all later G1 runs
(E3, E4) use the top 50, as you said. If not → they use the whole validation set; for E4 that makes
G1 about 18× longer on the office server (87k instead of 4,800 pairs), so I would ask you first.

### E3 — 2B VLM with 16 frames (Q2, option C)

**What:**
1. G0 at 16 frames (GPU memory, time, repeatability), with the reworded shortcut check (§4).
2. G1 (top 50, or whole validation if E2 says so) for **Instruct-tight and Reranker-tight**. The
   context view is not repeated: it was worse for both models.
3. **Only if G1 passes:** teacher labels with 16 frames → GT+KD and KD students retrained →
   validation selection → eval top-k scoring → report rows. **If G1 fails:** recorded as "more
   frames do not fix the 2B VLM", and E3 stops.

**Time:** G1 ~1 h per model. If it passes: ~10 h of teacher labelling (Reranker; ~2 h if Instruct),
students ~1 h, eval scoring ~3 h.

### E4 — a bigger VLM served from the office server (Q2 + Q3)

**What (owner, 2026-11-03):** download a bigger open Qwen3-VL model onto the office server and run
it there as **our own API**. The laptop sends frames and questions to it and gets scores back. No
paid service.

**Which model (depends on the office GPU's memory, §8):** the **same family** as the 2B models, so
only the size changes.

| Office GPU memory | Model | Download at the office |
|---|---|---|
| 24 GB | Qwen3-VL-8B-Instruct, bf16 (4× the 2B) | ~17 GB |
| 48 GB | 8B, or Qwen3-VL-32B-Instruct in FP8 (8-bit) | ~17 / ~33 GB |
| 80 GB | Qwen3-VL-32B-Instruct, bf16 (16× the 2B) | ~66 GB |

Recommended: the 8B in bf16 first. It is the cleanest comparison: same precision, no quantisation,
and the same chat format as the 2B Instruct, so prompt, score and frames are unchanged. If memory
allows, the 32B next gives a size trend (2B → 8B → 32B). A bigger Reranker (8B) is added only if
Qwen publishes one; this is checked before download.

**How it is served:** the server runs **our own scorer code** (`sfs.teacher.vlm.VLMScorer`, the code
that produced the 2B scores) behind a small HTTP service.
- Same prompt, same frame preparation, same score (log p(yes) − log p(no)), same video-once
  shortcut. Any difference in results is then the model's, not the serving stack's.
- A general serving engine (for example vLLM) would be faster, but it prepares frames and prompts its
  own way, so 2B and big-model scores would not be directly comparable. It also needs Linux.
- Requests: one clip's frames (the stored crop bytes, lossless) + its list of questions → one score
  per question. `GET /info` returns the model name and revision; these go into every score file, as
  for the 2B runs.
- The laptop side is a `RemoteScorer` with the same interface, so `sfs.teacher.run` works unchanged
  (`--remote http://<office-server>:<port>`). Resume, part files and identity checks also stay as
  they are.

**Access and safety ("open it"):**
- The service listens on the **office network only**, not the public internet.
- Every request carries a shared secret. You create it and set it as an environment variable on both
  machines; I never see or handle it.
- Reaching it from home needs an SSH tunnel, not an open port. Opening a firewall port on the office
  server is your (or office IT's) step.
- Requests are size-limited, and the service only scores; it does not read or write files on the
  server.

**Steps, each with a stop point:**
1. **Set-up check:** the server answers `/info`; a laptop-side G0 measures memory, speed and
   repeatability. The 2B Instruct is also served once on a few clips, and its scores must match the
   laptop's to bf16 noise. That proves the API adds no error.
2. **G1 (top 50, or whole validation if E2 says so; tight; 8 frames, or 16 if E3 is better on
   validation).**
3. **Only if G1 passes (approved 2026-11-03):** teacher labels on the student training pools (41,266
   pairs), GT+KD and KD students retrained, eval top-5 / top-10 scoring for the cascade. **If G1
   fails:** recorded as "a bigger VLM does not help either", and E4 stops.

**Time:** depends on the office GPU (§8). The 8B is roughly 3–4× slower per call than the 2B. G1
(4,800 pairs, ~1,000 clips) is about 1 h; teacher labels are several hours.

**Data transfer:** ~0.5 MB of crop bytes per clip, sent once per clip with all its questions. G1 is
~0.5 GB, teacher labels ~10 GB. So these runs happen while the laptop is **at the office** (same
network). From home, only G1 would be practical.

**Licence:** Qwen3-VL open weights are Apache 2.0; MEVA is CC BY 4.0. Both allow this use.

#### E4 revised after the server survey (2026-11-03, D-029; replaces the model table and access notes above)

**Office server `lt-ai-server` (surveyed read-only over SSH):** RTX 4070 Ti SUPER **16 GB** (driver
615, FP8-capable), 12 cores, 30 GB RAM, Ubuntu 24.04, **62 GB disk free**, Hugging Face at ~15 MB/s,
LAN 192.168.100.63. Shared team machine: the team's Ollama service holds ~6.5 GB of the GPU while its
models are loaded (idle models unload after ~30 min). No sudo, no Docker access; neither is needed.

**Model (owner, 2026-11-03): Qwen3-VL-8B-Instruct-FP8** (Qwen's official checkpoint, revision
`9cdc6310a8cb770ce18efaf4e9935334512aee45`, Apache 2.0, 10.6 GB). Only the language model is FP8
(block-wise 128×128, dynamic activations); the vision tower and `lm_head` stay bf16, so the video side
is computed at the same precision as the 2B runs. Estimated peak ~13 GB. The 8B in bf16 (17.5 GB)
does not fit; the 4B in bf16 is the fallback if FP8 does not load. Qwen3-VL-Reranker-8B exists only in
bf16 (does not fit), so E4 tests the Instruct model only.

**GPU sharing (owner):** the service checks free GPU memory before loading the model and **waits**
until enough is free; the owner tells the team. A request that hits out-of-memory (e.g. Ollama
reloads) returns an error and the laptop retries with back-off; nothing is written for that clip.

**G1 depth (owner, after E2 "not the same"):** top 50 first; **only if it passes**, the whole
validation set; both must pass.

**Environment:** private conda env in the owner's home on the server (`~/envs/sfs`, Python 3.12,
conda-forge channel only; the shared `/opt/miniconda3` envs are not touched), torch / transformers /
torchvision at the laptop's versions (2.14.1, 5.20.2, 0.29.1). The `sfs` package is copied from the
laptop (it is not on GitHub yet). Weights downloaded on the server with the library, pinned to the
revision above.

**Access (narrows "office network only"):** the service binds to **127.0.0.1 on the server**; the
laptop reaches it through an SSH tunnel (`ssh -L`), at the office and from home alike. No port is
opened on the shared machine and no firewall change is needed. The shared secret stays (other users
logged into the server can reach its localhost); the owner sets it as an environment variable on both
machines.

**FP8 needs `kernels`:** the server env adds Hugging Face's `kernels==0.17.2` (the FP8 matrix kernel is fetched from `kernels-community` once and cached).

**Versions must stay pinned:** transformers 5.20.2 on both machines. transformers 5.22 changes Qwen3-VL
video preprocessing (a per-frame pixel cap becomes the default; warning seen 2026-11-03), which would
change the token budget per frame and make 2B and 8B scores incomparable.

**Built and checked 2026-11-03 (D-030):** server env `~/envs/sfs` (torch 2.14.1+cu126, transformers
5.20.2, OpenCV 5.0.0; isolated from `~/.local` with PYTHONNOUSERSITE=1); both models downloaded and
every file verified against Hugging Face (8B-FP8 13 files, 2B 12 files); `sfs` code copied to
`~/sfs/code`. On the laptop the changed `VLMScorer` re-scores 19 stored G1 pairs exactly (max diff 0).

**FP8 note for the report:** the comparison is 2B (bf16) vs 8B (FP8 language model). Qwen reports
FP8 quality close to bf16, but this is not measured here; it is stated as a limitation.

---

### E6 — Gemma 4 instead of Qwen (proposed 2026-11-03; owner: "first experiment locally", see revision below)

**Why (owner, 2026-11-03):** a colleague's experiments with Google's Gemma 4 gave good output, and the
Qwen models have not given the results we want. E6 runs the **same G1 test** with Gemma 4, so the
report can compare model families, not only sizes.

**Models already on the server** (the owner's Hugging Face cache; no download). Both accept video in
transformers 5.20.2 (`Gemma4Processor` / `Gemma4UnifiedProcessor`, `videos=` input, `<|video|>`
token); Apache 2.0, not gated.

| Model | bf16 weights | On the 16 GB GPU | Plan |
|---|---|---|---|
| gemma-4-12B-it (cached rev `707f0a3b`; Hugging Face is now at `43c9d014`) | 23 GB | only with FP8 language model, ~12–13 GB + activations; needs the whole GPU | **recommended first** (closest in size to the Qwen 8B) |
| gemma-4-E4B-it (rev `ee0ef602`, current) | 15 GB (includes audio tower and per-layer embeddings) | bf16 leaves no room for activations; FP8 language model ~9 GB | fallback, or a second size point |
| gemma-4-26B-A4B / -it | 52 GB each | does not fit | not tested |

- **FP8 differs from E4:** Qwen's FP8 is an official checkpoint; for Gemma the language model is
  quantised by us at load time (transformers `FineGrainedFP8Config`, 128×128 blocks; vision and
  audio towers kept bf16). Stated as a limitation.
- **12B revision:** the cached files are first checked against Hugging Face at `707f0a3b` (no
  download). If the newer revision changed the weights, the owner chooses: keep the cached one, or
  download the new one at the office (23 GB, ~25 min).

**How it scores** (new `src/sfs/teacher/gemma.py`, `GemmaScorer`, same interface as `VLMScorer`):
- Same frames: `prepare_frames` output (8 frames, longest side 448, tight view), passed as a video
  with our frame metadata (no resampling). Gemma's own processor then sets its per-frame token budget
  (280 tokens per frame); this is the model's preprocessing, as Qwen's is for Qwen.
- Same question: "Does this video show the following: {query}? Answer yes or no.", video first,
  through Gemma's chat template. Same score: log p(yes) − log p(no) of the next token (yes/Yes vs
  no/No).
- Full forward pass per pair at first (G1 is 4,800 pairs). A video-once shortcut is added only if
  G1 passes (teacher labels need it), with the same cached-vs-full noise-floor test as Qwen.
- Served by the same service (`serve.py --family gemma`), so it runs from the office or, at night,
  directly on the server (below).

**Steps, each with a stop point (same rules as E4):** G0 on the server (memory, speed, determinism,
`--max-vram-gb 15`) → G1 top 50, tight → only if it passes: G1 on the whole validation set → only if
both pass: teacher labels, students, eval. If G1 fails: recorded as "Gemma 4 does not help either"
and E6 stops.

**Tests:** a tiny random-weight Gemma 4 built from the cached config (no download): the video comes
before the question in the prompt, the margin uses the right tokens, `score_full` is deterministic,
the interface matches `VLMScorer`; the service serves it end to end.

**Order:** one model in GPU memory at a time. Qwen 8B (E4) first, already set up; Gemma 12B (E6)
next. Each G1 is about 1 h if the GPU is free.

#### E6 revised: laptop first (owner, 2026-11-03, D-032)

The owner chose to **run Gemma on the laptop first** and use the office server only if needed. On the
8 GB laptop GPU only **gemma-4-E4B-it** can run (the 12B cannot). It is copied from the server's cache
over the office LAN (15 GB) and checked file by file against Hugging Face at `ee0ef602`.

**Memory (measured from the weight file):** language-model layers 7.9 GB, per-layer embedding table
5.8 GB, token embeddings 1.3 GB, audio tower 0.6 GB, vision tower 0.3 GB; 16 GB in bf16.

**Placement, all in bf16 (no quantisation, same precision as the Qwen 2B runs):**
- The per-layer embedding table stays in CPU memory and is looked up on the CPU; only the looked-up
  rows go to the GPU. It is a table lookup, so this is exact.
- The audio tower stays on the CPU and is never called (video only).
- The vision tower, token embeddings and as many decoder layers as fit in the GPU budget
  (`--gpu-budget-gb`, default 6.0 of the 7.5 GB limit) stay on the GPU. The remaining decoder layers
  are kept in CPU memory and copied to the GPU for each pass (our `stream_weights` hook; accelerate's
  `cpu_offload` copies the layer's arguments and so breaks Gemma's in-place KV sharing). Results are
  exact; only speed changes.
- **Video input:** transformers 5.20.2's `Gemma4Model.forward` does not pass `num_frames_per_video`
  to the vision encoder, so pixels-in always fails. The scorer runs the vision encoder itself, once
  per clip, and passes its output through the supported `mm_encoder_outputs` input.
- **Approved 2026-11-03 (owner): "start the experimentation with gemma4 that we have done with the
  qwen ... don't wait for the spec approval".** Weights copied from the server cache and verified
  against Hugging Face (7 files). Tests: `tests/test_gemma.py` (tiny Gemma 4 with KV sharing; split
  placement equals one device); full suite 139 pass.
- **Result 2026-11-03 (D-033):** run on the office server beside Ollama (laptop RAM too low). G1 fails
  (mean −0.002); G0 fails only on speed. E4B stops; the 12B is the owner's call.
- FP8 is not used on the laptop: transformers' FP8 kernels need Triton, which is not available on
  Windows. Quantisation would also confound the Gemma-vs-Qwen comparison.

**Scoring:** as in E6 above. Full forward pass per pair (`score_clip` is a loop of full passes,
reported as "no shortcut"). The G0 report adds the mean probability mass on yes/no, to show the model
answers in the expected form. Gemma's processor uses about 66 tokens per frame (528 for 8 frames).

**Run:** G0 on the laptop (`--max-vram-gb 7.5`), then G1 top 50 (tight, 8 frames). Same pass rule,
same next steps. If G1 is close or passes, the 12B on the office server is the next question for the
owner.

### E7 — Qwen2.5-VL from the owner's server cache (owner, 2026-11-03, D-038)

**Why:** "don't delete qwen model, instead use it in our experimentation and check if it is
contributing something or not if not then we will delete it" (Qwen2.5-VL-3B/7B-Instruct, already in
the owner's server cache).

**Models** (linked into `~/sfs/models`, no copy; every file checked against Hugging Face):
Qwen2.5-VL-3B-Instruct rev `66285546` (bf16, 7.1 GB; licence: Qwen research licence per its model
card, not in the metadata — academic use), Qwen2.5-VL-7B-Instruct rev `cc594898` (Apache 2.0; bf16
16 GB does not fit, so the language model is stored in FP8 at load time, 128×128 blocks, vision tower
and lm_head bf16; run name `+fp8load`).

**How:** `VLMScorer` now loads any Qwen-VL generation through transformers' auto class (Qwen3-VL
resolves to the same class as before; laptop 2B re-scores 19 stored pairs exactly). Same frames,
prompt, yes/no margin and video-once shortcut (tiny Qwen2.5-VL test: cached = full).

**Run (server, after the 8B cascade scoring):** G0 (context, limit 15 GB) and G1 scoring (top 50,
tight, 8 frames, same 4,800 pairs) for the 3B, then for the 7B. **Rule:** same as every VLM — pass
= gain ≥ 0.02 on a protocol; if neither passes, recorded as "not contributing" and the owner deletes
them.

### Night-time runs from home (owner, 2026-11-03)

**Data stays off the home connection:** at the office, the crop store (14 GB), manifest and request
files are copied to the server over the LAN. At night the jobs run **on the server** (the same
`sfs.teacher.run` code with `--model`, started with `nohup`), so they do not depend on the laptop,
the tunnel or the home link. From home only SSH and small result files (MB) are needed; scores are
copied back to the laptop. The service (`serve.py`) stays for office use. Disk: the server has 38 GB
free; the copy leaves ~24 GB (see the clean-up list given to the owner).

**Access from the internet (revised D-032: no IT person; the owner and a colleague set it up
themselves).** Claude does not set up remote-access software or change security settings. The
recommended route is a private mesh VPN (Tailscale) running as the owner's user on the server:
outbound-only, no open port, only the owner's devices, SSH only, no route to the office LAN. Step-by-step
guide and revocation checklist: `docs/planning/office-server-remote-access.md`. The options compared
were:
1. **Office VPN account** for the owner, if the office has a VPN. Nothing on the server is exposed.
2. **Mesh VPN (WireGuard / Tailscale) on the server**, with IT's approval: an outbound-only link,
   no open port, only the owner's devices can reach the server, and only its SSH port (no route to
   the rest of the office LAN).
3. **Router port-forward to SSH, allowed only from the owner's home IP**, key-only login for that
   path. Breaks when the home IP changes; last resort.

**Revocation when the experiments are done:** remove the VPN account / mesh node / forward and rule,
remove the laptop's key from `~/.ssh/authorized_keys` if wanted, stop any service, delete `~/sfs`,
`~/envs/sfs` and `~/.config/sfs/secret`. Checklist in the IT note.

## 4. Code changes

| File | Change |
|---|---|
| `src/sfs/teacher/gate.py` | G0 shortcut check compares against the 16-bit noise floor (slow pass, two attention kernels) instead of a fixed 0.1; G1 gets a whole-validation mode (`--k 0`, E2). **E4 (built 2026-11-03):** G0 uses the scorer's device hooks (works for a remote scorer), `--remote`, `--max-vram-gb` (laptop 7.5); `remote-check`: the 2B served remotely vs the laptop's own scores, judged by the shortcut rule against the laptop's eager-vs-sdpa noise floor |
| `src/sfs/teacher/vlm.py` | **E4:** device hooks `sync` / `reset_peak_memory` / `peak_memory_gb` / `set_attn`; a quantised checkpoint (FP8) loads straight onto the device, unquantised ones keep the recorded load path |
| `src/sfs/teacher/serve.py` (new, office server) | HTTP scoring service around `VLMScorer`: `POST /score` (a clip's frames as lossless PNG + questions → scores), `GET /info`, `POST /control` (peak memory, attention kernel for G0); binds 127.0.0.1 (SSH tunnel); shared secret from `SFS_SERVE_SECRET` or `~/.config/sfs/secret`; size limits (64 MB, 32 frames, 512 questions); waits for free GPU memory before loading; GPU out-of-memory → 503; one GPU job at a time. HTTP from the standard library; PNG via OpenCV (already a dependency) |
| `src/sfs/teacher/remote.py` (new, laptop) | `RemoteScorer`: same interface as `VLMScorer` (incl. device hooks); retries connection errors and 502/503/504 with exponential back-off, stops at once on 401/400/413; refuses requests over the server limit; records the server's model id and revision |
| `src/sfs/teacher/run.py` | `--remote <url>` (E4, `make_scorer` checks the served kind); timing uses the scorer's `sync`; `--frames 16` already exists |
| `tests/test_remote.py` (new), `tests/conftest.py` | E4 tests (§5); the tiny random-weight Qwen3-VL moved into a shared fixture |
| `src/sfs/eval/cascade_run.py` | `--fixed-choices` (E1: reuse Result 04's γ and β, select nothing new); `--extra-scores` accepts several stores per model |
| `src/sfs/eval/queries.py` + `cascade_run.py` | optional activity mapping (E1), applied before queries are built |
| `queries/phrasings_merged_turns.csv` (new) | the 3 neutral phrasings, once approved (the frozen `phrasings.csv` is not touched) |
| `docs/results/05-vlm-followups.md` (new) | Result 05: E1–E5 |
| README, EXPERIMENT_LOG, Result 04 (E5 row only) | updated in the same change |

## 5. Tests

- **Service + client, end to end on the laptop (local port, tiny random-weight Qwen3-VL):** remote
  scores equal local scores exactly; a wrong or missing secret is refused; an oversized request is
  refused; the client retries after a dropped connection; the secret never appears in logs or files.
- **Activity mapping:** merged relevance sets equal left ∪ right; the 22-activity queries are
  unchanged when no mapping is given.
- **Whole-validation G1:** equals the top-50 result when the corpus has ≤ 50 clips.
- **G0 noise-floor check:** passes for bf16-sized noise, fails for a real mismatch.
- **Before declaring done:** the full suite, plus the slow real-model tests (this touches the scorer).

## 6. State after this change

- Result 05 doc with: the Q1 "VLM on every candidate" point (E5); "vehicle turns" merged view (E1);
  whether the top-50 G1 shortcut holds on the whole validation set (E2); whether 16 frames rescue
  the 2B VLM (E3); whether a bigger VLM passes G1, and if so whether distillation then helps (E4).
- Results 02–04 unchanged; the shortcut check reworded and its old outcome kept in the record.
- Open for later: full slow re-scoring of eval (Q4 option C, ~11–12 h), to run if there is time or a
  bigger VLM shows gains.

## 7. Risks

| Risk | Handling |
|---|---|
| Office GPU too small for the 8B in bf16 | Choose by the §3 E4 table; quantised versions are a separate, labelled result |
| Service reachable from outside the office | Office network only, shared secret, SSH tunnel for home access |
| Serving changes the scores | Set-up check: the 2B served remotely must match the laptop's 2B scores |
| Laptop sleeps during long runs (seen on 2026-10-31) | All jobs resume; please keep it awake for E2 / E3 |
| Merged view read as replacing the frozen query set | Always reported as an extra, labelled view |

## 8. Decisions

**Approved by the owner, 2026-11-03:**
1. **E1:** an extra merged view, no retraining, with the three phrasings in §3 E1 ("go with your
   recommendation"). *Assumption:* this approval includes the phrasings as written; change them any
   time before E1's eval run.
4. **Distillation:** students are retrained only if the VLM passes G1 (E3 and E4).

**Still open:**
2. **E2:** skip Reranker-context (recommended), or include it (~4 h more)?
3. **E4, information about the office server:** GPU model and memory, operating system, free disk
   space, whether Python/conda with CUDA is installed, and whether the laptop can reach it on the
   office network. The model choice follows from the memory (§3 E4 table); recommended: 8B in bf16
   first.
5. **Order:** E5 (running) → E1 → E2 → E3 on the laptop; E4 when the office server is set up. The
   search demo comes after this.
6. **Approval of this spec as a whole** for E2, E3 and E4. E1 starts now on your go-ahead.

**Update 2026-11-03 (D-027):** E2 and E3 approved and started. Decision 2 taken at the recommended
default (Reranker-context skipped); say so if you want it added. Decision 3 (office server) arrives
the next morning; E4 starts after it.

**Update 2026-11-03 (D-028):** E5, E1, E2, E3 complete — `docs/results/05-vlm-followups.md`. E2 by
the fixed rule: **not the same** (same verdict, gains 0.012–0.019 lower on the whole set). E4's G1
depth is therefore an owner decision: whole validation (rule), top 50, or top 50 first and whole
validation only if top 50 passes (recommended). E3: G1 fails at 16 frames; E3 stopped (no labels, no
retraining). Optional new item: retrain the student on the merged turn label (not approved).

**Update 2026-11-03 (D-029):** office server surveyed (16 GB GPU, shared). Owner chose
Qwen3-VL-8B-Instruct-FP8, "wait for free GPU", and G1 at top 50 first, whole validation only if top 50
passes. Access by SSH tunnel to a localhost-only service. See "E4 revised after the server survey".

**Update 2026-11-03 (D-034):** Gemma stopped and removed (owner). E4 runs directly on the server
(the crop store is there); set-up check = the server's 2B re-scores the 4,800 G1 pairs and must match
the laptop within the bf16 noise floor (max ≤ 0.899, rank correlation ≥ 0.99). The 8B job waits for
≥ 14.5 GB free GPU memory.

**Update 2026-11-03 (D-036, D-037):** 8B G0 pass, G1 fails by 0.0009 (B +0.0191). Owner: continue on the free GPU →
the 8B is measured as the final cascade stage (fusion weight on validation, same eval top-k pairs as the 2B); no
distillation. Jobs hold their GPU memory for the run (`sfs.teacher.run --reserve-gb`); the team's Ollama is not
touched.

**Update 2026-11-03 (D-041):** Qwen2.5-VL 3B and 7B both fail G1 (E7 closed, kept for nexa-vlm). Owner: hold the GPU
for the whole experimentation (`sfs.teacher.queue --watch`, 14.5 GB) and test the bigger model: Qwen3-VL-32B-Instruct-FP8
(rev 4bf2c2f3, 35.5 GB) with the CPU/GPU split of `sfs.teacher.offload` (exact; 8B split check identical), unpinned on
the shared server. Same G0/G1 rules.

**Update 2026-11-04 (D-043..D-046):** the 32B passes G1 at both depths. Owner approved the chain (teacher labels
→ students → cascade rows, short list only), running in the server holder from 00:11. The steps after the labels run
unattended on the server (`~/sfs/after_labels.sh`): a reproducibility check, then GTKD/KD students. Code: `sfs.rerank.train
--text-emb` and `scripts/export_text_embeddings.py` (server has no SigLIP 2 weights; bit-exact, test added).

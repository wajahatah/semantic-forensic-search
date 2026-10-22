# Colab runbook — Manifest v2 + embedding job

Runtime: **T4 GPU** (`Runtime > Change runtime type`). Free-tier sessions time out, so every step
writes to Drive and is resumable — if the session dies, reconnect and re-run from Cell 1; finished
videos are skipped.

Run cells in order. Every cell is copy-paste.

---

### Cell 1 — mount Drive, set paths

```python
from google.colab import drive
drive.mount('/content/drive')

import os
BASE = '/content/drive/MyDrive/DL-Project/meva'
ANNO = next(p for p in [f'{BASE}/annotations/meva-data-repo',
                        '/content/drive/MyDrive/meva/annotations/meva-data-repo'] if os.path.isdir(p))
VIDEOS   = f'{BASE}/videos'
MANIFEST = f'{BASE}/manifest_v2.csv'
SHARDS   = f'{BASE}/embeddings_v2'   # v2: geom parser fixed; v1 output is invalid
print('annotations:', ANNO)
```

### Cell 2 — install the package

Open `colab/sfs_bundle.py` from the repo, copy **the whole file**, paste into this cell, run.
It prints `sfs bundle installed ... | source commit <sha>`. Regenerate it locally with
`python scripts/build_colab_bundle.py` whenever `src/sfs` changes.

### Cell 3 — dependencies

```python
!pip install -q -U "transformers>=4.49" sentencepiece
import transformers, torch; print(transformers.__version__, torch.cuda.get_device_name(0))
```

### Cell 4 — manifest v2 (splits, dedupe, leakage check)

```python
from sfs.data.manifest import main as build_manifest
build_manifest(['--anno', ANNO, '--videos', VIDEOS, '--out', MANIFEST])
```

**Paste the full output back to Claude.** If it prints `WARNING: eval has ... instances`, stop here —
the spec's D1 gate says the eval split needs a targeted download first.

### Cell 5 — smoke test on ONE video, with crops saved for inspection

```python
from sfs.embed.run import main as embed
embed(['--manifest', MANIFEST, '--out', f'{SHARDS}_smoke', '--limit-videos', '1', '--debug-crops', '18'])

from IPython.display import Image, display
import glob
for p in sorted(glob.glob(f'{SHARDS}_smoke/_debug/*.jpg'))[:18]:
    print(os.path.basename(p)); display(Image(p, width=220))
```

**Look at the crops.** `tight` should be centred on the person/vehicle doing the activity,
`context` should show them plus surroundings, `full` the whole frame. If tight crops show empty
road or wall, the boxes are misaligned — stop and report.

### Cell 6 — full run

```python
embed(['--manifest', MANIFEST, '--out', SHARDS])
```

Prints one line per video. Safe to interrupt and re-run — completed videos print `skipped`.

**Watch `fallback_rate` on each line.** It is the fraction of rows whose crop had no box and
used the whole frame instead. Expect roughly 0.0–0.1. If most lines show values near 1.0, stop
and report — that was the v1 failure. A clip whose actors have no boxes at all now stops with
`none of N actors have box tracks` instead of silently continuing.

### Cell 7 — consolidate for download

```python
import glob
from sfs.store.shards import consolidate
for enc in ['siglip', 'siglip2']:
    meta = consolidate(sorted(glob.glob(f'{SHARDS}/{enc}/*.npz')), f'{BASE}/embeddings_v2_{enc}.npz')
    print(enc, meta['encoder_id'], meta['n_shards'], 'videos')
!ls -lh {BASE}/embeddings_v2_*.npz {MANIFEST}
```

Then download `embeddings_v2_siglip.npz` and `embeddings_v2_siglip2.npz` from Drive
into the local `data/` folder.

---

# Crop store (spec-crops-queries-baseline, Part 1)

Saves the tight + context crops as JPEGs (one packed file per video) so LoRA training and the VLM
teacher never re-read the video. Run **Cells 1–3 above first** (paste the current bundle in Cell 2).

### Cell C1 — extract crops (~1–2 h, resumable)

```python
CROPS = f'{BASE}/crops_v1'
from sfs.data.extract_crops import main as crops
crops(['--manifest', MANIFEST, '--out', CROPS])
```

One line per video. Safe to interrupt and re-run — finished videos print `skipped`. Ends with
`index: N crops in 252 shards | failed=0`. Expect ~147,000 crops and roughly 3–4 GB.

### Cell C2 — integrity gate (must pass)

```python
crops(['verify', '--crops', CROPS, '--embeddings', f'{BASE}/embeddings_v2_siglip2.npz'])
```

Re-embeds 2,000 stored crops and compares them with the original embeddings. Must print
`INTEGRITY PASSED` (mean cosine ≥ 0.98). **Paste the output back** — if it fails, stop.

Nothing needs downloading from this step: the crops stay on Drive for the training steps.

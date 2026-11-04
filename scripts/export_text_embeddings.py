"""Export the SigLIP 2 text embeddings of the training captions and the eval phrasings, so student
training (`sfs.rerank.train train --text-emb`) can run on a machine without the SigLIP 2 weights
(office server, D-045). Each group is embedded in the same single call training makes, so the
values are bit-identical to what training computes itself.

  python scripts/export_text_embeddings.py --out data/rerank/text_emb_siglip2.npz
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def main(argv=None) -> int:
    from sfs.embed.encoder import ENCODERS
    from sfs.eval.queries import load_phrasings
    from sfs.rerank.train import _text_embeddings
    from sfs.train.captions import load_captions

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--captions", default="queries/train_captions.csv")
    ap.add_argument("--phrasings", default="queries/phrasings.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    caps, phr = load_captions(a.captions), load_phrasings(a.phrasings)
    # the two calls training makes, in its order (sfs.rerank.train.main)
    groups = [[t for c in sorted(caps) for t in caps[c]], [t for x in sorted(phr) for t in phr[x]]]
    out = {"meta": json.dumps({"model_id": ENCODERS["siglip2"], "captions": a.captions, "phrasings": a.phrasings})}
    for i, texts in enumerate(groups):
        out[f"texts_{i}"] = np.array(texts)
        out[f"emb_{i}"] = _text_embeddings(texts)
    np.savez(a.out, **out)
    print(f"wrote {a.out}: groups of {[len(g) for g in groups]} texts")
    return 0


if __name__ == "__main__":
    sys.exit(main())

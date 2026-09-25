"""
MEVA acquisition from Google Colab -> Google Drive.

WHY COLAB: gitlab.kitware.com is unreachable from the local network (WinError 10060).
Colab runs on Google's network, so the block does not apply. It also downloads from
S3 at datacenter speed and writes straight to your 5 TB Drive, using none of your
home bandwidth.

HOW TO USE
  1. colab.research.google.com -> New notebook
  2. Paste each CELL below into its own cell, run in order.
  3. Runtime > Change runtime type > GPU  (only needed for CELL 5)

Total download here is ~25 GB, NOT the full 587 GB. See CELL 4 for why.
"""

# =====================================================================
# CELL 1 - mount Drive and create the project tree
# =====================================================================
CELL_1 = r'''
from google.colab import drive
drive.mount('/content/drive')

import os
ROOT = '/content/drive/MyDrive/meva'
for sub in ['annotations', 'examples', 'videos', 'embeddings']:
    os.makedirs(f'{ROOT}/{sub}', exist_ok=True)
print('project root:', ROOT)
!df -h /content/drive | tail -1
'''

# =====================================================================
# CELL 2 - THE BLOCKER: clone the annotations repo
# This is the step that fails on the local machine.
# =====================================================================
CELL_2 = r'''
# reachability check first
!curl -s -o /dev/null -w "gitlab.kitware.com -> HTTP %{http_code}\n" \
    --max-time 30 https://gitlab.kitware.com/meva/meva-data-repo

# full annotations: 64 h evaluation + 120.2 h training
%cd /content
!git clone --depth 1 https://gitlab.kitware.com/meva/meva-data-repo.git

!du -sh /content/meva-data-repo
!find /content/meva-data-repo -name "*.yml" | wc -l
!find /content/meva-data-repo -maxdepth 2 -type d | head -20

# copy to Drive so it survives the runtime being recycled
!cp -r /content/meva-data-repo /content/drive/MyDrive/meva/annotations/
print('ANNOTATIONS SAVED TO DRIVE')
'''

# =====================================================================
# CELL 3 - the labelled example set (900 MB, 121 clips, 36 activities)
# =====================================================================
CELL_3 = r'''
import urllib.request, urllib.parse, xml.etree.ElementTree as ET, os
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
BASE = "https://mevadata-public-01.s3.amazonaws.com/"

def list_all(prefix):
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token: q["continuation-token"] = token
        with urllib.request.urlopen(BASE + "?" + urllib.parse.urlencode(q), timeout=60) as r:
            root = ET.fromstring(r.read())
        out += [(c.find(NS+"Key").text, int(c.find(NS+"Size").text))
                for c in root.findall(NS+"Contents")]
        token = root.findtext(NS+"NextContinuationToken")
        if not token: return out

def grab(prefix, dest):
    objs = list_all(prefix)
    print(f"{prefix}: {len(objs)} objects, {sum(s for _,s in objs)/2**30:.2f} GB")
    for i, (k, s) in enumerate(objs, 1):
        local = os.path.join(dest, *k.split("/"))
        os.makedirs(os.path.dirname(local), exist_ok=True)
        if os.path.exists(local) and os.path.getsize(local) == s:
            continue
        urllib.request.urlretrieve(BASE + urllib.parse.quote(k), local)
        if i % 25 == 0: print(f"  {i}/{len(objs)}")
    print("done:", prefix)

grab("examples/", "/content/drive/MyDrive/meva/examples")
'''

# =====================================================================
# CELL 4 - a REAL-FOOTAGE subset. Read this before running.
#
# The full drops-123-r13/ release is 443 GB. You do not need it, because:
#
#   * Video is read ONCE, to compute embeddings. After that the pipeline
#     only touches vectors.
#   * 4,259 clips x 512 dims x 4 bytes = ~9 MB. The entire retrieval index
#     is smaller than a single video file.
#   * Reading many large files from a mounted Drive during training is slow
#     and will bottleneck you far worse than having less data.
#
# So: take a few hours of real footage, not the whole release.
# =====================================================================
CELL_4 = r'''
# Each date/hour folder is roughly 1.4 GB. Six hours ~ 8 GB.
HOURS = [
    "drops-123-r13/2018-03-05/09/",
    "drops-123-r13/2018-03-05/10/",
    "drops-123-r13/2018-03-07/16/",   # has matching example annotations
    "drops-123-r13/2018-03-09/10/",
    "drops-123-r13/2018-03-11/14/",
    "drops-123-r13/2018-03-13/20/",
]
for h in HOURS:
    grab(h, "/content/drive/MyDrive/meva/videos")

!du -sh /content/drive/MyDrive/meva
'''

# =====================================================================
# CELL 5 - verify: decode a clip, parse an annotation, confirm both work
# =====================================================================
CELL_5 = r'''
import cv2, glob, yaml, os

vids = sorted(glob.glob('/content/drive/MyDrive/meva/examples/examples/videos/*.mp4'))
print(f"{len(vids)} example clips")
cap = cv2.VideoCapture(vids[0])
ok, frame = cap.read()
print("decode:", ok, "| shape:", None if not ok else frame.shape,
      "| frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
      "| fps:", cap.get(cv2.CAP_PROP_FPS))
cap.release()

acts = sorted(glob.glob('/content/drive/MyDrive/meva/examples/examples/annotations/*/*/*.activities.yml'))
print(f"\n{len(acts)} activity annotation files")
with open(acts[0]) as f:
    doc = yaml.safe_load(f)
for row in doc[:4]:
    print(" ", row)

# activity vocabulary -> this is your natural-language query vocabulary
import re, collections
vocab = collections.Counter()
for p in acts:
    for m in re.finditer(r'act\d+:\s*\{\s*(\w+):', open(p).read()):
        vocab[m.group(1)] += 1
print(f"\n{len(vocab)} distinct activity types across the annotation set:")
for a, n in vocab.most_common():
    print(f"  {n:>4}  {a}")
'''

if __name__ == "__main__":
    for name, cell in [("CELL 1 - mount Drive", CELL_1),
                       ("CELL 2 - clone annotations (THE BLOCKER)", CELL_2),
                       ("CELL 3 - example set, 900 MB", CELL_3),
                       ("CELL 4 - real footage subset, ~8 GB", CELL_4),
                       ("CELL 5 - verify decode + annotations", CELL_5)]:
        print("\n" + "=" * 70)
        print("#  " + name)
        print("=" * 70)
        print(cell)

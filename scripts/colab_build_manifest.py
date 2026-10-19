"""
CELL 7 - Annotation inventory + video alignment + query-set manifest.

Run in Colab after the setup notebook. Answers three questions:
  1. What is actually in the 13,968 annotation files?
  2. How many annotated activity instances do you have MATCHING VIDEO for?
  3. Produce the manifest that the natural-language query set is built from.

Output: /content/drive/MyDrive/DL-Project/meva/manifest.csv
"""

CELL_7 = r'''
import os, re, glob, yaml, csv, collections, json

ROOT  = '/content/drive/MyDrive/DL-Project/meva'
ANNO  = next((p for p in [f'{ROOT}/annotations/meva-data-repo', '/content/drive/MyDrive/meva/annotations/meva-data-repo'] if os.path.isdir(p)), None)
assert ANNO, 'meva-data-repo not found on Drive - check where Cell 2 copied it'
VIDEO = f'{ROOT}/videos'

# ---------- 1. what is in the repo ----------
act_files = glob.glob(f'{ANNO}/**/*.activities.yml', recursive=True)
geom_files = glob.glob(f'{ANNO}/**/*.geom.yml', recursive=True)
print(f'activity files : {len(act_files)}')
print(f'geometry files : {len(geom_files)}')

# annotation dirs (train/test split lives here)
dirs = collections.Counter('/'.join(os.path.relpath(os.path.dirname(p), ANNO).split(os.sep)[:4]) for p in act_files)
print('\nannotation groups:')
for d, n in dirs.most_common(12):
    print(f'  {n:>6}  {d}')

# ---------- 2. which videos do we actually have ----------
have = {}
for p in glob.glob(f'{VIDEO}/**/*.avi', recursive=True):
    # 2018-03-05.09-49-41.09-50-01.school.G339.r13.avi -> key without .r13.avi
    stem = os.path.basename(p).replace('.r13.avi', '')
    have[stem] = p
print(f'\nvideo files on disk: {len(have)}')

# ---------- 3. parse activities, align to video ----------
rows, vocab, matched_vocab = [], collections.Counter(), collections.Counter()
no_video = 0

for ap in act_files:
    stem = os.path.basename(ap).replace('.activities.yml', '')
    vpath = have.get(stem)
    try:
        doc = yaml.load(open(ap), Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))
    except Exception:
        continue
    if not doc:
        continue
    for item in doc:
        act = item.get('act') if isinstance(item, dict) else None
        if not act:
            continue
        # activity name lives under actN: {Name: 1.0}
        name = None
        for k, v in act.items():
            if re.fullmatch(r'act\d+', k) and isinstance(v, dict) and v:
                name = next(iter(v))
                break
        if not name:
            continue
        vocab[name] += 1
        ts = act.get('timespan') or [{}]
        span = (ts[0].get('tsr0') or [None, None]) if isinstance(ts[0], dict) else [None, None]
        n_actors = len(act.get('actors') or [])
        if vpath:
            matched_vocab[name] += 1
            rows.append({
                'clip_stem': stem, 'video_path': vpath, 'activity': name,
                'frame_start': span[0], 'frame_end': span[1],
                'n_actors': n_actors,
                'site': stem.split('.')[3] if len(stem.split('.')) > 3 else '',
                'camera': stem.split('.')[4] if len(stem.split('.')) > 4 else '',
                'date': stem.split('.')[0],
            })
        else:
            no_video += 1

print(f'\n=== ACTIVITY INSTANCES ===')
print(f'total in annotations      : {sum(vocab.values()):,}')
print(f'WITH matching video       : {len(rows):,}   <-- your usable corpus')
print(f'without video (skipped)   : {no_video:,}')
print(f'distinct activity types   : {len(vocab)} total, {len(matched_vocab)} usable')

print(f'\n--- usable instances per activity (top 25) ---')
for a, n in matched_vocab.most_common(25):
    print(f'  {n:>6}  {a}')

# ---------- 4. write manifest ----------
out = f'{ROOT}/manifest.csv'
if rows:
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'\nmanifest written: {out}  ({len(rows):,} rows)')
else:
    print('\n!! NO ROWS - annotations and videos do not align. Check naming/paths above.')

json.dump({'vocab_all': dict(vocab), 'vocab_matched': dict(matched_vocab)},
          open(f'{ROOT}/vocab.json', 'w'), indent=1)
print('vocab written:', f'{ROOT}/vocab.json')
'''

# =====================================================================
# CELL 8 - sanity check: can we cut a real clip for an activity instance?
# =====================================================================
CELL_8 = r'''
import pandas as pd, cv2, os
ROOT = '/content/drive/MyDrive/DL-Project/meva'
df = pd.read_csv(f'{ROOT}/manifest.csv')
print(df.shape)
print(df['activity'].value_counts().head(15))
print()

# pick one instance and extract its frames -> proves the manifest is usable
r = df[df.frame_start.notna()].iloc[0]
print('instance:', r.activity, '| frames', r.frame_start, '->', r.frame_end)
cap = cv2.VideoCapture(r.video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame_start))
ok, frame = cap.read()
print('decode ok:', ok, '| shape:', None if not ok else frame.shape,
      '| duration_s:', round((r.frame_end - r.frame_start) / max(fps, 1), 2))
cap.release()

# instance duration distribution - informs N_max in the spec
df['dur_frames'] = df.frame_end - df.frame_start
print('\nactivity duration (frames):')
print(df.dur_frames.describe())
'''

if __name__ == '__main__':
    for name, cell in [('CELL 7 - annotation inventory + manifest', CELL_7),
                       ('CELL 8 - verify a clip can be cut', CELL_8)]:
        print('\n' + '=' * 70)
        print('#  ' + name)
        print('=' * 70)
        print(cell)

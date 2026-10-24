"""Fakeddit manifest (run from $CURB_DATA; reads meta/v2/multimodal_only_samples/multimodal_train.tsv).

  1. keep posts whose image domain is available (reddit preview CDN);
  2. keep posts from 2016-01-01 on;
  3. balance the two labels with per-subreddit quotas;
  4. sort by created_utc into one time-ordered stream (data/build_streams.py keeps the first 80,000 posts).
"""
import pandas as pd, numpy as np, os

TARGET_PER_CLASS = 65_000
OK_DOM = {'external-preview.redd.it', 'preview.redd.it', 'i.redd.it'}
SINCE = 1451606400  # 2016-01-01
SEED = 12345
SRC = 'meta/v2/multimodal_only_samples/multimodal_train.tsv'


def waterfill(counts: dict, total: int) -> dict:
    """Equal quotas per key, capped by availability, redistributing leftovers."""
    quota = {k: 0 for k in counts}
    remaining, active = total, set(counts)
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        progressed = False
        for k in sorted(active):
            take = min(share, counts[k] - quota[k], remaining)
            if take <= 0:
                active.discard(k); continue
            quota[k] += take; remaining -= take; progressed = True
            if quota[k] >= counts[k]:
                active.discard(k)
            if remaining <= 0:
                break
        if not progressed:
            break
    return quota


df = pd.read_csv(SRC, sep='\t', low_memory=False)
df = df[df['image_url'].notna() & df['clean_title'].notna()].copy()
df['dom'] = df['image_url'].str.extract(r'https?://([^/]+)/')[0]
df = df[df['dom'].isin(OK_DOM)]
df = df[df['created_utc'] >= SINCE]
df = df[df['clean_title'].astype(str).str.strip().str.len() >= 3]
df = df.drop_duplicates(subset=['id'])

parts = []
for lab in (0, 1):
    sub = df[df['2_way_label'] == lab]
    counts = sub['subreddit'].value_counts().to_dict()
    quota = waterfill(counts, TARGET_PER_CLASS)
    for s, q in quota.items():
        if q > 0:
            parts.append(sub[sub['subreddit'] == s].sample(q, random_state=SEED))

sel = pd.concat(parts).sort_values('created_utc', kind='mergesort').reset_index(drop=True)
sel['uid'] = sel['id'].astype(str)
sel['fname'] = sel['uid'] + '.jpg'

os.makedirs('manifest', exist_ok=True)
cols = ['uid','fname','image_url','clean_title','created_utc','subreddit','author','2_way_label']
sel[cols].to_parquet('manifest/d1_manifest.parquet', index=False)

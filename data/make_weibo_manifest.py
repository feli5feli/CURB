"""Weibo manifest: Weibo multimodal rumour set (Jin et al. 2017), as shipped by EANN-KDD18.

Records are 3 lines each: metadata | image urls | text. Labels come from the file name
(rumor / nonrumor); every post carries its own image (fetched with data/fetch_images.py).
"""
import argparse, os, re, sys
import numpy as np, pandas as pd

SRC = "https://raw.githubusercontent.com/yaqingwang/EANN-KDD18/master/data/weibo/tweets"
FILES = [("train_rumor", 0), ("test_rumor", 0), ("train_nonrumor", 1), ("test_nonrumor", 1)]
URL_RE = re.compile(r"https?://[^|\s]+?\.(?:jpg|jpeg|png|gif)", re.I)


def parse(path, label):
    lines = open(path, encoding="utf-8", errors="ignore").read().split("\n")
    out = []
    for i in range(0, len(lines) - 2, 3):
        meta, imgs, text = lines[i], lines[i + 1], lines[i + 2].strip()
        urls = URL_RE.findall(imgs)
        parts = meta.split("|")
        if not urls or len(text) < 4 or len(parts) < 10:
            continue
        # the two halves of the release use different time formats: the rumour files
        # carry local Beijing timestamps, the non-rumour files epoch milliseconds
        raw = parts[4].strip()
        if raw.isdigit() and len(raw) >= 12:
            ts = pd.to_datetime(int(raw), unit="ms", utc=True)
        else:
            ts = pd.to_datetime(raw, errors="coerce")
            if pd.notna(ts):
                ts = ts.tz_localize("Asia/Shanghai", ambiguous="NaT",
                                    nonexistent="NaT").tz_convert("UTC")
        if pd.isna(ts):
            continue
        out.append(dict(uid=parts[0], created=ts, text=text[:300], image_url=urls[0], label=label))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="dir holding weibo_*.txt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for name, lab in FILES:
        p = os.path.join(a.raw, f"weibo_{name}.txt")
        if not os.path.exists(p):
            sys.exit(f"missing {p} -- fetch {SRC}/{name}.txt")
        rows += parse(p, lab)
    df = pd.DataFrame(rows).drop_duplicates(subset=["uid"]).drop_duplicates(subset=["image_url"])
    df["created"] = pd.to_datetime(df["created"], utc=True)
    df = df.sort_values("created", kind="mergesort").reset_index(drop=True)
    df["fname"] = [os.path.basename(u).split("?")[0] for u in df["image_url"]]
    df["uid"] = df["uid"].astype(str)
    # the two label groups were collected over different periods: interleave them at the global ratio,
    # keeping time order within each group; the stream position is the clock
    order, queues = [], {k: list(g.index) for k, g in df.groupby("label", sort=True)}
    counts = {k: len(v) for k, v in queues.items()}
    total = sum(counts.values())
    taken = {k: 0 for k in queues}
    for step in range(total):
        # take from whichever group is furthest behind its global share
        k = min(queues, key=lambda k: (taken[k] / counts[k]) if counts[k] else 2.0)
        order.append(queues[k][taken[k]]); taken[k] += 1
        if taken[k] >= counts[k]:
            queues.pop(k); counts.pop(k); taken.pop(k)
    df = df.loc[order].reset_index(drop=True)
    df["created_utc"] = np.arange(len(df), dtype="int64")   # stream position is the clock

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df[["uid", "fname", "image_url", "text", "created_utc", "label"]].to_parquet(a.out, index=False)


if __name__ == "__main__":
    main()

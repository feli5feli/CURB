"""Image fetcher for any manifest with `fname` / `image_url` columns.

Rate-limited (AIMD), 429-aware, resumable. Rejects CDN placeholders, resizes to a 224 short side and caches as JPEG
sharded by the first two characters of the filename.
"""
import argparse, io, os, random, threading, time
import pandas as pd, requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--referer", default=None, help="some CDNs (Sina) require it")
ap.add_argument("--threads", type=int, default=12)
ap.add_argument("--rate", type=float, default=8.0)
a = ap.parse_args()

UA = "curb-research-image-fetcher/1.0"
PLACEHOLDER = {(130, 60), (336, 478), (161, 81)}
RATE_MIN, RATE_MAX, MAX_RETRY, SHORT, MIN_SIDE = 2.0, 40.0, 5, 224, 48


class Bucket:
    def __init__(self, rate):
        self.rate, self.tokens, self.t = rate, rate, time.time()
        self.lk, self.wins = threading.Lock(), 0

    def take(self):
        while True:
            with self.lk:
                now = time.time()
                self.tokens = min(self.rate, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= 1:
                    self.tokens -= 1; return
                wait = (1 - self.tokens) / self.rate
            time.sleep(min(wait, 0.5))

    def win(self):
        with self.lk:
            self.wins += 1
            if self.wins >= 150:
                self.wins = 0; self.rate = min(RATE_MAX, self.rate + 1.0)

    def throttle(self):
        with self.lk:
            self.wins = 0; self.rate = max(RATE_MIN, self.rate * 0.7)


bucket = Bucket(a.rate)
df = pd.read_parquet(a.manifest)
for p in {os.path.join(a.out, f[:2]) for f in df["fname"]}:
    os.makedirs(p, exist_ok=True)
tl = threading.local()


def session():
    if not hasattr(tl, "s"):
        s = requests.Session(); s.headers["User-Agent"] = UA
        if a.referer:
            s.headers["Referer"] = a.referer
        for scheme in ("https://", "http://"):
            s.mount(scheme, requests.adapters.HTTPAdapter(pool_maxsize=a.threads, max_retries=0))
        tl.s = s
    return tl.s


def fetch(row):
    dst = os.path.join(a.out, row.fname[:2], row.fname)
    if os.path.exists(dst) and os.path.getsize(dst) > 512:
        return
    for attempt in range(MAX_RETRY):
        bucket.take()
        try:
            r = session().get(row.image_url, timeout=25)
        except Exception:
            time.sleep(1.5 * (attempt + 1)); continue
        if r.status_code == 429:
            bucket.throttle()
            time.sleep(min(20, 2 ** attempt) + random.random() * 2); continue
        if r.status_code != 200:
            return
        try:
            im = Image.open(io.BytesIO(r.content))
            if im.size in PLACEHOLDER or min(im.size) < MIN_SIDE:
                return
            im = im.convert("RGB")
            w, h = im.size
            if min(w, h) > SHORT:
                sc = SHORT / min(w, h)
                im = im.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.BICUBIC)
            im.save(dst, "JPEG", quality=90, optimize=True)
            bucket.win()
        except Exception:
            pass
        return


with ThreadPoolExecutor(a.threads) as ex:
    list(ex.map(fetch, df.itertuples()))

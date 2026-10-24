"""Build the benign contribution tensor U[T, m, 2d] of one (dataset, seed) stream.

Every service identity owns one unique, fixed (image pipeline, text template) pair for the whole stream; its
contribution to a post is that post re-encoded through its pipeline by the shared frozen CLIP encoder
(data/encode_pipelines.py), L2-normalised per modality. No representation-space transformation or noise is added.

  --mode standard        N identities, each post observed by m of them, Sybil share alpha among the observers
  --mode identity_split  Sybil scaling: m_h honest observers + a reserved pool of s_max Sybil identities
  --switch_*             benign pipeline upgrade: switch_nodes honest identities move to an unused pipeline pair at
                         fraction switch_at of the stream, keeping their identity
  --node_mode kmeans     identities from k-means clustering of the text embeddings (Weibo); default: subreddit-author
                         partition (Fakeddit)

Outputs (in --out): uploads.npy (fp16), obs.npy, labels.npy, sybil_mask.npy, meta.json.
"""
import argparse, hashlib, os, json, numpy as np, pandas as pd, torch


# ---- Fakeddit identities: each subreddit is split into author-hash buckets, quotas proportional to its size ----
def _alloc(counts, n_nodes):
    """Largest-remainder allocation of n_nodes across subreddits, min 1 each."""
    keys = list(counts)
    tot = sum(counts.values())
    raw = {k: max(1.0, n_nodes * counts[k] / tot) for k in keys}
    base = {k: max(1, int(np.floor(raw[k]))) for k in keys}
    while sum(base.values()) > n_nodes:                      # trim the largest
        k = max(base, key=lambda k: (base[k], counts[k]))
        if base[k] > 1:
            base[k] -= 1
        else:
            break
    rem = n_nodes - sum(base.values())
    order = sorted(keys, key=lambda k: -(raw[k] - base[k]))
    for i in range(rem):
        base[order[i % len(order)]] += 1
    return base


def assign_nodes(subreddits, authors, n_nodes, seed=0):
    """Node id per post: each subreddit is split by a stable author hash, with a quota proportional to its size."""
    subs = np.asarray(subreddits)
    auth = np.asarray(authors)
    counts = {s: int((subs == s).sum()) for s in np.unique(subs)}
    node_id = np.full(len(subs), -1, np.int32)
    alloc = _alloc(counts, n_nodes)
    nid = 0
    for s in sorted(alloc, key=lambda k: -counts[k]):
        k = alloc[s]
        idx = np.where(subs == s)[0]
        if k == 1:
            node_id[idx] = nid
            nid += 1
            continue
        h = np.array([int(hashlib.md5(f"{seed}|{s}|{a}".encode()).hexdigest()[:8], 16) % k
                      for a in auth[idx]], np.int32)
        for b in range(k):
            node_id[idx[h == b]] = nid
            nid += 1
    assert nid == n_nodes, f"allocated {nid} nodes, wanted {n_nodes}"
    assert (node_id >= 0).all(), "every post must belong to a node"
    return node_id


# ---- observers of each post ----
def sample_observers(node_of_post, sybil_mask, m, alpha, rng):
    """Owner + (m-1) other identities, stratified so that the Sybil share among the m observers is alpha."""
    T = len(node_of_post)
    syb_nodes = np.where(sybil_mask)[0]
    hon_nodes = np.where(~sybil_mask)[0]
    obs = np.empty((T, m), np.int32)
    n_syb_target = int(round(alpha * m))
    for t in range(T):
        own = node_of_post[t]
        own_syb = bool(sybil_mask[own])
        need_s = n_syb_target - (1 if own_syb else 0)
        need_h = (m - 1) - need_s
        s = rng.choice(syb_nodes[syb_nodes != own] if own_syb else syb_nodes,
                       size=max(need_s, 0), replace=False) if need_s > 0 else np.empty(0, int)
        h = rng.choice(hon_nodes[hon_nodes != own] if not own_syb else hon_nodes,
                       size=max(need_h, 0), replace=False) if need_h > 0 else np.empty(0, int)
        row = np.concatenate([[own], s, h]).astype(np.int32)
        rng.shuffle(row[1:])
        obs[t] = row[:m]
    return obs


def sample_observers_split(node_of_post, sybil_mask, pool, m_h, rng):
    """Identity expansion: m_h honest observers (owner first when honest) + every reserved Sybil identity of the
    pool, in a fixed order; a run keeps the first S of the pool (cfg s_used)."""
    T = len(node_of_post)
    hon = np.where(~sybil_mask)[0]
    pool = np.asarray(pool, np.int32)
    obs = np.empty((T, m_h + len(pool)), np.int32)
    for t in range(T):
        own = node_of_post[t]
        if sybil_mask[own]:
            h = rng.choice(hon, size=m_h, replace=False)
        else:
            h = np.concatenate([[own], rng.choice(hon[hon != own], size=m_h - 1, replace=False)])
        obs[t, :m_h] = h
        obs[t, m_h:] = pool
    return obs


M, ALPHA = 10, 0.5          # active identities per post, Sybil share among them
p = argparse.ArgumentParser()
p.add_argument("--embed", required=True, help="data/encode_clip.py output: post index and text embeddings")
p.add_argument("--pipes_dir", required=True, help="parent dir of pipe_img_<name>/ and pipe_txt_<name>/")
p.add_argument("--img_pipes", required=True)
p.add_argument("--txt_pipes", required=True)
p.add_argument("--out", required=True)
p.add_argument("--n_nodes", type=int, default=100)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--limit", type=int, default=0, help="keep the first `limit` posts of the time-ordered stream")
p.add_argument("--node_mode", default="community", choices=["community", "kmeans"])
p.add_argument("--mode", default="standard", choices=["standard", "identity_split"])
p.add_argument("--m_h", type=int, default=10)
p.add_argument("--s_max", type=int, default=20)
p.add_argument("--switch_at", type=float, default=0.0)
p.add_argument("--switch_nodes", type=int, default=0)
p.add_argument("--switch_img_pipes", default="")
p.add_argument("--switch_txt_pipes", default="")
a = p.parse_args()

os.makedirs(a.out, exist_ok=True)
Y = np.load(os.path.join(a.embed, "txt.npy"))
idx = pd.read_parquet(os.path.join(a.embed, "index.parquet"))
order = np.argsort(idx["created_utc"].values, kind="mergesort")
Y, idx = Y[order], idx.iloc[order].reset_index(drop=True)

ipn = [v.strip() for v in a.img_pipes.split(",") if v.strip()]
tpn = [v.strip() for v in a.txt_pipes.split(",") if v.strip()]
key = idx["uid"].astype(str).values


def _load(sub, name, fn):
    """One pipeline's cache, aligned to the stream by post id."""
    d = os.path.join(a.pipes_dir, f"{sub}_{name}")
    u = pd.read_parquet(os.path.join(d, "index.parquet"))["uid"].astype(str).values
    pos = {x: i for i, x in enumerate(u)}
    take = np.array([pos.get(x, -1) for x in key])
    assert (take >= 0).all(), f"{int((take < 0).sum())} posts missing from {d}"
    return np.asarray(np.load(os.path.join(d, fn), mmap_mode="r")[take], np.float32)


PIPE = ([_load("pipe_img", n, "img.npy") for n in ipn], [_load("pipe_txt", n, "txt.npy") for n in tpn])
SW = None
if a.switch_nodes > 0:
    sin = [v.strip() for v in a.switch_img_pipes.split(",") if v.strip()]
    stn = [v.strip() for v in a.switch_txt_pipes.split(",") if v.strip()]
    SW = ([_load("pipe_img", n, "img.npy") for n in sin], [_load("pipe_txt", n, "txt.npy") for n in stn])
if a.limit:
    Y, idx = Y[:a.limit], idx.iloc[:a.limit].reset_index(drop=True)
    PIPE = ([v[:a.limit] for v in PIPE[0]], [v[:a.limit] for v in PIPE[1]])
    if SW is not None:
        SW = ([v[:a.limit] for v in SW[0]], [v[:a.limit] for v in SW[1]])
T, d = len(idx), PIPE[0][0].shape[1]

# ---- identities: the content region each post's owner belongs to ----
if a.node_mode == "kmeans":
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=a.n_nodes, random_state=a.seed, n_init=5, batch_size=1024)
    node_of_post = km.fit_predict(Y).astype(np.int32)
else:
    node_of_post = assign_nodes(idx["subreddit"].values, idx["author"].values, a.n_nodes, seed=a.seed)

# ---- Sybil identities and observer sets ----
rng = np.random.RandomState(a.seed + 77)
sybil = np.zeros(a.n_nodes, bool)
if a.mode == "identity_split":
    pool = rng.choice(a.n_nodes, a.s_max, replace=False)
    sybil[pool] = True
    obs = sample_observers_split(node_of_post, sybil, pool, a.m_h, np.random.RandomState(a.seed + 123))
    m = a.m_h + a.s_max
else:
    m = M
    sybil[rng.choice(a.n_nodes, int(round(ALPHA * a.n_nodes)), replace=False)] = True
    obs = sample_observers(node_of_post, sybil, m, ALPHA, np.random.RandomState(a.seed + 123))

# ---- one unique (image pipeline, text template) pair per identity ----
n_i, n_t = len(PIPE[0]), len(PIPE[1])
assert a.n_nodes <= n_i * n_t, f"{a.n_nodes} nodes need distinct pairs, only {n_i * n_t} exist"
slot = np.random.RandomState(a.seed + 4711).permutation(n_i * n_t)[:a.n_nodes]   # independent of the Sybil draw
pipe_of = np.stack([slot % n_i, slot // n_i], 1)
switch_of, switch_t0 = {}, None
if SW is not None:
    hon = np.where(~sybil)[0]
    sw_ids = np.sort(np.random.RandomState(a.seed + 9091).choice(hon, a.switch_nodes, replace=False))
    n_si, n_st = len(SW[0]), len(SW[1])
    assert a.switch_nodes <= n_si * n_st, "need one unused switch pair per switching node"
    switch_of = {int(j): (k % n_si, k // n_si) for k, j in enumerate(sw_ids)}
    switch_t0 = int(a.switch_at * T)

dev = "cuda" if torch.cuda.is_available() else "cpu"
post_idx = np.repeat(np.arange(T, dtype=np.int64), m)
node_idx = obs.reshape(-1).astype(np.int64)
order = np.argsort(node_idx, kind="stable")
uniq, counts = np.unique(node_idx[order], return_counts=True)
out = np.empty((T * m, 2 * d), np.float16)
off = 0
for j, c in zip(uniq.tolist(), counts.tolist()):
    sel = order[off:off + c]; off += c
    xr = torch.from_numpy(PIPE[0][pipe_of[j, 0]][post_idx[sel]]).to(dev)
    yr = torch.from_numpy(PIPE[1][pipe_of[j, 1]][post_idx[sel]]).to(dev)
    if int(j) in switch_of:
        late_np = post_idx[sel] >= switch_t0
        if late_np.any():
            si, ti = switch_of[int(j)]
            late = torch.from_numpy(late_np).to(dev)
            xr[late] = torch.from_numpy(SW[0][si][post_idx[sel][late_np]]).to(dev)
            yr[late] = torch.from_numpy(SW[1][ti][post_idx[sel][late_np]]).to(dev)
    xr = torch.nn.functional.normalize(xr, dim=-1)
    yr = torch.nn.functional.normalize(yr, dim=-1)
    out[sel] = torch.cat([xr, yr], -1).half().cpu().numpy()
assert off == T * m
U = np.lib.format.open_memmap(os.path.join(a.out, "uploads.npy"), mode="w+",
                              dtype=np.float16, shape=(T, m, 2 * d))
U[:] = out.reshape(T, m, 2 * d)
U.flush()

np.save(os.path.join(a.out, "obs.npy"), obs)
np.save(os.path.join(a.out, "labels.npy"), idx["2_way_label"].values.astype(np.uint8))
np.save(os.path.join(a.out, "sybil_mask.npy"), sybil)
json.dump(dict(n_nodes=a.n_nodes, mode=a.mode, m_h=a.m_h if a.mode == "identity_split" else None),
          open(os.path.join(a.out, "meta.json"), "w"), indent=1)

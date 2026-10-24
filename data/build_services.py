"""Heterogeneous services: align every service to the CLIP ViT-B/32 space (S1) and write the stream.

Alignment: a modality-wise ridge mapping per service, fitted on unlabeled posts 0-3999, with the regularization
chosen by held-out cosine to S1 on unlabeled posts 4000-7999; S1 uses identity mappings; aligned representations
are unit-normalized within each modality. Stream: registration = posts 8000-11999, test = posts 20000-23999, and
every service contributes to every post.
  --n 6   S1-S6, Sybils S2, S4, S6                 -> OUT/up_six
  --n 10  S1-S10, Sybils S2, S4, S6, S8, S10       -> OUT/up_ten
"""
import argparse, json, os, numpy as np, pandas as pd
p = argparse.ArgumentParser()
p.add_argument("--emb", required=True, help="directory of emb_S*.npz (encode_services.py)")
p.add_argument("--n", type=int, choices=[6, 10], required=True); p.add_argument("--out", required=True)
a = p.parse_args()
D = os.environ.get("CURB_DATA", "cache")
SV = [f"S{k}" for k in range(1, a.n + 1)]; SYB = [s for s in ("S2", "S4", "S6", "S8", "S10") if s in SV]
E = {s: np.load(f"{a.emb}/emb_{s}.npz", allow_pickle=True) for s in SV}
post = E["S1"]["post"]; pos = {int(p): i for i, p in enumerate(post)}; sel = lambda lo, hi: np.array([pos[p] for p in range(lo, hi)])
FIT, HOLD, REG, TEST = sel(0, 4000), sel(4000, 8000), sel(8000, 12000), sel(20000, 24000)
nrm = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
mp = {}
for s in SV:
    mp[s] = {}
    for mod in ("img", "txt"):
        X, R = E[s][mod].astype(np.float64), E["S1"][mod].astype(np.float64)
        if s == "S1":
            W = np.eye(X.shape[1])
        else:
            best = None
            for l_ in (0.01, 0.1, 1, 10, 100):
                A = X[FIT]; W_ = np.linalg.solve(A.T @ A + l_ * np.eye(A.shape[1]), A.T @ R[FIT])
                c = float((nrm(X[HOLD] @ W_) * R[HOLD]).sum(1).mean())
                if best is None or c > best[0]: best = (c, l_, W_)
            W = best[2]
        mp[s][mod] = nrm(X @ W).astype(np.float32)
idx = pd.read_parquet(f"{D}/embed_d1/index.parquet"); order = np.argsort(idx["created_utc"].values, kind="mergesort")[:80000]
lab = idx["2_way_label"].values[order].astype(np.uint8)
STREAM = np.r_[REG, TEST]; n, m = len(STREAM), len(SV); U = np.zeros((n, m, 1024), np.float16)
for k, s in enumerate(SV):
    U[:, k, :512] = mp[s]["img"][STREAM]; U[:, k, 512:] = mp[s]["txt"][STREAM]
d = f"{a.out}/up_" + ("six" if a.n == 6 else "ten"); os.makedirs(d, exist_ok=True)
np.save(f"{d}/uploads.npy", U); np.save(f"{d}/obs.npy", np.tile(np.arange(m, dtype=np.int32), (n, 1)))
np.save(f"{d}/labels.npy", lab[post[STREAM]]); np.save(f"{d}/sybil_mask.npy", np.array([s in SYB for s in SV]))
json.dump(dict(n_nodes=m), open(f"{d}/meta.json", "w"))

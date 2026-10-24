"""Run CURB on one stream under one attack and write <out>/posts.npz (and <out>/diag.npz with --diag).

  --method   curb, or curb_no_X[+Y] without score term(s) X, Y in {a, u, q, c}  (curb_no_a+u, curb_no_u, curb_no_q, curb_no_c)
  --attack   clean, direct, content_init, diversified, adaptive
"""
import argparse, json, os, sys
import numpy as np, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from curb import CURB, TERMS
from attacks import REGISTRY
import protocol

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_stream(updir, cfg):
    meta = json.load(open(os.path.join(updir, "meta.json")))
    lab = np.load(os.path.join(updir, "labels.npy"))
    T = len(lab)
    w, v = int(cfg["warm_frac"] * T), int(cfg["val_frac"] * T)
    data = dict(U=np.load(os.path.join(updir, "uploads.npy"), mmap_mode="r"), obs=np.load(os.path.join(updir, "obs.npy")),
                labels=lab, sybil_mask=np.load(os.path.join(updir, "sybil_mask.npy")), n_ids=meta["n_nodes"],
                warm_idx=np.arange(0, w), test_idx=np.arange(w + v, T))
    if meta.get("mode") == "identity_split":                 # Sybil scaling: 10 honest slots + the first s_used Sybils
        k = int(cfg["s_used"])
        data["slots"] = np.arange(meta["m_h"] + k)
    r = int(cfg.get("reg_posts", 0))                         # registration history: the last reg_posts posts only
    if 0 < r < w:
        data["warm_idx"], data["attack_warm_idx"] = np.arange(w - r, w), np.arange(0, w)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uploads", required=True, help="stream directory (data/build_streams.py or data/build_services.py)")
    ap.add_argument("--method", default="curb")
    ap.add_argument("--attack", default="clean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides of config.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--diag", action="store_true", help="also write every score term to diag.npz")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    cfg["diag"] = a.diag
    assert a.method == "curb" or a.method.startswith("curb_no_"), a.method
    terms = TERMS if a.method == "curb" else tuple(t for t in TERMS if t not in a.method[len("curb_no_"):].split("+"))
    cfg.setdefault("evade_terms", ",".join(terms))           # the adaptive attack evades the variant's own terms
    data = load_stream(a.uploads, cfg)
    dim = data["U"].shape[2]
    method = CURB(dim, data["n_ids"], cfg, terms=terms)
    attack = REGISTRY[a.attack](dim, cfg, np.random.RandomState(a.seed + 77))

    posts = protocol.run_stream(data, method, attack, cfg, a.seed)
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "posts.npz"), **posts)
    if a.diag:
        diag = {k: np.stack(v) for k, v in method.log.items()}
        diag["is_sybil"] = posts["syb"]
        np.savez_compressed(os.path.join(a.out, "diag.npz"), **diag)


if __name__ == "__main__":
    main()

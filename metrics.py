"""Paper metrics from two runs of the same method, seed and stream: one attacked, one clean.

  Acc  accuracy on the scored half (latter half of the test stream, attack at full strength)
  FR   flip rate: fraction of scored posts whose prediction differs from the clean run
  RFR  ramp flip rate: the same on the ramp, posts [0.1 n, 0.5 n) where the attack strength is still rising
  SWS  Sybil aggregation-weight share (%), averaged over the ramp
  ms_per_post  CURB aggregation time per post (attacked run)

usage: python metrics.py <attacked_run_dir> <clean_run_dir>
"""
import sys
import numpy as np


def metrics(run, clean):
    x = np.load(run + "/posts.npz"); y = np.load(clean + "/posts.npz")
    n = len(x["y"]); h = n // 2; r0, r1 = int(.1 * n), int(.5 * n)
    fl = x["pred"] != y["pred"]
    W = 100 * (x["w"] * x["syb"]).sum(1)
    return dict(acc=100 * (x["pred"] == x["y"])[h:].mean(), FR=100 * fl[h:].mean(),
                RFR=100 * fl[r0:r1].mean(), SWS=W[r0:r1].mean(), ms_per_post=float(x["agg_ms_per_post"]))


if __name__ == "__main__":
    m = metrics(sys.argv[1], sys.argv[2])
    print("  ".join(f"{k} {v:.2f}" for k, v in m.items()))

"""Online collaboration protocol: registration, then predict-then-update, one post at a time."""
import collections
import copy
import time
import numpy as np
from nn import MLP2, ce_grad


class Ctx:
    """Server-side models: prediction model g, active cross-modal predictor f_act and its EMA reference f_ref."""
    def __init__(self, dim, cfg, seed):
        dm = dim // 2
        self.g = MLP2(dim, cfg["g_hidden"], 2, lr=cfg["lr"], seed=seed)
        self.f_act = MLP2(dm, cfg["f_hidden"], dm, lr=cfg["lr"], seed=seed + 1)
        self.f_ref = MLP2(dm, cfg["f_hidden"], dm, lr=cfg["lr"], seed=seed + 1)
        self.f_ref.copy_params_from(self.f_act)
        self.eta = cfg["eta"]
        self.cm_weight = cfg["cm_weight"]
        self.dm = dm
        self.g_frozen = False

    def update(self, uploads, z_agg, y, w_tilde):
        """One Adam step on CE(g(z_agg), y) and on cm_weight * L_cm with weights w_tilde; then f_ref <- EMA(f_act)."""
        if not self.g_frozen:
            logits, hg = self.g.forward(z_agg)
            self.g.step(self.g.backward(z_agg, hg, ce_grad(logits, y))[0])
        m = len(uploads)
        self.f_act.sq_step_batch(uploads[:, :self.dm], uploads[:, self.dm:], w_tilde, self.cm_weight / m)
        self.f_ref.load_from(self.f_act, self.eta)


def nu_schedule(n_online, start_frac, full_frac, nu0=0.15):
    """Attack strength: 0 before start_frac, then nu0 -> 1 linearly until full_frac, then 1."""
    nu = np.zeros(n_online, np.float32)
    s = int(start_frac * n_online)
    f = max(int(full_frac * n_online), s + 1)
    nu[s:f] = np.linspace(nu0, 1.0, f - s, dtype=np.float32)
    nu[f:] = 1.0
    return nu


def _renormalise(out, idx, dm):
    for sl in (slice(0, dm), slice(dm, 2 * dm)):
        v = out[idx, sl]
        out[idx, sl] = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return out


def _shift_warm(up, smask, frac, dm):
    """Registration poisoning (Shift): move each controlled registration contribution a fraction frac toward the
    post's mean contribution."""
    if frac <= 0 or not smask.any():
        return up
    out = up.copy()
    idx = np.where(smask)[0]
    out[idx] = (1.0 - frac) * out[idx] + frac * up.mean(0)
    return _renormalise(out, idx, dm)


def _shape_warm(up, smask, scale, dm, seed, t):
    """Registration poisoning (Spread): add Gaussian noise of std `scale` to controlled registration contributions,
    deterministic in (seed, t) so that both registration passes see the same contributions."""
    if scale <= 0 or not smask.any():
        return up
    rng = np.random.RandomState((1000003 * (seed + 1) + t) % (2 ** 31 - 1))
    out = up.copy()
    idx = np.where(smask)[0]
    out[idx] += rng.randn(len(idx), up.shape[1]).astype(np.float32) * scale
    return _renormalise(out, idx, dm)


def run_stream(data, method, attack, cfg, seed):
    """Registration, then the online test stream.

    Registration pass 1 trains g and f_act (f_ref by EMA) with one update per post on uniform aggregates; pass 2,
    without model updates, gives CURB its registration statistics. Online, each post is aggregated and predicted
    before its label is used for the update.
    """
    U, obs, lab, sybil = data["U"], data["obs"], data["labels"], data["sybil_mask"]
    slots = data.get("slots")                                  # Sybil scaling: the active slots of each post
    if slots is not None:
        get_u = lambda t: np.asarray(U[t], np.float32)[slots]
        get_o = lambda t: obs[t][slots]
    else:
        get_u = lambda t: np.asarray(U[t], np.float32)
        get_o = lambda t: obs[t]
    warm_idx, online_idx = data["warm_idx"], data["test_idx"]
    attack_warm_idx = data.get("attack_warm_idx")
    dim = U.shape[2]; dm = dim // 2
    ctx = Ctx(dim, cfg, seed)
    warm_shape, warm_shift = float(cfg.get("warm_shape", 0.0)), float(cfg.get("warm_shift", 0.0))
    prep = lambda t, ids: _shift_warm(_shape_warm(get_u(t), sybil[ids], warm_shape, dm, seed, t), sybil[ids], warm_shift, dm)

    # ---- registration, pass 1 ----
    m = len(slots) if slots is not None else U.shape[1]
    w_eq = np.full(m, 1.0 / m, np.float32)
    for t in warm_idx:
        up = prep(t, get_o(t))
        ctx.update(up, w_eq @ up, int(lab[t]), np.ones(m, np.float32))

    # ---- registration, pass 2 ----
    for t in warm_idx:
        ids = get_o(t); up = prep(t, ids)
        method.warm_pass(up, ids, ctx)
        if attack_warm_idx is None:
            attack.observe_warm(up[sybil[ids]], ids[sybil[ids]])
    if attack_warm_idx is not None:                            # shorter server history; the attacker keeps its own
        for t in attack_warm_idx:
            ids = get_o(t)
            attack.observe_warm(prep(t, ids)[sybil[ids]], ids[sybil[ids]])
    method.finalize_warm(ctx)
    for t in warm_idx:
        ids = get_o(t)
        method.warm_pass2(prep(t, ids), ids, ctx)
    method.finalize_warm2(ctx)
    attack.finalize_warm()

    # ---- online ----
    ctx.g_frozen = bool(cfg.get("g_freeze_online", False))    # f_act / f_ref still adapt
    n = len(online_idx)
    nu = nu_schedule(n, cfg["attack_start"], cfg["attack_full"]) if attack.active else np.zeros(n, np.float32)
    if cfg.get("nu_cap") is not None:                          # low-magnitude sustained attack
        nu = np.minimum(nu, float(cfg["nu_cap"])).astype(np.float32)
    atk_g = copy.deepcopy(ctx.g) if cfg.get("atk_knowledge", "current") == "reg" else ctx.g   # reg: g frozen at registration
    label_frac, label_delay = float(cfg.get("label_frac", 1.0)), int(cfg.get("label_delay", 0))
    pending = collections.deque()
    preds, ys = np.zeros(n, np.uint8), np.zeros(n, np.uint8)
    W, SYB = [], []
    agg_time = 0.0
    for i, t in enumerate(online_idx):
        up, ids = get_u(t), get_o(t)
        smask = sybil[ids]
        if attack.active and nu[i] > 0 and smask.any():
            own = up[smask]
            # estimated label: the attacker's prediction on its unmodified contributions (or the ground truth)
            y_atk = int(np.argmax(atk_g(own.mean(0)))) if cfg["atk_label"] == "pred" else int(lab[t])
            up = up.copy()
            up[smask] = attack.craft(own, ids[smask], len(ids), y_atk, atk_g, float(nu[i]))
        t0 = time.perf_counter()
        z_agg, w = method.aggregate(up, ids, ctx)
        agg_time += time.perf_counter() - t0
        preds[i] = int(np.argmax(np.asarray(ctx.g(z_agg), np.float64)))
        y = int(lab[t])
        ys[i] = y
        W.append(np.asarray(w, np.float32)); SYB.append(np.asarray(smask, bool))
        w_tilde = w / (w.max() + 1e-6)
        labelled = label_frac >= 1.0 or ((1103515245 * (t + seed) + 12345) % 1000) < label_frac * 1000
        if labelled:
            if label_delay > 0:
                pending.append((up, z_agg, y, w_tilde))
                if len(pending) > label_delay:
                    ctx.update(*pending.popleft())
            else:
                ctx.update(up, z_agg, y, w_tilde)
        attack.update_trend(z_agg)

    return dict(pred=preds, y=ys, w=np.stack(W).astype(np.float16), syb=np.stack(SYB),
                agg_ms_per_post=np.float64(1000.0 * agg_time / max(n, 1)))

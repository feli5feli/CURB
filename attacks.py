"""Sybil attacks: Direct, Content-Init., Diversified and Adaptive.

The adversary sees its controlled identities' registration and unmodified contributions, the public aggregate trend
(EMA of past aggregates) and the prediction model g. A controlled identity submits Pi((1 - nu) z_trend + nu z_adv),
where Pi projects each modality onto the unit sphere and z_adv comes from projected gradient ascent on CE of g.
"""
import numpy as np
from nn import MLP2, ce_grad, ce_loss


def _unit(v, eps=1e-12):
    return v / (np.linalg.norm(v) + eps)


class Attack:
    name = "clean"
    active = False

    def __init__(self, dim, cfg, rng):
        self.d, self.dm, self.cfg, self.rng = dim, dim // 2, cfg, rng
        self.trend_retention = cfg["trend_retention"]
        self.steps, self.step = cfg["pgd_steps"], cfg["pgd_step"]
        self.z_trend = None

    def _project(self, z):
        out = np.empty_like(z)
        for sl in (slice(0, self.dm), slice(self.dm, 2 * self.dm)):
            out[sl] = z[sl] / (np.linalg.norm(z[sl]) + 1e-12)
        return out

    def observe_warm(self, own, own_ids):
        pass

    def finalize_warm(self):
        pass

    def update_trend(self, z_agg):
        r = self.trend_retention
        self.z_trend = z_agg.copy() if self.z_trend is None else r * self.z_trend + (1 - r) * z_agg

    def _trend(self):
        assert self.z_trend is not None, "the attack starts after the first released aggregate"
        return self.z_trend

    def craft(self, own, own_ids, m, y, g, nu):
        """own: unmodified contributions of the controlled identities own_ids; m: active participants on the post."""
        return own

    def _submit(self, trend, z, off, own, nu):
        out = own.copy()
        for k in range(len(own)):
            out[k] = self._project((1.0 - nu) * trend + nu * (z + off[k]))
        return out

    def _pgd(self, trend, frac, y, g, evade=None, z0=None):
        """PGD on CE(g((1 - frac) trend + frac z)). With `evade`, each step is unit(damage) - lam_P unit(evasion);
        further restarts start around the trend and the best one on the attack objective is kept."""
        n_re = int(self.cfg.get("pgd_restarts", 1))
        lam_P = getattr(self, "lam_P", None)
        best, best_score = None, -np.inf
        for attempt in range(n_re):
            start = z0.copy() if (z0 is not None and attempt == 0) else trend + 0.3 * self.rng.randn(self.d).astype(np.float32)
            z = self._project(start)
            for _ in range(self.steps):
                logits, h = g.forward((1.0 - frac) * trend + frac * z)
                grad = frac * g.input_grad(h, ce_grad(logits, y))
                if evade is not None:
                    grad = _unit(grad) - lam_P * _unit(evade(z))
                z = self._project(z + self.step * _unit(grad))
            logits, _ = g.forward((1.0 - frac) * trend + frac * z)
            score = ce_loss(logits, y)
            if evade is not None:
                score -= lam_P * float(np.linalg.norm(evade(z)))
            if score > best_score:
                best, best_score = z, score
        return best


class Direct(Attack):
    """Direct: optimizes the contribution from the public trend; each controlled identity adds its own jitter."""
    name = "direct"
    active = True

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._hist = []

    def observe_warm(self, own, own_ids):
        if len(own):
            self._hist.append(own)

    def finalize_warm(self):
        H = np.concatenate(self._hist, 0)
        self._hist_sample = H[self.rng.choice(len(H), min(512, len(H)), replace=False)].copy()
        self._hist = None

    def _offsets(self, own_ids):
        return [self.cfg["jitter"] * self.rng.randn(self.d).astype(np.float32) for _ in own_ids]

    def craft(self, own, own_ids, m, y, g, nu):
        trend = self._trend()
        off = self._offsets(own_ids)
        frac = len(own_ids) / m
        return self._submit(trend, self._pgd(trend, frac, y, g), off, own, nu)


class ContentInit(Direct):
    """Content-Init.: starts from a real controlled image representation and its nearest real text representation,
    and optimizes the text side only."""
    name = "content_init"

    def finalize_warm(self):
        super().finalize_warm()
        H = self._hist_sample
        self.pool_x, self.pool_y = H[:, :self.dm].copy(), H[:, self.dm:].copy()
        Xn = self.pool_x / (np.linalg.norm(self.pool_x, axis=1, keepdims=True) + 1e-12)
        Yn = self.pool_y / (np.linalg.norm(self.pool_y, axis=1, keepdims=True) + 1e-12)
        self.match = (Xn @ Yn.T).argmax(1)                     # nearest real text per image

    def craft(self, own, own_ids, m, y, g, nu):
        k = self.rng.randint(len(self.pool_x))
        z = np.concatenate([self.pool_x[k], self.pool_y[self.match[k]]]).astype(np.float32)
        trend = self._trend()
        frac = len(own_ids) / m
        for _ in range(self.steps):
            logits, h = g.forward((1.0 - frac) * trend + frac * z)
            dz = frac * g.input_grad(h, ce_grad(logits, y))
            dz[:self.dm] = 0.0
            z = self._project(z + self.step * _unit(dz))
        return self._submit(trend, z, self._offsets(own_ids), own, nu)


class Diversified(ContentInit):
    """Diversified: Content-Init. with a fixed identity-specific direction (strength beta_het) per controlled
    identity, so that controlled identities remain distinct while pursuing the same objective."""
    name = "diversified"

    def finalize_warm(self):
        super().finalize_warm()
        self.beta_het = self.cfg["beta_het"]
        self._dirs = {}

    def _dir(self, j):
        if j not in self._dirs:
            self._dirs[j] = _unit(self.rng.randn(self.d).astype(np.float32))
        return self._dirs[j]

    def _offsets(self, own_ids):
        return [self.beta_het * self._dir(j) for j in own_ids]


class Adaptive(Diversified):
    """Adaptive: each controlled identity starts from its identity-specific direction and optimizes damage against
    differentiable proxies of CURB's score terms and weights, estimated from the controlled identities' own
    registration contributions with a surrogate image-to-text model:

      a, u  residual magnitude above the identity's registration median (u through rho2)
      q     distance above the identity's registration anchor band
      c     relation of exponentially smoothed residual directions on the attacker's high-variance dimensions,
            centered and calibrated on the controlled identities' registration signatures; each signature is
            pushed away from its most related accomplice
      w     the pull on an identity is scaled by exp(-score), its share of the weights

    evade_terms: score terms of the evaluated variant; adaptive_init: "trend" or "registration" (mean registration
    contribution of the controlled identities) as the start.
    """
    name = "adaptive"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._own = {}

    def observe_warm(self, own, own_ids):
        super().observe_warm(own, own_ids)
        for z, j in zip(own, own_ids):
            rows = self._own.setdefault(int(j), [])
            if len(rows) < 4000:
                rows.append(np.asarray(z, np.float32))

    def finalize_warm(self):
        super().finalize_warm()
        c = self.cfg
        self.lam_P = c["lam_P"]
        self.rho2, self.lam_q, self.lam_c, self.beta_h, self.K = c["rho2"], c["lam_q"], c["lam_c"], c["beta_h"], c["K"]
        self.terms = set(c["evade_terms"].split(","))
        f = MLP2(self.dm, c["f_hidden"], self.dm, lr=1e-3, seed=11)       # surrogate image-to-text model
        H = self._hist_sample
        for _ in range(6):
            for i in self.rng.permutation(len(H)):
                x, y = H[i, :self.dm], H[i, self.dm:]
                pred, hh = f.forward(x)
                f.step(f.backward(x, hh, 2.0 * (pred - y))[0])
        self.f_sur = f
        E = H[:, self.dm:] - f.forward_batch(H[:, :self.dm])
        self.I = np.argsort(-E.var(0))[:self.K]
        self.reg_mean = H.mean(0).astype(np.float32)
        mad = lambda v: float(np.median(np.abs(v - np.median(v))) * 1.4826) + 1e-3
        # signatures of the controlled identities, their background and relation calibration
        S = {}
        for j, rows in self._own.items():
            Hj = np.stack(rows)
            V = (Hj[:, self.dm:] - f.forward_batch(Hj[:, :self.dm]))[:, self.I]
            V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
            s_ = np.zeros(self.K, np.float32)
            for v in V:
                s_ = self.beta_h * s_ + (1 - self.beta_h) * v
            S[j] = s_
        M = np.stack(list(S.values())); self.h_bg = M.mean(0).astype(np.float32)
        C = M - self.h_bg; C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        G = C @ C.T; np.fill_diagonal(G, -np.inf); rm = G.max(1)
        self.mu_c = float(np.median(rm)); self.sg_c = mad(rm)
        self.h = dict(S)
        # identity-relative residual band and anchor distance band
        self.band = {}
        for j, rows in self._own.items():
            Hj = np.stack(rows)
            rj = np.linalg.norm(Hj[:, self.dm:] - f.forward_batch(Hj[:, :self.dm]), axis=1)
            pj = Hj.mean(0).astype(np.float32); dj = np.linalg.norm(Hj - pj, axis=1)
            self.band[j] = (float(np.median(rj)), mad(rj), pj, float(np.median(dj)), mad(dj))
        self._own = None

    def _evade(self, j):
        """Gradient of the attacker's estimate of identity j's unreliability score."""
        mu_r, sg_r, anchor, mu_d, sg_d = self.band[j]
        w_res = (1.0 if "a" in self.terms else 0.0) + (self.rho2 if "u" in self.terms else 0.0)
        w_anc = self.lam_q if "q" in self.terms else 0.0

        def grad(z):
            x, y = z[:self.dm], z[self.dm:]
            pred, hh = self.f_sur.forward(x)
            e = y - pred
            r = np.linalg.norm(e) + 1e-12
            gr = np.zeros_like(z)
            if r > mu_r and w_res > 0:
                gr[:self.dm] += (w_res / sg_r) * self.f_sur.input_grad(hh, -e / r)
                gr[self.dm:] += (w_res / sg_r) * (e / r)
            dvec = z - anchor
            dn = np.linalg.norm(dvec) + 1e-12
            if dn > mu_d and w_anc > 0:
                gr += w_anc * (dvec / dn) / sg_d
            score = w_res * max((r - mu_r) / sg_r, 0.0) + w_anc * max((dn - mu_d) / sg_d, 0.0)
            gr *= float(np.exp(-min(score, 40.0)))
            others = [v for k, v in self.h.items() if k != j and k in self._active]
            if "c" in self.terms and others:
                O = np.stack(others) - self.h_bg
                O = O / (np.linalg.norm(O, axis=1, keepdims=True) + 1e-12)
                eI = e[self.I]; nI = np.linalg.norm(eI) + 1e-12; v = eI / nI
                c_ = self.beta_h * self.h.get(j, np.zeros(self.K, np.float32)) + (1 - self.beta_h) * v - self.h_bg
                nc = np.linalg.norm(c_) + 1e-12
                cos = O @ (c_ / nc); k = int(np.argmax(cos))
                if (cos[k] - self.mu_c) / self.sg_c > 0:
                    dc = (O[k] - cos[k] * c_ / nc) / nc
                    dv = (1 - self.beta_h) * dc
                    de = np.zeros(self.dm, np.float32)
                    de[self.I] = (dv - v * float(v @ dv)) / nI
                    wc = self.lam_c / self.sg_c
                    gr[:self.dm] += wc * self.f_sur.input_grad(hh, -de)
                    gr[self.dm:] += wc * de
            return gr
        return grad

    def craft(self, own, own_ids, m, y, g, nu):
        trend = self._trend()
        frac = len(own_ids) / m
        out = own.copy()
        self._active = set(int(j) for j in own_ids)
        from_reg = self.cfg.get("adaptive_init", "trend") == "registration"
        for i, j in enumerate(own_ids):
            j = int(j)
            base = self.reg_mean if from_reg else trend
            z0 = self._project(base + self.beta_het * self._dir(j))
            z = self._pgd(trend, frac, y, g, self._evade(j), z0=z0)
            out[i] = self._project((1.0 - nu) * trend + nu * z)
            e = z[self.dm:] - self.f_sur.forward_batch(z[None, :self.dm])[0]
            v = e[self.I] / (np.linalg.norm(e[self.I]) + 1e-12)
            prev = self.h.get(j)
            self.h[j] = ((1 - self.beta_h) * v if prev is None else self.beta_h * prev + (1 - self.beta_h) * v).astype(np.float32)
        return out


REGISTRY = {c.name: c for c in (Attack, Direct, ContentInit, Diversified, Adaptive)}

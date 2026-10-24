"""CURB: robust aggregation from cross-modal behavioral evidence.

For every post, each active identity j contributes z_j = [v_j ; t_j]. With the cross-modal residual
e_j = t_j - f_ref(v_j), CURB computes the unreliability score and the aggregation weight

    E_j = a_j + u_j + lam_q * q_j + lam_c * cbar_j,        w_j = exp(-E_j) / sum_k exp(-E_k),

  a_j     current deviation of the residual magnitude from the identity's registration median / MAD
  u_j     accumulated deviation, u_j <- rho1 * u_j + rho2 * a_j
  q_j     deviation of z_j from the identity's registration anchor p_j
  cbar_j  calibrated cross-identity relation of the behavioral signatures (EMA of the unit residual against the
          frozen registration reference f_reg on the dimensions I), centered on the registration background
"""
import copy
import numpy as np

TERMS = ("a", "u", "q", "c")


def _mad(x, axis=0):
    med = np.median(x, axis=axis)
    return np.median(np.abs(x - med), axis=axis) * 1.4826


def softmin_weights(E):
    """w_j = exp(-E_j) / sum_k exp(-E_k), shifted by min(E) so that the largest term is exp(0) = 1."""
    w = np.exp(-(E - E.min()))
    return w / w.sum()


class CURB:
    def __init__(self, dim, n_ids, cfg, terms=TERMS):
        self.d, self.N, self.dm = dim, n_ids, dim // 2
        self.kappa = float(cfg["kappa"])           # tolerance threshold (MADs)
        self.gamma = cfg["gamma"]                  # penalty of unusually small residual magnitudes
        self.rho1, self.rho2 = cfg["rho1"], cfg["rho2"]
        self.lam_q, self.lam_c = cfg["lam_q"], cfg["lam_c"]
        self.beta_h = cfg["beta_h"]                # EMA retention of the behavioral signature
        self.K = cfg["K"]                          # |I|
        self.tau_delta = cfg["tau_delta"]          # staleness discount time constant (posts)
        self.eps = 1e-6
        self.A_max = 50.0                          # cap of a and q
        # off only in the calibration ablation
        self.center = bool(cfg.get("center", True))        # subtract the registration background h_bg
        self.calibrate = bool(cfg.get("calibrate", True))  # z-score against (mu_c, sigma_c); off = raw cosine
        self.diag = bool(cfg.get("diag", False))           # log the score terms for the complementarity analysis
        self.terms = set(terms)
        self.f_reg = None
        self.h_bg = None
        self._warm = {j: [] for j in range(n_ids)}
        self._warm_res = []
        self._n_warm = 0

    # ------------------------------------------------------------------ registration
    def _rnorm(self, e):
        return np.linalg.norm(e / self.s, axis=-1) / np.sqrt(self.dm)

    def warm_pass(self, uploads, ids, ctx):
        """Registration, second pass (no model updates), first sweep: residuals and contributions."""
        self._n_warm += 1
        x, y = uploads[:, :self.dm], uploads[:, self.dm:]
        e = (y - ctx.f_ref.forward_batch(x)).astype(np.float32)
        for i, j in enumerate(ids):
            self._warm[j].append((e[i], uploads[i]))
        self._warm_res.append(e)

    def finalize_warm(self, ctx):
        """Residual scale s, dimensions I, and the identity-relative bands and anchors."""
        self.f_reg = copy.deepcopy(ctx.f_ref)                  # frozen registration reference
        N = self.N
        E = np.concatenate(self._warm_res, 0)
        self.s = (E.std(0) + 1e-6).astype(np.float32)          # per-dimension residual scale
        self.I = np.argsort(-E.var(0))[:self.K].copy()         # shared top-variance residual dimensions
        self.mu_r = np.zeros(N, np.float32); self.sig_r = np.zeros(N, np.float32)
        self.mu_d = np.zeros(N, np.float32); self.sig_d = np.zeros(N, np.float32)
        self.p = np.zeros((N, self.d), np.float32)
        self.registered = np.array([len(self._warm[j]) > 0 for j in range(N)])   # identities present in this stream
        for j in np.where(self.registered)[0]:
            rec = self._warm[j]
            assert len(rec) >= 5, f"identity {j} has {len(rec)} registration contributions"
            r = self._rnorm(np.stack([t[0] for t in rec]))
            Z = np.stack([t[1] for t in rec])
            p = Z.mean(0)
            dist = np.linalg.norm(Z - p, axis=1)
            self.mu_r[j], self.sig_r[j] = np.median(r), _mad(r)
            self.p[j] = p
            self.mu_d[j], self.sig_d[j] = np.median(dist), _mad(dist)

        self.u = np.zeros(N, np.float32)
        self.h = np.zeros((N, self.K), np.float32)             # behavioral signatures
        self.last = np.full(N, -10 ** 9, np.int64)             # last post each identity contributed to
        self.t = 0
        self._warm = self._warm_res = None
        self._calib = []
        self._half = self._n_warm // 2
        self._calib_n = 0
        self.log = {k: [] for k in ("a", "u", "q", "c", "cbar", "w", "w_no_cbar")}

    def warm_pass2(self, uploads, ids, ctx):
        """Second sweep: signatures and h_bg on the first half of registration, relation calibration on the second."""
        self.t += 1
        ids = np.asarray(ids)
        x, y = uploads[:, :self.dm], uploads[:, self.dm:]
        e = y - ctx.f_ref.forward_batch(x)                     # f_ref == f_reg during this sweep
        if self.h_bg is None:
            self._update_signature(e, ids)
            self.last[ids] = self.t
            self._calib_n += 1
            if self._calib_n >= self._half:
                self.h_bg = self.h[self.registered].mean(0).astype(np.float32)
            return
        self._calib.append(self._relation(e, ids))

    def finalize_warm2(self, ctx):
        cc = np.concatenate(self._calib)
        self.mu_c = float(np.median(cc))
        self.sig_c = float(_mad(cc))
        self._calib = None
        self.last[:] = self.t                                  # staleness is measured from the end of registration

    # ------------------------------------------------------------------ cross-identity relation
    def _update_signature(self, e, ids):
        v = e[:, self.I]
        self.h[ids] = self.beta_h * self.h[ids] + (1 - self.beta_h) * v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    def _relation(self, e, ids):
        """c_j: strongest background-centered, staleness-discounted relation with another active identity."""
        stale = (self.t - self.last[ids]).astype(np.float32)   # t - t_last (previous participation)
        self.last[ids] = self.t
        self._update_signature(e, ids)
        H = self.h[ids]
        if self.center and self.h_bg is not None:
            H = H - self.h_bg
        Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
        C = Hn @ Hn.T
        delta = np.exp(-stale / self.tau_delta)
        C *= np.outer(delta, delta)
        np.fill_diagonal(C, -np.inf)
        c = C.max(1)
        return c if self.calibrate else np.maximum(c, 0.0)

    # ------------------------------------------------------------------ online
    def aggregate(self, uploads, ids, ctx):
        """Return (aggregate, weights) from the contributions and identities alone (no labels, no Sybil mask)."""
        self.t += 1
        ids = np.asarray(ids)
        m = len(ids)
        x, y = uploads[:, :self.dm], uploads[:, self.dm:]
        e = y - ctx.f_ref.forward_batch(x)
        r = self._rnorm(e)

        zr = (r - self.mu_r[ids]) / (self.sig_r[ids] + self.eps)
        a = np.clip(np.maximum(zr - self.kappa, 0.0) + self.gamma * np.maximum(-zr - self.kappa, 0.0), 0.0, self.A_max)
        self.u[ids] = self.rho1 * self.u[ids] + self.rho2 * a
        u = self.u[ids]
        dist = np.linalg.norm(uploads - self.p[ids], axis=1)
        q = np.clip(np.maximum((dist - self.mu_d[ids]) / (self.sig_d[ids] + self.eps) - self.kappa, 0.0), 0.0, self.A_max)

        e_reg = y - self.f_reg.forward_batch(x)
        c = self._relation(e_reg, ids)
        cbar = np.maximum((c - self.mu_c) / (self.sig_c + self.eps) - self.kappa, 0.0) if self.calibrate else c

        E = np.zeros(m, np.float32)
        if "a" in self.terms: E += a
        if "u" in self.terms: E += u
        if "q" in self.terms: E += self.lam_q * q
        if "c" in self.terms: E += self.lam_c * cbar

        w = softmin_weights(E)
        if self.diag:
            L = self.log
            L["a"].append(a); L["u"].append(u.copy()); L["q"].append(q); L["c"].append(np.array(c, np.float32))
            L["cbar"].append(cbar); L["w"].append(w)
            L["w_no_cbar"].append(np.asarray(softmin_weights(E - (self.lam_c * cbar if "c" in self.terms else 0.0)), np.float32))
        return w @ uploads, w

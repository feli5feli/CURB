"""Two-layer MLPs (Linear - ReLU - Linear) with explicit gradients and Adam."""
import numpy as np


class MLP2:
    def __init__(self, d_in, h, d_out, lr=5e-4, seed=0, beta1=0.9, beta2=0.999, eps=1e-8):
        rng = np.random.RandomState(seed)
        self.W1 = (rng.randn(h, d_in) * np.sqrt(2.0 / d_in)).astype(np.float32)      # He initialization
        self.b1 = np.zeros(h, np.float32)
        self.W2 = (rng.randn(d_out, h) * np.sqrt(1.0 / h)).astype(np.float32)
        self.b2 = np.zeros(d_out, np.float32)
        self.lr, self.b1m, self.b2m, self.eps = lr, beta1, beta2, eps
        self._p = [self.W1, self.b1, self.W2, self.b2]
        self._m = [np.zeros_like(p) for p in self._p]
        self._v = [np.zeros_like(p) for p in self._p]
        self._t = 0

    def forward(self, z):
        a = self.W1 @ z + self.b1
        h = np.maximum(a, 0.0)
        return self.W2 @ h + self.b2, h

    def __call__(self, z):
        return self.forward(z)[0]

    def forward_batch(self, Z):
        H = np.maximum(Z @ self.W1.T + self.b1, 0.0)
        return H @ self.W2.T + self.b2

    def backward(self, z, h, dout):
        """dL/dout -> (parameter gradients, dL/dz)."""
        gW2 = np.outer(dout, h)
        gb2 = dout
        dh = self.W2.T @ dout
        da = dh * (h > 0)
        gW1 = np.outer(da, z)
        gb1 = da
        dz = self.W1.T @ da
        return [gW1, gb1, gW2, gb2], dz

    def input_grad(self, h, dout):
        """dL/dz only."""
        da = (self.W2.T @ dout) * (h > 0)
        return self.W1.T @ da

    def sq_step_batch(self, X, Y, w, scale):
        """One Adam step on scale * sum_i w_i ||f(x_i) - y_i||^2."""
        A = X @ self.W1.T + self.b1
        H = np.maximum(A, 0.0)
        D = (H @ self.W2.T + self.b2) - Y
        D *= (2.0 * scale * w)[:, None]
        gW2 = D.T @ H
        gb2 = D.sum(0)
        dA = (D @ self.W2) * (A > 0)
        gW1 = dA.T @ X
        gb1 = dA.sum(0)
        self.step([gW1, gb1, gW2, gb2])

    def step(self, grads):
        self._t += 1
        bc1 = 1.0 - self.b1m ** self._t
        bc2 = 1.0 - self.b2m ** self._t
        for p, g, m, v in zip(self._p, grads, self._m, self._v):
            m *= self.b1m; m += (1 - self.b1m) * g
            v *= self.b2m; v += (1 - self.b2m) * (g * g)
            p -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)

    def load_from(self, other, tau):
        """EMA: self <- tau * self + (1 - tau) * other."""
        for p, q in zip(self._p, other._p):
            p *= tau
            p += (1 - tau) * q

    def copy_params_from(self, other):
        for p, q in zip(self._p, other._p):
            np.copyto(p, q)


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def ce_grad(logits, y):
    """d CE / d logits for integer label y."""
    p = softmax(logits)
    p[y] -= 1.0
    return p


def ce_loss(logits, y):
    return float(np.log(np.exp(logits - logits.max()).sum()) - (logits[y] - logits.max()))

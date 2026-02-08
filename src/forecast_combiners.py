"""
forecast_combiners.py

A lightweight, dependency-minimal module implementing:

1) Regular (static) model averaging (equal weights)
2) Online Gradient Descent (OGD) combiner (projected onto simplex)
3) Multiplicative Weights Update Method (MWUM) combiner
4) KL-penalised variants that shrink toward equal weights:
   - KL-penalised OGD (extra gradient term from KL(w || uniform))
   - KL-penalised MWUM that uses a *fixed uniform prior* (a "softmax-to-uniform" / entropy-to-uniform update)
5) A simple RL contextual bandit using a Dirichlet policy over weights (REINFORCE) with a simple reward/loss.

Design notes:
- All combiners operate on forecast vectors yhat (shape: [N]) and update weights online.
- Simplex constraints are enforced for OGD via Euclidean projection.
- MWUM maintains strictly positive weights; initialise with interior weights to avoid zero-weight absorbing states.
- "KL-penalised MWUM toward equal weights" is implemented as the closed-form solution to:
      argmin_{w in simplex} <w, loss_vec> + lambda * KL(w || uniform)
  which yields:
      w_i ∝ exp(-loss_i / lambda)
  (uniform prior cancels in normalization).
- The RL bandit is intentionally simple: Dirichlet policy with REINFORCE and an optional weight-change penalty.

Dependencies: numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, Tuple

import numpy as np


# -----------------------------
# Utilities
# -----------------------------

def _safe_normalize(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    s = float(np.sum(w))
    if s <= eps:
        # fallback to uniform
        return np.ones_like(w) / len(w)
    return w / s


def project_to_simplex(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    """
    Euclidean projection of v onto the simplex {w >= 0, sum w = z}.
    Standard algorithm (sorting + threshold).
    """
    v = np.asarray(v, dtype=float)
    n = v.size
    if n == 0:
        return v

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - z))[0]
    if rho.size == 0:
        # If all entries are too small, return uniform
        return np.ones(n) * (z / n)
    rho = rho[-1]
    theta = (cssv[rho] - z) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    # numerical fix
    return _safe_normalize(w) * z


def squared_error(y: float, yhat: float) -> float:
    e = y - yhat
    return float(e * e)


def linex_loss(y: float, yhat: float, a: float) -> float:
    """
    LINEX loss L_a(e)=exp(ae) - ae - 1.
    a>0 penalizes under-prediction more (e = y - yhat > 0),
    a<0 penalizes over-prediction more.
    """
    e = float(y - yhat)
    return float(np.exp(a * e) - a * e - 1.0)


def kl_to_uniform(w: np.ndarray, eps: float = 1e-12) -> float:
    """
    KL(w || uniform) = sum_i w_i log(w_i / (1/N)) = sum_i w_i log w_i + log N.
    """
    w = np.asarray(w, dtype=float)
    n = w.size
    w = np.clip(w, eps, 1.0)
    return float(np.sum(w * np.log(w)) + np.log(n))


# -----------------------------
# Base interface
# -----------------------------

class BaseCombiner:
    """
    Common interface:
      - weights: current weights (numpy array of shape [N])
      - combine(yhat): returns combined forecast (scalar)
      - update(y, yhat, **kwargs): updates weights given realized outcome y and forecast vector yhat
    """

    def __init__(self, n_models: int):
        self.n_models = int(n_models)
        self.weights = np.ones(self.n_models, dtype=float) / self.n_models

    def reset(self, n_models: Optional[int] = None):
        if n_models is not None:
            self.n_models = int(n_models)
        self.weights = np.ones(self.n_models, dtype=float) / self.n_models

    def combine(self, yhat: np.ndarray) -> float:
        yhat = np.asarray(yhat, dtype=float).reshape(-1)
        assert yhat.size == self.weights.size, "yhat size must match number of weights/models"
        return float(self.weights @ yhat)

    def update(self, y: float, yhat: np.ndarray, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


# -----------------------------
# 1) Regular model averaging
# -----------------------------

class EqualWeightAverager(BaseCombiner):
    """
    Always equal weights. No learning.
    """

    def update(self, y: float, yhat: np.ndarray, **kwargs) -> Dict[str, Any]:
        # weights remain uniform
        self.weights = np.ones(self.n_models, dtype=float) / self.n_models
        y_c = self.combine(yhat)
        return {"y_combined": y_c, "loss": squared_error(y, y_c)}


# -----------------------------
# 2) OGD (projected)
# -----------------------------

class OGDCombiner(BaseCombiner):
    """
    Projected Online Gradient Descent on the simplex.

    Loss: L(y - w'yhat). Default uses squared error; can pass custom loss_fn.

    Gradient for squared error:
        l = (y - w'x)^2
        grad = -2 (y - w'x) x
    """

    def __init__(
        self,
        n_models: int,
        eta: float = 0.1,
        eta_schedule: Optional[Callable[[Optional[np.ndarray]], float]] = None,
        loss_fn: Optional[Callable[[float, float], float]] = None,
    ):
        super().__init__(n_models)
        self.eta = float(eta)
        self.eta_schedule = eta_schedule
        self.loss_fn = loss_fn if loss_fn is not None else squared_error

    def _eta(self, state: Optional[np.ndarray] = None) -> float:
        if self.eta_schedule is None:
            return self.eta
        return float(self.eta_schedule(state))

    def update(self, y: float, yhat: np.ndarray, state: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        x = np.asarray(yhat, dtype=float).reshape(-1)
        assert x.size == self.weights.size

        y_c = float(self.weights @ x)
        e = float(y - y_c)

        # gradient for squared error by default; for other losses, user can override by passing grad_fn via kwargs.
        grad_fn = kwargs.get("grad_fn", None)
        if grad_fn is None:
            grad = -2.0 * e * x
        else:
            grad = np.asarray(grad_fn(y, y_c, x, self.weights), dtype=float)

        eta_t = self._eta(state)
        w_new = self.weights - eta_t * grad
        w_new = project_to_simplex(w_new)

        self.weights = w_new
        y_new = float(self.weights @ x)
        loss = float(self.loss_fn(y, y_new))
        return {"y_combined": y_new, "loss": loss, "eta": eta_t, "e": e}


# -----------------------------
# 3) Vanilla MWUM (multiplicative weights)
# -----------------------------

class MWUMCombiner(BaseCombiner):
    """
    Vanilla MWUM update:
        w_{i,t+1} ∝ w_{i,t} * exp(-eta * loss_i,t)

    loss_i,t is per-expert loss (e.g. squared error of expert i).
    """

    def __init__(
        self,
        n_models: int,
        eta: float = 0.5,
        eta_schedule: Optional[Callable[[Optional[np.ndarray]], float]] = None,
        per_expert_loss_fn: Optional[Callable[[float, float], float]] = None,
        floor: float = 1e-12,
    ):
        super().__init__(n_models)
        self.eta = float(eta)
        self.eta_schedule = eta_schedule
        self.per_expert_loss_fn = per_expert_loss_fn if per_expert_loss_fn is not None else squared_error
        self.floor = float(floor)
        # keep strictly interior weights to avoid zeros
        self.weights = _safe_normalize(self.weights + self.floor)

    def _eta(self, state: Optional[np.ndarray] = None) -> float:
        if self.eta_schedule is None:
            return self.eta
        return float(self.eta_schedule(state))

    def update(self, y: float, yhat: np.ndarray, state: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        x = np.asarray(yhat, dtype=float).reshape(-1)
        assert x.size == self.weights.size

        losses = np.array([self.per_expert_loss_fn(y, float(x[i])) for i in range(x.size)], dtype=float)
        eta_t = self._eta(state)

        w = np.clip(self.weights, self.floor, 1.0)
        w_new = w * np.exp(-eta_t * losses)
        w_new = np.clip(w_new, self.floor, None)
        w_new = _safe_normalize(w_new)

        self.weights = w_new
        y_c = float(self.weights @ x)
        return {"y_combined": y_c, "loss": float(squared_error(y, y_c)), "eta": eta_t, "losses": losses}


# -----------------------------
# 4a) KL-penalised OGD toward equal weights
# -----------------------------

class KLPenalisedOGDCombiner(OGDCombiner):
    """
    OGD with an explicit KL(w || uniform) penalty in the instantaneous objective:

        minimize   l_t(w) + lambda_t * KL(w || uniform)
        gradient   grad = grad_loss + lambda_t * (log w + 1)

    We still project back to the simplex after the gradient step.

    Notes:
    - Requires interior weights due to log(w).
    - Use floor to keep weights > 0.
    """

    def __init__(
        self,
        n_models: int,
        eta: float = 0.05,
        lambda_: float = 0.1,
        eta_schedule: Optional[Callable[[Optional[np.ndarray]], float]] = None,
        lambda_schedule: Optional[Callable[[Optional[np.ndarray]], float]] = None,
        loss_fn: Optional[Callable[[float, float], float]] = None,
        floor: float = 1e-12,
    ):
        super().__init__(n_models, eta=eta, eta_schedule=eta_schedule, loss_fn=loss_fn)
        self.lambda_ = float(lambda_)
        self.lambda_schedule = lambda_schedule
        self.floor = float(floor)
        self.weights = _safe_normalize(self.weights + self.floor)

    def _lambda(self, state: Optional[np.ndarray] = None) -> float:
        if self.lambda_schedule is None:
            return self.lambda_
        return float(self.lambda_schedule(state))

    def update(self, y: float, yhat: np.ndarray, state: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        x = np.asarray(yhat, dtype=float).reshape(-1)
        assert x.size == self.weights.size

        y_c = float(self.weights @ x)
        e = float(y - y_c)

        grad_fn = kwargs.get("grad_fn", None)
        if grad_fn is None:
            grad_loss = -2.0 * e * x
        else:
            grad_loss = np.asarray(grad_fn(y, y_c, x, self.weights), dtype=float)

        lam_t = self._lambda(state)
        w_clipped = np.clip(self.weights, self.floor, 1.0)
        grad_kl = np.log(w_clipped) + 1.0  # derivative of sum w log w (constant log N drops out)

        eta_t = self._eta(state)
        w_new = self.weights - eta_t * (grad_loss + lam_t * grad_kl)
        w_new = np.clip(w_new, self.floor, None)
        w_new = project_to_simplex(w_new)

        self.weights = np.clip(w_new, self.floor, None)
        self.weights = _safe_normalize(self.weights)

        y_new = float(self.weights @ x)
        loss = float(self.loss_fn(y, y_new))
        return {
            "y_combined": y_new,
            "loss": loss,
            "eta": eta_t,
            "lambda": lam_t,
            "e": e,
            "kl_to_uniform": kl_to_uniform(self.weights),
        }


# -----------------------------
# 4b) "MWUM" that penalises movements from equal weights (KL-to-uniform)
# -----------------------------

class KLPenalisedMWUMToUniform(BaseCombiner):
    """
    Closed-form entropy/KL-to-uniform regularised weights:

        w_{t+1} = argmin_{w in simplex} <w, losses_t> + lambda_t * KL(w || uniform)
               => w_i ∝ exp( - losses_i / lambda_t )

    This is *not* vanilla MWUM (which uses KL(w||w_t)).
    It directly shrinks toward uniform as lambda_t increases.

    - lambda_t acts like a temperature:
        lambda -> infinity => w -> uniform
        lambda -> 0        => concentrate on best (min-loss) expert
    """

    def __init__(
        self,
        n_models: int,
        lambda_: float = 0.5,
        lambda_schedule: Optional[Callable[[Optional[np.ndarray]], float]] = None,
        per_expert_loss_fn: Optional[Callable[[float, float], float]] = None,
        min_lambda: float = 1e-6,
    ):
        super().__init__(n_models)
        self.lambda_ = float(lambda_)
        self.lambda_schedule = lambda_schedule
        self.per_expert_loss_fn = per_expert_loss_fn if per_expert_loss_fn is not None else squared_error
        self.min_lambda = float(min_lambda)

    def _lambda(self, state: Optional[np.ndarray] = None) -> float:
        if self.lambda_schedule is None:
            return self.lambda_
        return float(self.lambda_schedule(state))

    def update(self, y: float, yhat: np.ndarray, state: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        x = np.asarray(yhat, dtype=float).reshape(-1)
        assert x.size == self.weights.size

        losses = np.array([self.per_expert_loss_fn(y, float(x[i])) for i in range(x.size)], dtype=float)
        lam_t = max(self._lambda(state), self.min_lambda)

        logits = -losses / lam_t
        logits = logits - np.max(logits)  # stability
        w_new = np.exp(logits)
        w_new = _safe_normalize(w_new)

        self.weights = w_new
        y_c = float(self.weights @ x)
        return {
            "y_combined": y_c,
            "loss": float(squared_error(y, y_c)),
            "lambda": lam_t,
            "losses": losses,
            "kl_to_uniform": kl_to_uniform(self.weights),
        }


# -----------------------------
# 5) Simple RL contextual bandit (Dirichlet policy)
# -----------------------------

@dataclass
class BanditConfig:
    state_dim: int
    n_models: int
    lr: float = 0.01
    alpha_min: float = 0.1          # ensures interior Dirichlet
    entropy_bonus: float = 0.0      # optional: encourage exploration/diversification
    weight_change_penalty: float = 0.0  # penalty on ||w_t - w_{t-1}||^2
    baseline_momentum: float = 0.9  # running reward baseline
    seed: Optional[int] = None


class DirichletPolicyBandit:
    """
    A minimal contextual bandit with a Dirichlet policy over weights.

    Policy:
      alpha(s) = softplus(W s + b) + alpha_min
      w ~ Dirichlet(alpha(s))

    Reward:
      r_t = - loss(y, w'yhat) - c * ||w - w_prev||^2 + entropy_bonus * H(w)
    (Entropy bonus encourages diversification/exploration; set to 0 if undesired.)

    Learning:
      REINFORCE with a running baseline:
        grad_theta += (r - b) * grad_theta log pi(w|alpha(s))

    This is intentionally simple (good as a robustness-check scaffold).
    """

    def __init__(
        self,
        config: BanditConfig,
        loss_fn: Callable[[float, float], float] = squared_error,
    ):
        self.cfg = config
        self.loss_fn = loss_fn
        self.rng = np.random.default_rng(config.seed)

        self.W = 0.01 * self.rng.standard_normal((config.n_models, config.state_dim))
        self.b = np.zeros(config.n_models, dtype=float)

        self.w_prev = np.ones(config.n_models, dtype=float) / config.n_models
        self.baseline = 0.0

    @staticmethod
    def _softplus(x: np.ndarray) -> np.ndarray:
        # stable softplus
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

    def alpha(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float).reshape(-1)
        assert s.size == self.cfg.state_dim
        z = self.W @ s + self.b
        a = self._softplus(z) + self.cfg.alpha_min
        return a

    def sample_weights(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a = self.alpha(s)
        w = self.rng.dirichlet(a)
        return w, a

    def _dirichlet_logpdf_grad_alpha(self, w: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        Gradient of log Dirichlet(w | alpha) w.r.t alpha:
          d/d alpha_i [ log Gamma(sum alpha) - sum log Gamma(alpha_i) + sum (alpha_i - 1) log w_i ]
        = psi(sum alpha) - psi(alpha_i) + log w_i
        """
        from math import lgamma  # only for completeness; we don't need it here

        # Use scipy would be nicer, but we keep numpy-only. Implement digamma approx.
        # Digamma approximation (good enough for robustness-check prototype).
        def digamma(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            # approximation using asymptotic expansion for x>0
            # shift up small values
            y = x.copy()
            out = np.zeros_like(y)
            mask = y < 6.0
            while np.any(mask):
                out[mask] -= 1.0 / y[mask]
                y[mask] += 1.0
                mask = y < 6.0
            r = 1.0 / y
            out += np.log(y) - 0.5 * r - (1.0 / 12.0) * r**2 + (1.0 / 120.0) * r**4 - (1.0 / 252.0) * r**6
            return out

        w = np.asarray(w, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        w = np.clip(w, 1e-12, 1.0)
        psi_sum = float(digamma(np.array([np.sum(alpha)]))[0])
        psi_alpha = digamma(alpha)
        return (psi_sum - psi_alpha + np.log(w))

    def step(self, s: np.ndarray, yhat: np.ndarray, y: float) -> Dict[str, Any]:
        """
        One bandit interaction:
          observe s, choose w ~ pi_theta(.|s), get reward from yhat and realized y,
          update theta by REINFORCE.

        Returns diagnostics.
        """
        s = np.asarray(s, dtype=float).reshape(-1)
        x = np.asarray(yhat, dtype=float).reshape(-1)
        assert s.size == self.cfg.state_dim
        assert x.size == self.cfg.n_models

        w, alpha = self.sample_weights(s)
        y_c = float(w @ x)

        loss = float(self.loss_fn(y, y_c))
        turnover_pen = float(np.sum((w - self.w_prev) ** 2))
        # entropy (Shannon) of weights
        w_clip = np.clip(w, 1e-12, 1.0)
        H = float(-np.sum(w_clip * np.log(w_clip)))

        reward = -loss - self.cfg.weight_change_penalty * turnover_pen + self.cfg.entropy_bonus * H

        # baseline update (running)
        self.baseline = self.cfg.baseline_momentum * self.baseline + (1.0 - self.cfg.baseline_momentum) * reward
        adv = reward - self.baseline

        # Grad log pi wrt alpha
        grad_logp_alpha = self._dirichlet_logpdf_grad_alpha(w, alpha)  # shape [N]

        # Chain rule: alpha = softplus(z)+alpha_min, z = W s + b
        z = self.W @ s + self.b
        softplus_grad = 1.0 / (1.0 + np.exp(-z))  # sigmoid
        grad_logp_z = grad_logp_alpha * softplus_grad  # elementwise

        # Parameter gradients
        grad_W = np.outer(grad_logp_z, s)  # [N, d]
        grad_b = grad_logp_z              # [N]

        # REINFORCE ascent on expected reward: theta += lr * adv * grad log pi
        self.W += self.cfg.lr * adv * grad_W
        self.b += self.cfg.lr * adv * grad_b

        self.w_prev = w

        return {
            "weights": w,
            "alpha": alpha,
            "y_combined": y_c,
            "loss": loss,
            "reward": reward,
            "advantage": adv,
            "turnover": turnover_pen,
            "entropy": H,
            "baseline": self.baseline,
        }


# -----------------------------
# Example usage (optional)
# -----------------------------

if __name__ == "__main__":
    # Example: 3 forecasters, dummy data stream
    N = 3
    T = 10

    # Fake forecasts and outcomes
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=T)
    yhat_stream = rng.normal(size=(T, N)) + 0.2 * y_true[:, None]

    # State: 2-dim context (e.g., volatility proxy, disagreement proxy)
    s_stream = rng.normal(size=(T, 2))

    # 1) Equal weights
    eq = EqualWeightAverager(N)

    # 2) OGD
    ogd = OGDCombiner(N, eta=0.2)

    # 3) Vanilla MWUM
    mw = MWUMCombiner(N, eta=0.8)

    # 4a) KL-penalised OGD
    klogd = KLPenalisedOGDCombiner(N, eta=0.05, lambda_=0.1)

    # 4b) KL-to-uniform "MWUM"
    klmw = KLPenalisedMWUMToUniform(N, lambda_=0.5)

    # 5) Bandit
    bandit = DirichletPolicyBandit(
        BanditConfig(state_dim=2, n_models=N, lr=0.02, weight_change_penalty=0.1, entropy_bonus=0.0, seed=1)
    )

    for t in range(T):
        y = float(y_true[t])
        x = yhat_stream[t]
        s = s_stream[t]

        print(f"\n--- t={t} ---")
        print("EQ  ", eq.update(y, x)["y_combined"], eq.weights)
        print("OGD ", ogd.update(y, x, state=s)["y_combined"], ogd.weights)
        print("MW  ", mw.update(y, x, state=s)["y_combined"], mw.weights)
        print("KLOGD", klogd.update(y, x, state=s)["y_combined"], klogd.weights)
        print("KLMW ", klmw.update(y, x, state=s)["y_combined"], klmw.weights)

        b = bandit.step(s, x, y)
        print("BANDIT", b["y_combined"], b["weights"], "reward=", b["reward"])

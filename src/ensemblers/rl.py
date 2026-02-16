from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np


LossType = Literal["mse", "linex"]


def _linex(e: np.ndarray, a: float) -> np.ndarray:
    return np.exp(a * e) - a * e - 1.0


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    ez = np.exp(z)
    return ez / np.sum(ez)


def _hhi(w: np.ndarray) -> float:
    return float(np.sum(w**2))


def _kl_to_uniform(w: np.ndarray, eps: float = 1e-12) -> float:
    w = np.clip(w, eps, 1.0)
    n = w.size
    return float(np.sum(w * np.log(w)) + np.log(n))


def _loss_scalar(y: float, yhat: float, loss: LossType, linex_a: float) -> float:
    e = y - yhat
    if loss == "mse":
        return float(e * e)
    if loss == "linex":
        return float(np.exp(linex_a * e) - linex_a * e - 1.0)
    raise ValueError("loss must be 'mse' or 'linex'")


def _expert_losses(y_t: float, f_t: np.ndarray, loss: LossType, linex_a: float) -> np.ndarray:
    e = y_t - f_t
    if loss == "mse":
        return e * e
    if loss == "linex":
        return _linex(e, linex_a)
    raise ValueError("loss must be 'mse' or 'linex'")


@dataclass
class BanditRunResult:
    yhat: np.ndarray
    weights: np.ndarray
    loss_t: np.ndarray
    reward_t: np.ndarray
    hhi_t: np.ndarray
    meta: Dict[str, np.ndarray]


class _LinUCB:
    def __init__(self, n_actions: int, context_dim: int, alpha: float = 0.5):
        self.n_actions = int(n_actions)
        self.context_dim = int(context_dim)
        self.alpha = float(alpha)
        self.a = [np.eye(self.context_dim) for _ in range(self.n_actions)]
        self.b = [np.zeros(self.context_dim) for _ in range(self.n_actions)]

    def select(self, x: np.ndarray) -> int:
        scores: List[float] = []
        for k in range(self.n_actions):
            a_inv = np.linalg.inv(self.a[k])
            theta = a_inv @ self.b[k]
            mean = float(x @ theta)
            unc = self.alpha * float(np.sqrt(x @ a_inv @ x))
            scores.append(mean + unc)
        return int(np.argmax(scores))

    def update(self, k: int, x: np.ndarray, reward: float) -> None:
        self.a[k] += np.outer(x, x)
        self.b[k] += reward * x


class _OnlineRule:
    def __init__(
        self,
        n_forecasters: int,
        kind: str,
        eta: float = 0.1,
        kappa: float = 1.0,
        loss: LossType = "mse",
        linex_a: float = 1.0,
    ):
        self.n = int(n_forecasters)
        self.kind = kind
        self.eta = float(eta)
        self.kappa = float(kappa)
        self.loss = loss
        self.linex_a = float(linex_a)
        self.pi = np.ones(self.n) / self.n
        self.w = self.pi.copy()

    def _norm(self, w: np.ndarray) -> np.ndarray:
        w = np.maximum(w, 0.0)
        s = float(np.sum(w))
        if s <= 0.0:
            return self.pi.copy()
        return w / s

    def _project_simplex(self, v: np.ndarray) -> np.ndarray:
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        rho = np.where(u - cssv / (np.arange(v.size) + 1) > 0)[0]
        if rho.size == 0:
            return self.pi.copy()
        rho_idx = int(rho[-1])
        theta = cssv[rho_idx] / (rho_idx + 1.0)
        return self._norm(np.maximum(v - theta, 0.0))

    def _dlde(self, e: float) -> float:
        if self.loss == "mse":
            return 2.0 * e
        return float(self.linex_a * (np.exp(self.linex_a * e) - 1.0))

    def step(self, f_t: np.ndarray, y_t: float, s_t: float = 1.0) -> np.ndarray:
        w_pred = self.w.copy()
        e = float(y_t - w_pred @ f_t)
        grad = -self._dlde(e) * f_t
        ell = _expert_losses(y_t, f_t, self.loss, self.linex_a)
        lam = max(self.kappa * float(s_t), 0.0)

        if self.kind == "mean":
            self.w = self.pi.copy()
            return w_pred
        if self.kind == "ogd":
            self.w = self._project_simplex(self.w - self.eta * grad)
            return w_pred
        if self.kind == "mwum":
            logw = np.log(np.clip(self.w, 1e-300, None)) - self.eta * ell
            self.w = _softmax(logw)
            return w_pred
        if self.kind == "ogd_both":
            denom = 1.0 + 2.0 * self.eta * lam
            w_tilde = (self.w - self.eta * grad + 2.0 * self.eta * lam * self.pi) / denom
            self.w = self._project_simplex(w_tilde)
            return w_pred
        if self.kind == "ogd_conc":
            if lam == 0.0:
                self.w = self._project_simplex(-grad)
            else:
                self.w = self._project_simplex(self.pi - grad / (2.0 * lam))
            return w_pred
        if self.kind == "mwum_both":
            alpha = 1.0 / (1.0 + self.eta * lam)
            beta = (self.eta * lam) / (1.0 + self.eta * lam)
            scale = self.eta / (1.0 + self.eta * lam)
            logw_new = (
                alpha * np.log(np.clip(self.w, 1e-300, None))
                + beta * np.log(np.clip(self.pi, 1e-300, None))
                - scale * ell
            )
            self.w = _softmax(logw_new)
            return w_pred
        if self.kind == "mwum_conc":
            if lam == 0.0:
                i_star = int(np.argmin(ell))
                self.w = np.zeros(self.n)
                self.w[i_star] = 1.0
            else:
                self.w = _softmax(np.log(np.clip(self.pi, 1e-300, None)) - ell / lam)
            return w_pred
        raise ValueError(f"Unknown rule kind: {self.kind}")


class RuleSelectionBandit:
    """
    Contextual bandit that selects one online weighting rule per period.
    """

    def __init__(
        self,
        n_forecasters: int,
        context_dim: int,
        alpha: float = 0.5,
        action_specs: Optional[List[Dict[str, float | str]]] = None,
        loss: LossType = "mse",
        linex_a: float = 1.0,
    ):
        self.n_forecasters = int(n_forecasters)
        self.context_dim = int(context_dim)
        self.loss = loss
        self.linex_a = float(linex_a)
        if action_specs is None:
            action_specs = [
                {"name": "Mean", "kind": "mean"},
                {"name": "OGDVanilla", "kind": "ogd", "eta": 0.05},
                {"name": "MWUMVanilla", "kind": "mwum", "eta": 0.3},
                {"name": "OGDBoth", "kind": "ogd_both", "eta": 0.05, "kappa": 0.8},
                {"name": "OGDConcOnly", "kind": "ogd_conc", "kappa": 0.8},
                {"name": "MWUMBothKL", "kind": "mwum_both", "eta": 0.3, "kappa": 0.8},
                {"name": "MWUMConcOnlyKL", "kind": "mwum_conc", "kappa": 0.8},
            ]
        self.action_names = [str(spec["name"]) for spec in action_specs]
        self.actions: List[_OnlineRule] = []
        for spec in action_specs:
            self.actions.append(
                _OnlineRule(
                    n_forecasters=self.n_forecasters,
                    kind=str(spec["kind"]),
                    eta=float(spec.get("eta", 0.1)),
                    kappa=float(spec.get("kappa", 1.0)),
                    loss=self.loss,
                    linex_a=self.linex_a,
                )
            )
        self.bandit = _LinUCB(len(self.actions), self.context_dim, alpha=alpha)

    def run(self, F: np.ndarray, y: np.ndarray, X: np.ndarray, s: Optional[np.ndarray] = None) -> BanditRunResult:
        F = np.asarray(F, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        X = np.asarray(X, dtype=float)
        t, n = F.shape
        if y.size != t or X.shape != (t, self.context_dim):
            raise ValueError("Need y:(T,), X:(T,d), F:(T,N)")
        if n != self.n_forecasters:
            raise ValueError("F has unexpected number of forecasters")
        s_arr = np.ones(t) if s is None else np.asarray(s, dtype=float).reshape(-1)
        if s_arr.size != t:
            raise ValueError("s must have length T")

        yhat = np.full(t, np.nan)
        W = np.full((t, n), np.nan)
        loss_t = np.full(t, np.nan)
        reward_t = np.full(t, np.nan)
        hhi_t = np.full(t, np.nan)
        actions_t = np.full(t, -1, dtype=int)

        for i in range(t):
            if not (np.all(np.isfinite(F[i])) and np.isfinite(y[i]) and np.all(np.isfinite(X[i]))):
                continue
            k = self.bandit.select(X[i])
            actions_t[i] = k
            w_i = self.actions[k].step(F[i], float(y[i]), float(s_arr[i]))
            yhat_i = float(w_i @ F[i])
            l_i = _loss_scalar(float(y[i]), yhat_i, self.loss, self.linex_a)
            r_i = -l_i
            self.bandit.update(k, X[i], r_i)

            yhat[i] = yhat_i
            W[i] = w_i
            loss_t[i] = l_i
            reward_t[i] = r_i
            hhi_t[i] = _hhi(w_i)

        return BanditRunResult(
            yhat=yhat,
            weights=W,
            loss_t=loss_t,
            reward_t=reward_t,
            hhi_t=hhi_t,
            meta={"actions_t": actions_t, "action_names": np.array(self.action_names, dtype=object)},
        )


class KappaBandit:
    """
    Contextual bandit that chooses kappa each period for concentration-only MWUM update.

    Timing convention is strictly online:
      - choose kappa_t from context x_t
      - predict with current weights w_t
      - observe y_t and receive reward from prediction loss
      - update bandit model with (x_t, reward_t)
      - update weights to w_{t+1} using chosen kappa_t and realised expert losses at t
    """

    def __init__(
        self,
        kappa_grid: np.ndarray,
        context_dim: int,
        alpha: float = 0.5,
        loss: LossType = "mse",
        linex_a: float = 1.0,
    ):
        self.kappa_grid = np.asarray(kappa_grid, dtype=float).reshape(-1)
        if self.kappa_grid.size == 0:
            raise ValueError("kappa_grid must be non-empty")
        self.context_dim = int(context_dim)
        self.loss = loss
        self.linex_a = float(linex_a)
        self.bandit = _LinUCB(self.kappa_grid.size, self.context_dim, alpha=alpha)

    def run(self, F: np.ndarray, y: np.ndarray, X: np.ndarray, s: np.ndarray) -> BanditRunResult:
        F = np.asarray(F, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        X = np.asarray(X, dtype=float)
        s = np.asarray(s, dtype=float).reshape(-1)
        t, n = F.shape
        if y.size != t or X.shape != (t, self.context_dim) or s.size != t:
            raise ValueError("Need y:(T,), X:(T,d), s:(T,), F:(T,N)")

        pi = np.ones(n) / n
        w = pi.copy()
        yhat = np.full(t, np.nan)
        W = np.full((t, n), np.nan)
        loss_t = np.full(t, np.nan)
        reward_t = np.full(t, np.nan)
        hhi_t = np.full(t, np.nan)
        kappa_t = np.full(t, np.nan)
        action_t = np.full(t, -1, dtype=int)
        lambda_t = np.full(t, np.nan)

        for i in range(t):
            if not (np.all(np.isfinite(F[i])) and np.isfinite(y[i]) and np.all(np.isfinite(X[i])) and np.isfinite(s[i])):
                continue
            k = self.bandit.select(X[i])
            kappa = float(self.kappa_grid[k])
            lam = max(1e-6, kappa * float(s[i]))

            # Predict with current weights (pre-update).
            w_i = w.copy()
            yhat_i = float(w_i @ F[i])
            l_i = _loss_scalar(float(y[i]), yhat_i, self.loss, self.linex_a)
            r_i = -l_i
            self.bandit.update(k, X[i], r_i)

            # Post-prediction update to next-step weights using realised expert losses.
            ell = _expert_losses(float(y[i]), F[i], self.loss, self.linex_a)
            w = _softmax(np.log(np.clip(pi, 1e-300, None)) - ell / lam)

            yhat[i] = yhat_i
            W[i] = w_i
            loss_t[i] = l_i
            reward_t[i] = r_i
            hhi_t[i] = _hhi(w_i)
            kappa_t[i] = kappa
            action_t[i] = k
            lambda_t[i] = lam

        return BanditRunResult(
            yhat=yhat,
            weights=W,
            loss_t=loss_t,
            reward_t=reward_t,
            hhi_t=hhi_t,
            meta={"kappa_t": kappa_t, "actions_t": action_t, "lambda_t": lambda_t},
        )

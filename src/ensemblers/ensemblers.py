# forecast_ensembles.py
"""
Forecast combination ensemblers (online/sequential).

Implements exactly the following ensemblers:

(1) Mean (equal weights):           w_{i,t} = 1/N
(2) Median aggregator:              y^c_t = median_i \hat y_{i,t}

(3) OGD (vanilla, projected):       w_{t+1} = Π_Δ( w_t - η ∇_w ℓ_t(w_t) )
    Equivalent prox form:
        w_{t+1} = argmin_{w∈Δ} <∇ℓ_t(w_t), w> + (1/(2η))||w-w_t||_2^2

(4) MWUM (vanilla):                 w_{i,t+1} ∝ w_{i,t} exp( -η ℓ_{i,t} )
    Mirror descent:
        w_{t+1} = argmin_{w∈Δ} <w, ℓ_t> + (1/η) KL(w || w_t)

(5) OGD + Euclidean concentration (both penalties):
        w_{t+1} = argmin_{w∈Δ} <∇ℓ_t(w_t), w> + (1/(2η))||w-w_t||_2^2 + λ_t||w-π||_2^2
    with λ_t = κ s_t (state-dependent). Hyperparameters: (η, κ).

(6) OGD concentration-only (no adjustment penalty):
        w_{t+1} = argmin_{w∈Δ} <∇ℓ_t(w_t), w> + λ_t||w-π||_2^2
    with λ_t = κ s_t. Hyperparameters: κ (and optionally η if you scale the gradient).

(7) MWUM + KL concentration (both):
        w_{t+1} = argmin_{w∈Δ} <w, ℓ_t> + (1/η) KL(w||w_t) + λ_t KL(w||π)
    with λ_t = κ s_t. Hyperparameters: (η, κ).

(8) MWUM concentration-only (no adjustment KL):
        w_{t+1} = argmin_{w∈Δ} <w, ℓ_t> + λ_t KL(w||π)
    with λ_t = κ s_t. Hyperparameters: κ.

Notes
-----
- Δ denotes the unit simplex: {w ≥ 0, 1'w = 1}.
- π is the baseline weight vector (default uniform 1/N).
- Loss choices provided:
    * Squared: L(e)=e^2
    * LINEX:   L(e)=exp(a e) - a e - 1, with a>0
  For OGD, we use the gradient via dL/de.
- Data interface: forecasts[t, i] predicts y[t] (synchronous). If you use forecast horizons,
  just align your arrays before calling these classes.

Typical usage
-------------
    ens = OGDConcentrationBoth(eta=0.2, kappa=1.0, baseline="uniform", loss="squared")
    yhat_c, W = ens.run(forecasts, y, s=s_t)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict
import numpy as np


LossName = Literal["squared", "linex"]


# -----------------------------
# Utilities
# -----------------------------

def _check_2d_forecasts(forecasts: np.ndarray) -> Tuple[int, int]:
    if forecasts.ndim != 2:
        raise ValueError(f"`forecasts` must be 2D array (T,N); got shape {forecasts.shape}")
    T, N = forecasts.shape
    if N < 1 or T < 1:
        raise ValueError("`forecasts` must have T>=1 and N>=1.")
    return T, N


def _simplex_projection(v: np.ndarray) -> np.ndarray:
    """
    Euclidean projection onto the unit simplex Δ = {w>=0, sum w = 1}.
    Duchi et al. (2008) algorithm.
    """
    v = np.asarray(v, dtype=float)
    if v.ndim != 1:
        raise ValueError("Projection expects a 1D vector.")
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.where(u - cssv / (np.arange(n) + 1) > 0)[0]
    if rho.size == 0:
        # fallback: uniform
        return np.ones(n) / n
    rho = rho[-1]
    theta = cssv[rho] / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    # numerical cleanup
    s = w.sum()
    if s <= 0:
        return np.ones(n) / n
    return w / s


def _baseline_pi(N: int, baseline: Literal["uniform"] | np.ndarray = "uniform") -> np.ndarray:
    if isinstance(baseline, str):
        if baseline != "uniform":
            raise ValueError("Only baseline='uniform' supported as string.")
        return np.ones(N) / N
    pi = np.asarray(baseline, dtype=float).reshape(-1)
    if pi.size != N:
        raise ValueError(f"Baseline pi must have length N={N}, got {pi.size}.")
    if np.any(pi < 0) or not np.isfinite(pi).all():
        raise ValueError("Baseline pi must be nonnegative and finite.")
    s = pi.sum()
    if abs(s - 1.0) > 1e-8:
        pi = pi / s
    return pi


def _loss_and_grad_e(e: float, loss: LossName, linex_a: float = 1.0) -> Tuple[float, float]:
    """
    Returns (L(e), dL/de).
    """
    if loss == "squared":
        L = e * e
        dL = 2.0 * e
        return L, dL
    if loss == "linex":
        a = float(linex_a)
        # L(e) = exp(a e) - a e - 1
        expae = float(np.exp(a * e))
        L = expae - a * e - 1.0
        dL = a * (expae - 1.0)
        return L, dL
    raise ValueError(f"Unknown loss: {loss}")


def _ensure_state(T: int, s: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if s is None:
        return None
    s = np.asarray(s, dtype=float).reshape(-1)
    if s.size != T:
        raise ValueError(f"State series s must have length T={T}, got {s.size}.")
    return s


# -----------------------------
# Base interface
# -----------------------------

@dataclass
class EnsembleResult:
    yhat: np.ndarray        # (T,)
    weights: np.ndarray     # (T,N) weights used at each t (pre-update)
    meta: Dict[str, np.ndarray]  # optional diagnostics


class BaseEnsembler:
    """
    Base class for sequential forecast combiners.

    Conventions:
    - At time t, you form combined forecast with current weights w_t:
          yhat_c[t] = w_t' forecasts[t]
    - You then observe y[t] and update weights to w_{t+1}.
    - Returned `weights[t]` equals w_t (weights used to predict y[t]).
    """

    def run(
        self,
        forecasts: np.ndarray,
        y: np.ndarray,
        s: Optional[np.ndarray] = None,
    ) -> EnsembleResult:
        raise NotImplementedError


# -----------------------------
# Simple baselines
# -----------------------------

class MeanEnsembler(BaseEnsembler):
    """Equal-weight mean: w_{i,t}=1/N for all t."""
    def run(self, forecasts: np.ndarray, y: np.ndarray, s: Optional[np.ndarray] = None) -> EnsembleResult:
        T, N = _check_2d_forecasts(forecasts)
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != T:
            raise ValueError(f"y must have length T={T}, got {y.size}")
        w = np.ones(N) / N
        yhat = forecasts @ w
        W = np.tile(w, (T, 1))
        return EnsembleResult(yhat=yhat, weights=W, meta={})


class MedianEnsembler(BaseEnsembler):
    """Median aggregator: y^c_t = median_i forecasts[t,i]."""
    def run(self, forecasts: np.ndarray, y: np.ndarray, s: Optional[np.ndarray] = None) -> EnsembleResult:
        T, N = _check_2d_forecasts(forecasts)
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != T:
            raise ValueError(f"y must have length T={T}, got {y.size}")
        yhat = np.median(forecasts, axis=1)
        # weights are not linear; return NaNs
        W = np.full((T, N), np.nan)
        return EnsembleResult(yhat=yhat, weights=W, meta={})


# -----------------------------
# OGD family
# -----------------------------

@dataclass
class OGDVanilla(BaseEnsembler):
    """
    Projected Online Gradient Descent (Euclidean).

    Update:
        w_{t+1} = Π_Δ( w_t - η ∇_w ℓ_t(w_t) )

    Loss:
        ℓ_t(w) = L( y[t] - w' f[t] )
    """
    eta: float = 0.1
    loss: LossName = "squared"
    linex_a: float = 1.0
    baseline: Literal["uniform"] | np.ndarray = "uniform"
    w0: Optional[np.ndarray] = None

    def run(self, forecasts: np.ndarray, y: np.ndarray, s: Optional[np.ndarray] = None) -> EnsembleResult:
        T, N = _check_2d_forecasts(forecasts)
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != T:
            raise ValueError(f"y must have length T={T}, got {y.size}")

        pi = _baseline_pi(N, self.baseline)
        w = _simplex_projection(pi if self.w0 is None else self.w0)

        yhat = np.zeros(T)
        W = np.zeros((T, N))
        losses = np.zeros(T)

        eta = float(self.eta)
        if eta <= 0:
            raise ValueError("eta must be > 0")

        for t in range(T):
            f = forecasts[t]
            W[t] = w
            yhat[t] = float(w @ f)
            e = float(y[t] - yhat[t])
            L, dLde = _loss_and_grad_e(e, self.loss, self.linex_a)
            losses[t] = L
            # ∂/∂w L(y - w'f) = - dL/de * f
            grad_w = -dLde * f
            w = _simplex_projection(w - eta * grad_w)

        return EnsembleResult(yhat=yhat, weights=W, meta={"loss": losses})


@dataclass
class OGDConcentrationBoth(BaseEnsembler):
    """
    OGD with BOTH:
    - Euclidean adjustment penalty (relative to w_t)
    - Euclidean concentration penalty (relative to baseline π)

    Prox update:
        w_{t+1} = argmin_{w∈Δ} <∇ℓ_t(w_t), w> + (1/(2η))||w-w_t||^2 + λ_t||w-π||^2
    with λ_t = κ s_t.

    Hyperparameters: eta, kappa.
    """
    eta: float = 0.1
    kappa: float = 1.0
    loss: LossName = "squared"
    linex_a: float = 1.0
    baseline: Literal["uniform"] | np.ndarray = "uniform"
    w0: Optional[np.ndarray] = None

    def run(self, forecasts: np.ndarray, y: np.ndarray, s: Optional[np.ndarray] = None) -> EnsembleResult:
        T, N = _check_2d_forecasts(forecasts)
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != T:
            raise ValueError(f"y must have length T={T}, got {y.size}")
        s = _ensure_state(T, s)
        if s is None:
            raise ValueError("This ensembler requires a state series s (for λ_t=κ s_t).")

        pi = _baseline_pi(N, self.baseline)
        w = _simplex_projection(pi if self.w0 is None else self.w0)

        yhat = np.zeros(T)
        W = np.zeros((T, N))
        losses = np.zeros(T)
        lambdas = np.zeros(T)

        eta = float(self.eta)
        kappa = float(self.kappa)
        if eta <= 0:
            raise ValueError("eta must be > 0")
        if kappa < 0:
            raise ValueError("kappa must be >= 0")

        for t in range(T):
            f = forecasts[t]
            W[t] = w
            yhat[t] = float(w @ f)
            e = float(y[t] - yhat[t])
            L, dLde = _loss_and_grad_e(e, self.loss, self.linex_a)
            losses[t] = L
            grad_w = -dLde * f

            lam = kappa * float(s[t])
            lam = max(lam, 0.0)
            lambdas[t] = lam

            # Unconstrained closed-form (then project)
            # w_tilde = (w - eta grad + 2 eta lam pi) / (1 + 2 eta lam)
            denom = 1.0 + 2.0 * eta * lam
            w_tilde = (w - eta * grad_w + 2.0 * eta * lam * pi) / denom
            w = _simplex_projection(w_tilde)

        return EnsembleResult(yhat=yhat, weights=W, meta={"loss": losses, "lambda": lambdas})


@dataclass
class OGDConcentrationOnly(BaseEnsembler):
    """
    OGD with concentration penalty ONLY (no adjustment penalty).

    Prox update:
        w_{t+1} = argmin_{w∈Δ} <∇ℓ_t(w_t), w> + λ_t||w-π||^2
    with λ_t = κ s_t.

    Note: as written, there is no explicit η term in the objective. You can think of κ absorbing scaling.
    """
    kappa: float = 1.0
    loss: LossName = "squared"
    linex_a: float = 1.0
    baseline: Literal["uniform"] | np.ndarray = "uniform"
    w0: Optional[np.ndarray] = None

    def run(self, forecasts: np.ndarray, y: np.ndarray, s: Optional[np.ndarray] = None) -> EnsembleResult:
        T, N = _check_2d_forecasts(forecasts)
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != T:
            raise ValueError(f"y must have length T={T}, got {y.size}")
        s = _ensure_state(T, s)
        if s is None:
            raise ValueError("This ensembler requires a state series s (for λ_t=κ s_t).")

        pi = _baseline_pi(N, self.baseline)
        w = _simplex_projection(pi if self.w0 is None else self.w0)

        yhat = np.zeros(T)
        W = np.zeros((T, N))
        losses = np.zeros(T)
        lambdas = np.zeros(T)

        kappa = float(self.kappa)
        if kappa < 0:
            raise ValueError("kappa must be >= 0")

        for t in range(T):
            f = forecasts[t]
            W[t] = w
            yhat[t] = float(w @ f)
            e = float(y[t] - yhat[t])
            L, dLde = _loss_and_grad_e(e, self.loss, self.linex_a)
            losses[t] = L
            grad_w = -dLde * f

            lam = kappa * float(s[t])
            lam = max(lam, 0.0)
            lambdas[t] = lam

            # Solve unconstrained quadratic+linear:
            # minimize <grad, w> + lam ||w-pi||^2
            # FOC: grad + 2 lam (w - pi) = 0 -> w = pi - grad/(2 lam)
            if lam == 0.0:
                # no penalty; take a pure "greedy" linear step to boundary: project (-grad) direction
                w = _simplex_projection(-grad_w)
            else:
                w_tilde = pi - grad_w / (2.0 * lam)
                w = _simplex_projection(w_tilde)

        return EnsembleResult(yhat=yhat, weights=W, meta={"loss": losses, "lambda": lambdas})


# -----------------------------
# MWUM family
# -----------------------------

@dataclass
class MWUMVanilla(BaseEnsembler):
    """
    Vanilla multiplicative weights update method:
        w_{i,t+1} ∝ w_{i,t} exp( -η ℓ_{i,t} )

    Here ℓ_{i,t} is per-expert loss L(y - forecast_i).
    """
    eta: float = 0.5
    loss: LossName = "squared"
    linex_a: float = 1.0
    baseline: Literal["uniform"] | np.ndarray = "uniform"
    w0: Optional[np.ndarray] = None

    def run(self, forecasts: np.ndarray, y: np.ndarray, s: Optional[np.ndarray] = None) -> EnsembleResult:
        T, N = _check_2d_forecasts(forecasts)
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != T:
            raise ValueError(f"y must have length T={T}, got {y.size}")

        pi = _baseline_pi(N, self.baseline)
        w = _simplex_projection(pi if self.w0 is None else self.w0)

        yhat = np.zeros(T)
        W = np.zeros((T, N))
        avg_loss = np.zeros(T)

        eta = float(self.eta)
        if eta <= 0:
            raise ValueError("eta must be > 0")

        for t in range(T):
            f = forecasts[t]
            W[t] = w
            yhat[t] = float(w @ f)

            # per-expert losses
            ell = np.zeros(N)
            for i in range(N):
                e_i = float(y[t] - f[i])
                ell[i], _ = _loss_and_grad_e(e_i, self.loss, self.linex_a)

            avg_loss[t] = float(w @ ell)

            # multiplicative update
            logw = np.log(np.clip(w, 1e-300, None)) - eta * ell
            logw -= np.max(logw)
            w = np.exp(logw)
            w = w / w.sum()

        return EnsembleResult(yhat=yhat, weights=W, meta={"avg_loss": avg_loss})


@dataclass
class MWUMBothKL(BaseEnsembler):
    """
    MWUM with BOTH:
    - adjustment KL: (1/η) KL(w || w_t)
    - concentration KL: λ_t KL(w || π), with λ_t=κ s_t

    We implement this via the closed-form for the minimizer of:
        min_{w∈Δ} <w, ℓ_t> + (1/η) KL(w||w_t) + λ_t KL(w||π)

    Solution:
        w_i ∝ (w_{t,i})^{1/(1+ηλ_t)} * (π_i)^{(ηλ_t)/(1+ηλ_t)} * exp( -η ℓ_{i,t}/(1+ηλ_t) )
    """
    eta: float = 0.5
    kappa: float = 1.0
    loss: LossName = "squared"
    linex_a: float = 1.0
    baseline: Literal["uniform"] | np.ndarray = "uniform"
    w0: Optional[np.ndarray] = None

    def run(self, forecasts: np.ndarray, y: np.ndarray, s: Optional[np.ndarray] = None) -> EnsembleResult:
        T, N = _check_2d_forecasts(forecasts)
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != T:
            raise ValueError(f"y must have length T={T}, got {y.size}")
        s = _ensure_state(T, s)
        if s is None:
            raise ValueError("This ensembler requires a state series s (for λ_t=κ s_t).")

        pi = _baseline_pi(N, self.baseline)
        w = _simplex_projection(pi if self.w0 is None else self.w0)

        yhat = np.zeros(T)
        W = np.zeros((T, N))
        avg_loss = np.zeros(T)
        lambdas = np.zeros(T)

        eta = float(self.eta)
        kappa = float(self.kappa)
        if eta <= 0:
            raise ValueError("eta must be > 0")
        if kappa < 0:
            raise ValueError("kappa must be >= 0")

        for t in range(T):
            f = forecasts[t]
            W[t] = w
            yhat[t] = float(w @ f)

            ell = np.zeros(N)
            for i in range(N):
                e_i = float(y[t] - f[i])
                ell[i], _ = _loss_and_grad_e(e_i, self.loss, self.linex_a)
            avg_loss[t] = float(w @ ell)

            lam = max(kappa * float(s[t]), 0.0)
            lambdas[t] = lam

            # closed-form combined KL solution
            alpha = 1.0 / (1.0 + eta * lam)           # weight on past w_t in exponent
            beta = (eta * lam) / (1.0 + eta * lam)    # weight on baseline π in exponent
            scale = eta / (1.0 + eta * lam)

            logw_new = alpha * np.log(np.clip(w, 1e-300, None)) + beta * np.log(np.clip(pi, 1e-300, None)) - scale * ell
            logw_new -= np.max(logw_new)
            w = np.exp(logw_new)
            w = w / w.sum()

        return EnsembleResult(yhat=yhat, weights=W, meta={"avg_loss": avg_loss, "lambda": lambdas})


@dataclass
class MWUMConcentrationOnlyKL(BaseEnsembler):
    """
    MWUM concentration-only (no adjustment KL):

        w_{t+1} = argmin_{w∈Δ} <w, ℓ_t> + λ_t KL(w||π),  λ_t=κ s_t

    Closed form (softmax around π):
        w_i ∝ π_i * exp( -ℓ_{i,t} / λ_t )

    As λ_t -> ∞, w -> π (uniform if π uniform).
    As λ_t -> 0, w concentrates on argmin_i ℓ_{i,t}.
    """
    kappa: float = 1.0
    loss: LossName = "squared"
    linex_a: float = 1.0
    baseline: Literal["uniform"] | np.ndarray = "uniform"
    w0: Optional[np.ndarray] = None  # unused, but kept for interface symmetry

    def run(self, forecasts: np.ndarray, y: np.ndarray, s: Optional[np.ndarray] = None) -> EnsembleResult:
        T, N = _check_2d_forecasts(forecasts)
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != T:
            raise ValueError(f"y must have length T={T}, got {y.size}")
        s = _ensure_state(T, s)
        if s is None:
            raise ValueError("This ensembler requires a state series s (for λ_t=κ s_t).")

        pi = _baseline_pi(N, self.baseline)
        w = _simplex_projection(pi)  # start at baseline

        yhat = np.zeros(T)
        W = np.zeros((T, N))
        avg_loss = np.zeros(T)
        lambdas = np.zeros(T)

        kappa = float(self.kappa)
        if kappa < 0:
            raise ValueError("kappa must be >= 0")

        for t in range(T):
            f = forecasts[t]
            W[t] = w
            yhat[t] = float(w @ f)

            ell = np.zeros(N)
            for i in range(N):
                e_i = float(y[t] - f[i])
                ell[i], _ = _loss_and_grad_e(e_i, self.loss, self.linex_a)

            avg_loss[t] = float(w @ ell)

            lam = max(kappa * float(s[t]), 0.0)
            lambdas[t] = lam

            if lam == 0.0:
                # concentrates on argmin loss: put all mass on best expert
                i_star = int(np.argmin(ell))
                w = np.zeros(N)
                w[i_star] = 1.0
            else:
                logw = np.log(np.clip(pi, 1e-300, None)) - ell / lam
                logw -= np.max(logw)
                w = np.exp(logw)
                w = w / w.sum()

        return EnsembleResult(yhat=yhat, weights=W, meta={"avg_loss": avg_loss, "lambda": lambdas})

"""Online ensemble forecasting update rules.

Provides vanilla Online Gradient Descent (OGD) and Multiplicative Weights (MW)
plus KL-regularized variants that penalize deviation from equal weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


ArrayLike = np.ndarray


def _validate_forecasts(forecasts: ArrayLike, y_true: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    if forecasts.ndim != 2:
        raise ValueError("forecasts must be 2D: (n_models, n_steps)")
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1D: (n_steps,)")
    if forecasts.shape[1] != y_true.shape[0]:
        raise ValueError("forecasts and y_true must have matching time dimension")
    return forecasts.astype(float, copy=False), y_true.astype(float, copy=False)


def _normalize(weights: ArrayLike) -> ArrayLike:
    w = np.clip(weights, 0.0, np.inf)
    total = w.sum()
    if total <= 0:
        # fallback to uniform
        return np.ones_like(w) / w.size
    return w / total


def _uniform(n: int) -> ArrayLike:
    return np.ones(n, dtype=float) / n


@dataclass
class OnlineGradientDescent:
    """Vanilla OGD over simplex for ensemble weights.

    Loss: squared error of ensemble forecast.
    """

    eta: float = 0.1
    n_models: Optional[int] = None

    def _init_weights(self, n_models: int) -> None:
        self.n_models = n_models
        self.weights = _uniform(n_models)

    def predict(self, forecasts_t: ArrayLike) -> float:
        return float(np.dot(self.weights, forecasts_t))

    def update(self, forecasts_t: ArrayLike, y_t: float) -> ArrayLike:
        # gradient of 0.5*(w·f - y)^2 wrt w is (w·f - y)*f
        err = np.dot(self.weights, forecasts_t) - y_t
        grad = err * forecasts_t
        self.weights = _normalize(self.weights - self.eta * grad)
        return self.weights

    def fit(self, forecasts: ArrayLike, y_true: ArrayLike) -> ArrayLike:
        forecasts, y_true = _validate_forecasts(forecasts, y_true)
        n_models, n_steps = forecasts.shape
        if self.n_models is None:
            self._init_weights(n_models)
        if self.n_models != n_models:
            raise ValueError("n_models mismatch")
        history = np.zeros((n_steps, n_models), dtype=float)
        for t in range(n_steps):
            self.update(forecasts[:, t], y_true[t])
            history[t] = self.weights
        return history


@dataclass
class MultiplicativeWeights:
    """Vanilla multiplicative weights with squared-loss experts."""

    eta: float = 0.5
    n_models: Optional[int] = None

    def _init_weights(self, n_models: int) -> None:
        self.n_models = n_models
        self.weights = _uniform(n_models)

    def predict(self, forecasts_t: ArrayLike) -> float:
        return float(np.dot(self.weights, forecasts_t))

    def update(self, forecasts_t: ArrayLike, y_t: float) -> ArrayLike:
        losses = 0.5 * (forecasts_t - y_t) ** 2
        self.weights = _normalize(self.weights * np.exp(-self.eta * losses))
        return self.weights

    def fit(self, forecasts: ArrayLike, y_true: ArrayLike) -> ArrayLike:
        forecasts, y_true = _validate_forecasts(forecasts, y_true)
        n_models, n_steps = forecasts.shape
        if self.n_models is None:
            self._init_weights(n_models)
        if self.n_models != n_models:
            raise ValueError("n_models mismatch")
        history = np.zeros((n_steps, n_models), dtype=float)
        for t in range(n_steps):
            self.update(forecasts[:, t], y_true[t])
            history[t] = self.weights
        return history


@dataclass
class KLOnlineGradientDescent(OnlineGradientDescent):
    """OGD with KL penalty to the uniform prior.

    Adds gradient of lambda * KL(w || u) where u is uniform.
    """

    kl_lambda: float = 0.1

    def update(self, forecasts_t: ArrayLike, y_t: float) -> ArrayLike:
        err = np.dot(self.weights, forecasts_t) - y_t
        grad = err * forecasts_t
        # gradient of KL(w||u) is log(w/u) + 1
        u = _uniform(self.weights.size)
        kl_grad = np.log(np.clip(self.weights, 1e-12, 1.0) / u) + 1.0
        self.weights = _normalize(self.weights - self.eta * (grad + self.kl_lambda * kl_grad))
        return self.weights


@dataclass
class KLMultiplicativeWeights(MultiplicativeWeights):
    """Multiplicative weights with KL penalty toward uniform prior.

    Implemented as a convex combination between current weights and uniform,
    controlled by kl_lambda (0 -> none, 1 -> full reset to uniform).
    """

    kl_lambda: float = 0.1

    def update(self, forecasts_t: ArrayLike, y_t: float) -> ArrayLike:
        losses = 0.5 * (forecasts_t - y_t) ** 2
        w = _normalize(self.weights * np.exp(-self.eta * losses))
        u = _uniform(w.size)
        self.weights = _normalize((1.0 - self.kl_lambda) * w + self.kl_lambda * u)
        return self.weights


__all__ = [
    "OnlineGradientDescent",
    "MultiplicativeWeights",
    "KLOnlineGradientDescent",
    "KLMultiplicativeWeights",
]

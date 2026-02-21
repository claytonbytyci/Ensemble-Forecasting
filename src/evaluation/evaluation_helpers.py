from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List


# ----------------------------
# Core loss utilities
# ----------------------------

def _align_valid(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """Boolean mask where both y and yhat are finite."""
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    return np.isfinite(y) & np.isfinite(yhat)

def mse(y: np.ndarray, yhat: np.ndarray) -> float:
    m = _align_valid(y, yhat)
    if m.sum() == 0:
        return np.nan
    e = y[m] - yhat[m]
    return float(np.mean(e**2))

def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    m = _align_valid(y, yhat)
    if m.sum() == 0:
        return np.nan
    e = y[m] - yhat[m]
    return float(np.mean(np.abs(e)))

def linex_loss(y: np.ndarray, yhat: np.ndarray, a: float = 1.0) -> float:
    """
    LINEX loss averaged over t: L(e)=exp(a e) - a e - 1.
    a>0 penalises underprediction more if e=y-yhat>0 (i.e., y>yhat).
    For your 'overprediction of inflation is worse', set a<0 (since e<0 then penalised more).
    """
    m = _align_valid(y, yhat)
    if m.sum() == 0:
        return np.nan
    e = (y[m] - yhat[m])
    L = np.exp(a * e) - a * e - 1.0
    return float(np.mean(L))


# ----------------------------
# Best single forecaster
# ----------------------------

def best_forecaster_yhat(
    F: np.ndarray, y: np.ndarray, metric: str = "mse"
) -> Tuple[np.ndarray, int, float]:
    """
    F: (T,N) forecast matrix for individual forecasters.
    Returns: (best_yhat, best_idx, best_score)
    """
    F = np.asarray(F, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if F.ndim != 2:
        raise ValueError("F must be (T,N)")
    T, N = F.shape
    if y.size != T:
        raise ValueError("y length must match F rows")

    scores = []
    for i in range(N):
        yhat_i = F[:, i]
        if metric == "mse":
            s = mse(y, yhat_i)
        elif metric == "mae":
            s = mae(y, yhat_i)
        else:
            raise ValueError("metric must be 'mse' or 'mae'")
        scores.append(s)

    best_idx = int(np.nanargmin(scores))
    best_score = float(scores[best_idx])
    return F[:, best_idx].copy(), best_idx, best_score


# ----------------------------
# Rolling / cumulative loss series
# ----------------------------

def loss_series(
    y: np.ndarray,
    yhat: np.ndarray,
    loss: str = "sq",      # "sq" or "abs" or "linex"
    linex_a: float = 1.0
) -> np.ndarray:
    """Per-period loss ℓ_t."""
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    out = np.full_like(y, np.nan, dtype=float)

    m = _align_valid(y, yhat)
    e = y[m] - yhat[m]

    if loss == "sq":
        out[m] = e**2
    elif loss == "abs":
        out[m] = np.abs(e)
    elif loss == "linex":
        out[m] = np.exp(linex_a * e) - linex_a * e - 1.0
    else:
        raise ValueError("loss must be 'sq', 'abs', or 'linex'")

    return out

def cumulative_loss(loss_t: np.ndarray) -> np.ndarray:
    """Cumulative sum ignoring NaNs (NaNs treated as 0 increments, but remain NaN until first valid)."""
    loss_t = np.asarray(loss_t, dtype=float).reshape(-1)
    cum = np.zeros_like(loss_t)
    seen = False
    running = 0.0
    for t in range(len(loss_t)):
        if np.isfinite(loss_t[t]):
            running += float(loss_t[t])
            seen = True
        cum[t] = running if seen else np.nan
    return cum

def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean ignoring NaNs (requires full window of finite values to output finite)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    out = np.full_like(x, np.nan)
    if window <= 0:
        raise ValueError("window must be positive")
    for t in range(window - 1, len(x)):
        seg = x[t - window + 1 : t + 1]
        if np.all(np.isfinite(seg)):
            out[t] = float(np.mean(seg))
    return out


# ----------------------------
# Plotting helpers
# ----------------------------

def plot_loss_over_time(
    y: np.ndarray,
    yhats: Dict[str, np.ndarray],
    loss: str = "sq",              # "sq" or "abs" or "linex"
    linex_a: float = 1.0,
    mode: str = "cumulative",      # "cumulative" or "rolling"
    rolling_window: int = 40,
    title: Optional[str] = None,
    xlabel: str = "t",
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
    show: bool = True
) -> plt.Figure:
    """
    yhats: dict name -> yhat array (T,)
    Plots either cumulative loss or rolling mean loss.
    """
    y = np.asarray(y, dtype=float).reshape(-1)

    fig = plt.figure(figsize=figsize)
    for name, yhat in yhats.items():
        lt = loss_series(y, yhat, loss=loss, linex_a=linex_a)
        if mode == "cumulative":
            series = cumulative_loss(lt)
            ylab = ylabel or f"Cumulative {loss} loss"
        elif mode == "rolling":
            series = rolling_mean(lt, rolling_window)
            ylab = ylabel or f"Rolling mean {loss} loss (w={rolling_window})"
        else:
            raise ValueError("mode must be 'cumulative' or 'rolling'")

        plt.plot(series, label=name)

    plt.xlabel(xlabel)
    plt.ylabel(ylab)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    return fig


# ----------------------------
# Summary table
# ----------------------------

def loss_table(
    y: np.ndarray,
    F_individual: Optional[np.ndarray],
    yhats: Dict[str, np.ndarray],
    metric: str = "mse",          # "mse" or "mae" or "linex"
    linex_a: float = 1.0,
    include_best_forecaster: bool = True,
    forecaster_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Produces a table with loss for:
      - Best individual forecaster (optional, requires F_individual)
      - Each ensemble in yhats
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    rows = []

    # Best individual
    if include_best_forecaster:
        if F_individual is None:
            raise ValueError("F_individual must be provided to include_best_forecaster.")
        best_yhat, best_idx, best_score = best_forecaster_yhat(
            F_individual, y, metric=("mse" if metric == "mse" else "mae")
        )
        best_name = f"Best forecaster (#{best_idx})"
        if forecaster_names and 0 <= best_idx < len(forecaster_names):
            best_name = f"Best forecaster: {forecaster_names[best_idx]}"
        rows.append((best_name, best_score))

    # Ensembles
    for name, yhat in yhats.items():
        if metric == "mse":
            s = mse(y, yhat)
        elif metric == "mae":
            s = mae(y, yhat)
        elif metric == "linex":
            s = linex_loss(y, yhat, a=linex_a)
        else:
            raise ValueError("metric must be 'mse', 'mae', or 'linex'")
        rows.append((name, s))

    df = pd.DataFrame(rows, columns=["Model", metric.upper()])
    df = df.sort_values(by=metric.upper(), ascending=True).reset_index(drop=True)
    return df


# ----------------------------
# Convenience: end-to-end evaluation
# ----------------------------

def evaluate_and_plot(
    y: np.ndarray,
    F_individual: np.ndarray,
    ensemble_yhats: Dict[str, np.ndarray],
    metric: str = "mse",
    plot_loss_kind: str = "sq",
    plot_mode: str = "cumulative",
    rolling_window: int = 40,
    title_prefix: str = "",
    forecaster_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Adds best-forecaster series to plots automatically, returns summary table.
    """
    best_yhat, best_idx, _ = best_forecaster_yhat(F_individual, y, metric=("mse" if metric == "mse" else "mae"))
    best_label = f"Best forecaster (#{best_idx})"
    if forecaster_names and 0 <= best_idx < len(forecaster_names):
        best_label = f"Best forecaster: {forecaster_names[best_idx]}"

    yhats_for_plot = {best_label: best_yhat, **ensemble_yhats}

    plot_loss_over_time(
        y=y,
        yhats=yhats_for_plot,
        loss=plot_loss_kind,
        mode=plot_mode,
        rolling_window=rolling_window,
        title=(title_prefix + (" " if title_prefix else "") + f"{plot_mode.title()} loss"),
    )

    df = loss_table(
        y=y,
        F_individual=F_individual,
        yhats=ensemble_yhats,
        metric=metric,
        forecaster_names=forecaster_names,
        include_best_forecaster=True
    )
    return df

# Append these to evaluation_helpers.py


# ----------------------------
# Concentration: HHI utilities
# ----------------------------

def hhi_from_weights(W: np.ndarray) -> np.ndarray:
    """
    W: (T,N) weight matrix. Returns HHI_t = sum_i w_{i,t}^2 for each t.
    Rows are normalized onto the simplex before HHI calculation.
    Invalid rows (non-finite entries, non-positive row sum, negative weights)
    return NaN.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be (T,N)")

    finite = np.all(np.isfinite(W), axis=1)
    nonneg = np.all(W >= 0.0, axis=1)
    row_sum = np.sum(W, axis=1)
    valid = finite & nonneg & (row_sum > 0.0)

    hhi = np.full(W.shape[0], np.nan, dtype=float)
    if np.any(valid):
        Wn = W[valid] / row_sum[valid, None]
        hhi[valid] = np.sum(Wn ** 2, axis=1)
    return hhi


# ----------------------------
# Plot HHI over time
# ----------------------------

def plot_hhi_over_time(
    weights_dict: Dict[str, np.ndarray],
    title: Optional[str] = "HHI over time",
    xlabel: str = "t",
    ylabel: str = "HHI",
    figsize: Tuple[int, int] = (10, 4),
    show: bool = True
) -> plt.Figure:
    """
    weights_dict: name -> W (T,N)
    Plots HHI_t for each method with available weights.
    """
    fig = plt.figure(figsize=figsize)
    for name, W in weights_dict.items():
        hhi = hhi_from_weights(W)
        plt.plot(hhi, label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    return fig


# ----------------------------
# Plot kappa over time (kappa-bandit)
# ----------------------------

def plot_kappa_over_time(
    kappa_t: np.ndarray,
    title: Optional[str] = "Chosen $\kappa_t$ over time",
    xlabel: str = "t",
    ylabel: str = r"$\kappa_t$",
    figsize: Tuple[int, int] = (10, 3),
    show: bool = True
) -> plt.Figure:
    """
    kappa_t: (T,) chosen kappa series from your KappaBandit history.
    """
    kappa_t = np.asarray(kappa_t, dtype=float).reshape(-1)
    fig = plt.figure(figsize=figsize)
    plt.plot(kappa_t)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


# ----------------------------
# Plot action choice over time (rule-bandit)
# ----------------------------

def plot_actions_over_time(
    actions_t: np.ndarray,
    action_names: Optional[List[str]] = None,
    title: Optional[str] = "Chosen action over time",
    xlabel: str = "t",
    ylabel: str = "Action index",
    figsize: Tuple[int, int] = (10, 3),
    show: bool = True
) -> plt.Figure:
    """
    actions_t: (T,) integer action indices chosen over time.
    action_names: optional list mapping index -> name (for y-ticks).
    """
    a = np.asarray(actions_t).reshape(-1).astype(int)
    fig = plt.figure(figsize=figsize)
    plt.plot(a, linewidth=1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)

    if action_names is not None:
        uniq = np.unique(a)
        # only label seen actions to keep it readable
        plt.yticks(uniq, [action_names[i] if i < len(action_names) else str(i) for i in uniq])

    plt.tight_layout()
    if show:
        plt.show()
    return fig


# ----------------------------
# Combined convenience plot
# ----------------------------

def plot_policy_diagnostics(
    weights_dict: Optional[Dict[str, np.ndarray]] = None,
    kappa_t: Optional[np.ndarray] = None,
    actions_t: Optional[np.ndarray] = None,
    action_names: Optional[List[str]] = None
) -> None:
    """
    Convenience wrapper:
      - plots HHI for provided weights
      - plots kappa_t if provided
      - plots actions_t if provided
    """
    if weights_dict:
        plot_hhi_over_time(weights_dict, title="HHI (concentration) over time")
    if kappa_t is not None:
        plot_kappa_over_time(kappa_t)
    if actions_t is not None:
        plot_actions_over_time(actions_t, action_names=action_names)

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.evaluation import optuna_tuning as ot
from src.evaluation.m3_macro_experiment import aggregate, evaluate_series_horizon


def build_spf_state_components(
    mat: pd.DataFrame,
    vix_quarterly: pd.DataFrame,
    error_vol_window: int = 8,
) -> pd.DataFrame:
    """
    Build uncertainty components aligned to SPF origin periods:
      - vix_q_mean at origin quarter
      - cross-sectional forecast disagreement
      - recent consensus-error volatility (rolling std of lagged errors)
    """
    d = mat.copy()
    if "origin_period" not in d.columns or "actual" not in d.columns:
        raise ValueError("mat must include 'origin_period' and 'actual'.")

    fc_cols = [c for c in d.columns if c not in {"origin_period", "target_period", "actual"}]
    if not fc_cols:
        raise ValueError("mat must include forecaster columns.")

    d = d.sort_values("origin_period").reset_index(drop=True)
    F = d[fc_cols].to_numpy(dtype=float)
    disagreement = np.std(F, axis=1)
    consensus = np.mean(F, axis=1)
    errors = d["actual"].to_numpy(dtype=float) - consensus

    err_s = pd.Series(errors, dtype=float).shift(1)
    err_vol = err_s.rolling(window=int(error_vol_window), min_periods=max(3, int(error_vol_window // 2))).std()
    err_vol = err_vol.fillna(err_s.expanding(min_periods=2).std())

    vix = vix_quarterly.copy()
    if "period" not in vix.columns or "vix_q_mean" not in vix.columns:
        raise ValueError("vix_quarterly must include 'period' and 'vix_q_mean'.")
    vix["period"] = vix["period"].astype("period[Q]")
    vix_map = vix.set_index("period")["vix_q_mean"]

    out = pd.DataFrame(
        {
            "origin_period": d["origin_period"].astype("period[Q]"),
            "vix_q_mean": d["origin_period"].astype("period[Q]").map(vix_map),
            "disagreement": disagreement,
            "error_vol_recent": err_vol.to_numpy(dtype=float),
        }
    )
    return out


def pca_state_from_components(
    components: pd.DataFrame,
    feature_cols: List[str] | None = None,
    expanding: bool = True,
    min_train: int = 12,
) -> np.ndarray:
    """
    Convert uncertainty components into a 1D state via first principal component.
    If expanding=True, each t uses PCA loadings fit on rows <= t (online-safe).
    """
    cols = feature_cols if feature_cols is not None else ["vix_q_mean", "disagreement", "error_vol_recent"]
    x_raw = components[cols].astype(float).to_numpy()
    t, k = x_raw.shape
    state = np.full(t, np.nan, dtype=float)

    # Components are nonnegative uncertainty proxies; log1p stabilizes heavy tails.
    x = np.log1p(np.maximum(x_raw, 0.0))

    for i in range(t):
        train = x[: i + 1].copy() if expanding else x.copy()
        row = x[i].copy()

        # Require enough rows with at least one finite entry.
        row_ok = np.any(np.isfinite(train), axis=1)
        train = train[row_ok]
        if train.shape[0] < max(3, int(min_train)):
            continue

        # Column-wise median imputation from train only (online-safe).
        med = np.nanmedian(train, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        train = np.where(np.isfinite(train), train, med[None, :])
        row = np.where(np.isfinite(row), row, med)

        # Winsorize to reduce PCA instability from outliers.
        lo = np.quantile(train, 0.05, axis=0)
        hi = np.quantile(train, 0.95, axis=0)
        train = np.clip(train, lo, hi)
        row = np.clip(row, lo, hi)

        # Robust standardization.
        center = np.median(train, axis=0)
        scale = np.median(np.abs(train - center), axis=0)
        scale = np.where(scale > 1e-8, scale, np.std(train, axis=0))
        scale = np.where(scale > 1e-8, scale, 1.0)
        z_train = (train - center) / scale
        z_row = (row - center) / scale

        pca = PCA(n_components=1)
        pca.fit(z_train)
        comp = pca.components_[0].copy()

        # Fix sign so higher uncertainty features imply higher state.
        train_scores = z_train @ comp
        uncertainty_proxy = np.mean(z_train, axis=1)
        corr = np.corrcoef(train_scores, uncertainty_proxy)[0, 1] if train_scores.size > 1 else np.nan
        if np.isfinite(corr) and corr < 0.0:
            comp = -comp

        state[i] = float(z_row @ comp)

    if np.all(~np.isfinite(state)):
        fallback = np.nanmean(x, axis=1)
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        state = fallback

    # Fill early NaNs with first finite then forward-fill (keeps online chronology).
    finite = np.isfinite(state)
    if not np.any(finite):
        return np.ones(t, dtype=float)
    first = float(state[finite][0])
    state[~finite] = np.nan
    s = pd.Series(state, dtype=float).ffill().fillna(first).to_numpy()

    # Convert to a positive, bounded-variance scale for concentration lambda.
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med)))
    denom = mad if mad > 1e-8 else float(np.std(s))
    if denom <= 1e-8:
        return np.ones(t, dtype=float)
    z = (s - med) / denom
    z = np.clip(z, -4.0, 4.0)
    s_pos = np.exp(0.5 * z)
    s_pos = s_pos / max(np.mean(s_pos), 1e-12)
    return s_pos


def evaluate_spf_horizon_matrix(
    horizon_name: str,
    mat: pd.DataFrame,
    y: np.ndarray,
    F: np.ndarray,
    state: np.ndarray,
    loss_sections: List[str] | None = None,
    linex_a: float = 1.0,
    include_rl: bool = True,
    params_map: Dict[str, Dict[str, float]] | None = None,
    kappa_grid: np.ndarray | None = None,
) -> tuple[List[Dict[str, float]], List[Dict[str, float]], List[Dict[str, float]]]:
    """Evaluate all ensemble methods for one SPF horizon matrix."""
    losses = loss_sections if loss_sections is not None else ["mse", "linex"]
    p_map = {k: dict(v) for k, v in (params_map or ot.DEFAULT_METHOD_PARAMS).items()}
    k_grid = (
        np.asarray(kappa_grid, dtype=float)
        if kappa_grid is not None
        else np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 8.0], dtype=float)
    )

    rows: List[Dict[str, float]] = []
    diag_rows: List[Dict[str, float]] = []
    for loss_section in losses:
        r, d = evaluate_series_horizon(
            series_id=horizon_name,
            horizon=0,
            y=y,
            F=F,
            s=state,
            loss_section=loss_section,
            linex_a=float(linex_a),
            include_rl=bool(include_rl),
            params_map=p_map,
            kappa_grid=k_grid,
        )
        rows.extend(r)
        diag_rows.extend(d)

    summary_rows = aggregate(rows)
    for r in rows:
        r["horizon_name"] = horizon_name
        r["origin_start"] = str(mat["origin_period"].iloc[0])
        r["origin_end"] = str(mat["origin_period"].iloc[-1])
    for r in diag_rows:
        r["horizon_name"] = horizon_name
    for r in summary_rows:
        r["horizon_name"] = horizon_name

    return rows, diag_rows, summary_rows


def aggregate_spf_results(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Aggregate detailed SPF rows by (loss_section, method), averaging across horizons."""
    grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
    for r in rows:
        key = (str(r["loss_section"]), str(r["method"]))
        grouped.setdefault(key, []).append(r)

    out: List[Dict[str, float]] = []
    for (loss_section, method), grp in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        obj = np.array([g["objective_value"] for g in grp], dtype=float)
        mses = np.array([g["mse"] for g in grp], dtype=float)
        maes = np.array([g["mae"] for g in grp], dtype=float)
        linexes = np.array([g["linex"] for g in grp], dtype=float)
        hhis = np.array([g["avg_hhi"] for g in grp], dtype=float)
        lambdas = np.array([g.get("avg_lambda", math.nan) for g in grp], dtype=float)
        out.append(
            {
                "loss_section": loss_section,
                "method": method,
                "n_rows": float(len(grp)),
                "objective_mean": float(np.nanmean(obj)),
                "mse_mean": float(np.nanmean(mses)),
                "mae_mean": float(np.nanmean(maes)),
                "linex_mean": float(np.nanmean(linexes)),
                "avg_hhi_mean": float(np.nanmean(hhis)),
                "avg_lambda_mean": float(np.nanmean(lambdas)),
            }
        )
    return out

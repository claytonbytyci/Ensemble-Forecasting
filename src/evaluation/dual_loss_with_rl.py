from __future__ import annotations

import argparse
import ast
import csv
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.data import simulator
from src.ensemblers import ensemblers
from src.ensemblers.rl import KappaBandit, RuleSelectionBandit
from .evaluation_helpers import hhi_from_weights, linex_loss, mae, mse
from . import optuna_tuning as ot


def avg_hhi(weights: np.ndarray) -> float:
    hhi_t = np.asarray(hhi_from_weights(weights), dtype=float).reshape(-1)
    valid = np.isfinite(hhi_t)
    if not np.any(valid):
        return math.nan
    return float(np.mean(hhi_t[valid]))


def avg_finite(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    m = np.isfinite(x)
    if not np.any(m):
        return math.nan
    return float(np.mean(x[m]))


def improvement_pct_vs_best(method_obj: np.ndarray, best_obj: np.ndarray) -> np.ndarray:
    method_obj = np.asarray(method_obj, dtype=float)
    best_obj = np.asarray(best_obj, dtype=float)
    out = np.full(method_obj.shape, math.nan, dtype=float)
    valid = np.isfinite(method_obj) & np.isfinite(best_obj) & (np.abs(best_obj) > 1e-12)
    out[valid] = 100.0 * (best_obj[valid] - method_obj[valid]) / np.abs(best_obj[valid])
    return out


def is_concentration_penalized(method: str) -> bool:
    return method in {"OGDBoth", "OGDConcOnly", "MWUMBothKL", "MWUMConcOnlyKL", "RLKappaBandit"}


def regression_ols_1d(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n = int(x.size)
    if n < 2:
        return {"n": float(n), "intercept": math.nan, "slope": math.nan, "r2": math.nan, "slope_se": math.nan}
    X = np.column_stack([np.ones(n, dtype=float), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else math.nan
    slope_se = math.nan
    if n > 2:
        sigma2 = ss_res / (n - 2)
        xtx_inv = np.linalg.inv(X.T @ X)
        slope_se = float(math.sqrt(max(0.0, sigma2 * xtx_inv[1, 1])))
    return {
        "n": float(n),
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "r2": float(r2),
        "slope_se": slope_se,
    }


def align_for_horizon(
    pi: np.ndarray,
    forecasts_h: np.ndarray,
    s_unc: np.ndarray,
    h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_target = pi[h:]
    f = forecasts_h[:-h]
    s = s_unc[:-h]
    mask = np.isfinite(y_target) & np.isfinite(s) & np.all(np.isfinite(f), axis=1)
    return y_target[mask], f[mask], s[mask]


def build_context(F: np.ndarray, s: np.ndarray) -> np.ndarray:
    mean_fc = np.mean(F, axis=1)
    std_fc = np.std(F, axis=1)
    spread_fc = np.max(F, axis=1) - np.min(F, axis=1)
    return np.column_stack([np.ones(F.shape[0]), s, mean_fc, std_fc, spread_fc])


def build_models(
    params_map: Dict[str, Dict[str, float]] | None,
    loss_name: str,
    linex_a: float,
) -> Dict[str, object]:
    p = dict(ot.DEFAULT_METHOD_PARAMS)
    if params_map is not None:
        for k, v in params_map.items():
            p[k] = dict(v)
    ens_loss = "linex" if loss_name == "linex" else "squared"
    return {
        "Mean": ensemblers.MeanEnsembler(),
        "Median": ensemblers.MedianEnsembler(),
        "OGDVanilla": ensemblers.OGDVanilla(
            eta=float(p["OGDVanilla"].get("eta", 0.05)),
            loss=ens_loss,
            linex_a=linex_a,
        ),
        "MWUMVanilla": ensemblers.MWUMVanilla(
            eta=float(p["MWUMVanilla"].get("eta", 0.30)),
            loss=ens_loss,
            linex_a=linex_a,
        ),
        "OGDBoth": ensemblers.OGDConcentrationBoth(
            eta=float(p["OGDBoth"].get("eta", 0.05)),
            kappa=float(p["OGDBoth"].get("kappa", 0.80)),
            loss=ens_loss,
            linex_a=linex_a,
            state_smoothing=float(p["OGDBoth"].get("state_smoothing", ot.DEFAULT_STATE_SMOOTHING)),
            lambda_min=float(p["OGDBoth"].get("lambda_min", ot.DEFAULT_LAMBDA_MIN)),
        ),
        "OGDConcOnly": ensemblers.OGDConcentrationOnly(
            kappa=float(p["OGDConcOnly"].get("kappa", 0.80)),
            loss=ens_loss,
            linex_a=linex_a,
            state_smoothing=float(p["OGDConcOnly"].get("state_smoothing", ot.DEFAULT_STATE_SMOOTHING)),
            lambda_min=float(p["OGDConcOnly"].get("lambda_min", ot.DEFAULT_LAMBDA_MIN)),
        ),
        "MWUMBothKL": ensemblers.MWUMBothKL(
            eta=float(p["MWUMBothKL"].get("eta", 0.30)),
            kappa=float(p["MWUMBothKL"].get("kappa", 0.80)),
            loss=ens_loss,
            linex_a=linex_a,
            state_smoothing=float(p["MWUMBothKL"].get("state_smoothing", ot.DEFAULT_STATE_SMOOTHING)),
            lambda_min=float(p["MWUMBothKL"].get("lambda_min", ot.DEFAULT_LAMBDA_MIN)),
        ),
        "MWUMConcOnlyKL": ensemblers.MWUMConcentrationOnlyKL(
            kappa=float(p["MWUMConcOnlyKL"].get("kappa", 0.80)),
            loss=ens_loss,
            linex_a=linex_a,
            state_smoothing=float(p["MWUMConcOnlyKL"].get("state_smoothing", ot.DEFAULT_STATE_SMOOTHING)),
            lambda_min=float(p["MWUMConcOnlyKL"].get("lambda_min", ot.DEFAULT_LAMBDA_MIN)),
        ),
    }


def objective_fn(loss_section: str, linex_a: float) -> Callable[[np.ndarray, np.ndarray], float]:
    if loss_section == "mse":
        return mse
    return lambda y, yhat: linex_loss(y, yhat, a=linex_a)


def tuned_rule_action_specs(params_map: Dict[str, Dict[str, float]]) -> List[Dict[str, float | str]]:
    return [
        {"name": "Mean", "kind": "mean"},
        {"name": "OGDVanilla", "kind": "ogd", "eta": float(params_map["OGDVanilla"].get("eta", 0.05))},
        {"name": "MWUMVanilla", "kind": "mwum", "eta": float(params_map["MWUMVanilla"].get("eta", 0.30))},
        {
            "name": "OGDBoth",
            "kind": "ogd_both",
            "eta": float(params_map["OGDBoth"].get("eta", 0.05)),
            "kappa": float(params_map["OGDBoth"].get("kappa", 0.80)),
        },
        {"name": "OGDConcOnly", "kind": "ogd_conc", "kappa": float(params_map["OGDConcOnly"].get("kappa", 0.80))},
        {
            "name": "MWUMBothKL",
            "kind": "mwum_both",
            "eta": float(params_map["MWUMBothKL"].get("eta", 0.30)),
            "kappa": float(params_map["MWUMBothKL"].get("kappa", 0.80)),
        },
        {
            "name": "MWUMConcOnlyKL",
            "kind": "mwum_conc",
            "kappa": float(params_map["MWUMConcOnlyKL"].get("kappa", 0.80)),
        },
    ]


def load_tuned_params_from_csv(
    csv_path: Path,
    horizons: List[int],
    methods: List[str],
) -> Tuple[List[Dict[str, float]], Dict[str, Dict[int, Dict[str, Dict[str, float]]]]]:
    tuning_rows: List[Dict[str, float]] = []
    tuned_by_loss_h: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = {"mse": {}, "linex": {}}
    if not csv_path.exists():
        raise FileNotFoundError(f"Tuned params CSV not found: {csv_path}")
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            loss_section = str(row["loss_section"]).strip()
            h = int(float(row["horizon"]))
            method = str(row["method"]).strip()
            if loss_section not in tuned_by_loss_h or h not in horizons or method not in methods:
                continue
            tuned_by_loss_h[loss_section].setdefault(h, {})
            params_raw = str(row.get("best_params", "{}")).strip()
            try:
                parsed = ast.literal_eval(params_raw) if params_raw else {}
                if not isinstance(parsed, dict):
                    parsed = {}
            except (ValueError, SyntaxError):
                parsed = {}
            tuned_by_loss_h[loss_section][h][method] = {str(k): float(v) for k, v in parsed.items()}
            tuning_rows.append(
                {
                    "loss_section": loss_section,
                    "horizon": float(h),
                    "method": method,
                    "best_params": str(parsed),
                    "tuning_objective": float(row.get("tuning_objective", math.nan)),
                }
            )
    for loss_section in ["mse", "linex"]:
        for h in horizons:
            tuned_by_loss_h[loss_section].setdefault(h, {})
            for method in methods:
                tuned_by_loss_h[loss_section][h].setdefault(method, dict(ot.DEFAULT_METHOD_PARAMS.get(method, {})))
    return tuning_rows, tuned_by_loss_h


def evaluate_one_seed(
    seed: int,
    horizons: List[int],
    T: int,
    window: int,
    scenario: str,
    loss_section: str,
    linex_a: float,
    params_map_by_h: Dict[int, Dict[str, Dict[str, float]]],
    kappa_grid: np.ndarray,
) -> tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    data, forecasts_by_h, _, s_unc = simulator.make_environment_and_forecasts(
        T=T,
        horizons=horizons,
        window=window,
        seed=seed,
        include_oracle=False,
        scenario=scenario,
    )
    pi = data["pi"]
    rows: List[Dict[str, float]] = []
    diag_rows: List[Dict[str, float]] = []
    obj = objective_fn(loss_section, linex_a)

    for h in horizons:
        y, F, s = align_for_horizon(pi, forecasts_by_h[h], s_unc, h)
        x = build_context(F, s)
        indiv_obj = np.array([obj(y, F[:, i]) for i in range(F.shape[1])], dtype=float)
        best_idx = int(np.argmin(indiv_obj))
        best_obj = float(indiv_obj[best_idx])
        best_mse = float(mse(y, F[:, best_idx]))
        best_linex = float(linex_loss(y, F[:, best_idx], a=linex_a))
        tuned_params = params_map_by_h[h]

        models = build_models(tuned_params, loss_name=loss_section, linex_a=linex_a)
        for name, model in models.items():
            needs_state = name in ot.STATE_METHODS
            result = model.run(F, y, s=s if needs_state else None)
            yhat = np.asarray(result.yhat, dtype=float)
            metric_obj = float(obj(y, yhat))
            row = {
                "loss_section": loss_section,
                "seed": float(seed),
                "horizon": float(h),
                "method": name,
                "n_obs": float(y.size),
                "objective_value": metric_obj,
                "mse": float(mse(y, yhat)),
                "mae": float(mae(y, yhat)),
                "linex": float(linex_loss(y, yhat, a=linex_a)),
                "avg_hhi": float(avg_hhi(result.weights)),
                "avg_lambda": float(avg_finite(result.meta.get("lambda", np.array([math.nan])))),
                "best_individual_objective": best_obj,
                "best_individual_mse": best_mse,
                "best_individual_linex": best_linex,
                "best_individual_idx": float(best_idx),
            }
            rows.append(row)

        rl_loss = "linex" if loss_section == "linex" else "mse"

        rule_bandit = RuleSelectionBandit(
            n_forecasters=F.shape[1],
            context_dim=x.shape[1],
            alpha=0.5,
            action_specs=tuned_rule_action_specs(tuned_params),
            loss=rl_loss,
            linex_a=linex_a,
            state_smoothing=ot.DEFAULT_STATE_SMOOTHING,
            lambda_min=ot.DEFAULT_LAMBDA_MIN,
        )
        rule_res = rule_bandit.run(F=F, y=y, X=x, s=s)
        rule_actions = rule_res.meta["actions_t"].astype(int)
        valid_a = rule_actions[rule_actions >= 0]
        if valid_a.size:
            binc = np.bincount(valid_a, minlength=len(rule_bandit.action_names))
            dom_idx = int(np.argmax(binc))
            dom_name = rule_bandit.action_names[dom_idx]
            dom_share = float(binc[dom_idx] / np.sum(binc))
        else:
            dom_idx, dom_name, dom_share = -1, "NA", math.nan
        rows.append(
            {
                "loss_section": loss_section,
                "seed": float(seed),
                "horizon": float(h),
                "method": "RLRuleBandit",
                "n_obs": float(y.size),
                "objective_value": float(obj(y, rule_res.yhat)),
                "mse": float(mse(y, rule_res.yhat)),
                "mae": float(mae(y, rule_res.yhat)),
                "linex": float(linex_loss(y, rule_res.yhat, a=linex_a)),
                "avg_hhi": float(avg_hhi(rule_res.weights)),
                "avg_lambda": math.nan,
                "best_individual_objective": best_obj,
                "best_individual_mse": best_mse,
                "best_individual_linex": best_linex,
                "best_individual_idx": float(best_idx),
                "diag_main": float(dom_idx),
                "diag_aux": dom_share,
            }
        )
        for t_idx in np.where(rule_actions >= 0)[0]:
            diag_rows.append(
                {
                    "loss_section": loss_section,
                    "seed": float(seed),
                    "horizon": float(h),
                    "method": "RLRuleBandit",
                    "t": float(t_idx),
                    "action_idx": float(rule_actions[t_idx]),
                    "action_name": rule_bandit.action_names[rule_actions[t_idx]],
                    "kappa": math.nan,
                    "lambda_t": math.nan,
                    "hhi_t": float(rule_res.hhi_t[t_idx]),
                }
            )

        kappa_bandit = KappaBandit(
            kappa_grid=kappa_grid,
            context_dim=x.shape[1],
            alpha=0.5,
            loss=rl_loss,
            linex_a=linex_a,
            state_smoothing=ot.DEFAULT_STATE_SMOOTHING,
            lambda_min=ot.DEFAULT_LAMBDA_MIN,
        )
        kappa_res = kappa_bandit.run(F=F, y=y, X=x, s=s)
        kappa_t = np.asarray(kappa_res.meta["kappa_t"], dtype=float)
        high_thr = float(np.median(kappa_grid))
        rows.append(
            {
                "loss_section": loss_section,
                "seed": float(seed),
                "horizon": float(h),
                "method": "RLKappaBandit",
                "n_obs": float(y.size),
                "objective_value": float(obj(y, kappa_res.yhat)),
                "mse": float(mse(y, kappa_res.yhat)),
                "mae": float(mae(y, kappa_res.yhat)),
                "linex": float(linex_loss(y, kappa_res.yhat, a=linex_a)),
                "avg_hhi": float(avg_hhi(kappa_res.weights)),
                "avg_lambda": float(avg_finite(kappa_res.meta.get("lambda_t", np.array([math.nan])))),
                "best_individual_objective": best_obj,
                "best_individual_mse": best_mse,
                "best_individual_linex": best_linex,
                "best_individual_idx": float(best_idx),
                "diag_main": float(avg_finite(kappa_t)),
                "diag_aux": float(np.mean(kappa_t[np.isfinite(kappa_t)] >= high_thr)) if np.any(np.isfinite(kappa_t)) else math.nan,
            }
        )
        kappa_actions = np.asarray(kappa_res.meta["actions_t"], dtype=int)
        lambda_t = np.asarray(kappa_res.meta["lambda_t"], dtype=float)
        for t_idx in np.where(kappa_actions >= 0)[0]:
            diag_rows.append(
                {
                    "loss_section": loss_section,
                    "seed": float(seed),
                    "horizon": float(h),
                    "method": "RLKappaBandit",
                    "t": float(t_idx),
                    "action_idx": float(kappa_actions[t_idx]),
                    "action_name": "kappa_idx",
                    "kappa": float(kappa_t[t_idx]),
                    "lambda_t": float(lambda_t[t_idx]),
                    "hhi_t": float(kappa_res.hhi_t[t_idx]),
                }
            )

    return rows, diag_rows


def aggregate(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[str, float, str], List[Dict[str, float]]] = {}
    for r in rows:
        key = (str(r["loss_section"]), float(r["horizon"]), str(r["method"]))
        grouped.setdefault(key, []).append(r)

    out: List[Dict[str, float]] = []
    for (loss_section, horizon, method), grp in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        obj = np.array([g["objective_value"] for g in grp], dtype=float)
        mses = np.array([g["mse"] for g in grp], dtype=float)
        maes = np.array([g["mae"] for g in grp], dtype=float)
        linexes = np.array([g["linex"] for g in grp], dtype=float)
        hhis = np.array([g["avg_hhi"] for g in grp], dtype=float)
        lambdas = np.array([g.get("avg_lambda", math.nan) for g in grp], dtype=float)
        bests = np.array([g["best_individual_objective"] for g in grp], dtype=float)
        improvement_pct = improvement_pct_vs_best(obj, bests)
        diag_main = np.array([g.get("diag_main", math.nan) for g in grp], dtype=float)
        diag_aux = np.array([g.get("diag_aux", math.nan) for g in grp], dtype=float)
        out.append(
            {
                "loss_section": loss_section,
                "horizon": horizon,
                "method": method,
                "objective_mean": float(np.mean(obj)),
                "objective_std": float(np.std(obj)),
                "mse_mean": float(np.mean(mses)),
                "mae_mean": float(np.mean(maes)),
                "linex_mean": float(np.mean(linexes)),
                "avg_hhi_mean": float(np.mean(hhis[np.isfinite(hhis)])) if np.any(np.isfinite(hhis)) else math.nan,
                "avg_lambda_mean": float(np.mean(lambdas[np.isfinite(lambdas)])) if np.any(np.isfinite(lambdas)) else math.nan,
                "avg_excess_objective_vs_best_individual": float(np.mean(obj - bests)),
                "avg_improvement_pct_vs_best_individual": float(np.mean(improvement_pct[np.isfinite(improvement_pct)]))
                if np.any(np.isfinite(improvement_pct))
                else math.nan,
                "diag_main_mean": float(np.mean(diag_main[np.isfinite(diag_main)])) if np.any(np.isfinite(diag_main)) else math.nan,
                "diag_aux_mean": float(np.mean(diag_aux[np.isfinite(diag_aux)])) if np.any(np.isfinite(diag_aux)) else math.nan,
            }
        )
    return out


def plot_improvement_vs_hhi(summary_rows: List[Dict[str, float]], out_dir: Path, output_stem: str) -> List[Path]:
    out_paths: List[Path] = []
    groups: Dict[Tuple[str, int], List[Dict[str, float]]] = {}
    for r in summary_rows:
        key = (str(r["loss_section"]), int(float(r["horizon"])))
        groups.setdefault(key, []).append(r)

    for (loss_section, horizon), grp in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        plot_rows = [
            r
            for r in grp
            if np.isfinite(float(r.get("avg_hhi_mean", math.nan)))
            and np.isfinite(float(r.get("avg_improvement_pct_vs_best_individual", math.nan)))
        ]
        if not plot_rows:
            continue

        x = np.array([float(r["avg_hhi_mean"]) for r in plot_rows], dtype=float)
        y = np.array([float(r["avg_improvement_pct_vs_best_individual"]) for r in plot_rows], dtype=float)
        methods = [str(r["method"]) for r in plot_rows]
        reg = regression_ols_1d(x, y)

        fig, ax = plt.subplots(figsize=(10.5, 6.2))
        colors = plt.get_cmap("tab10", len(plot_rows))
        for i, (xi, yi, m) in enumerate(zip(x, y, methods)):
            marker = "x" if is_concentration_penalized(m) else "o"
            ax.scatter(xi, yi, s=80, marker=marker, color=colors(i), linewidths=1.5)
            ax.annotate(m, (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=8)

        if np.isfinite(reg["slope"]) and np.isfinite(reg["intercept"]):
            xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            yy = reg["intercept"] + reg["slope"] * xx
            ax.plot(xx, yy, color="black", linestyle="--", linewidth=1.25, alpha=0.85)

        reg_text = (
            f"OLS: improvement% = a + b*HHI\n"
            f"a={reg['intercept']:.3f}, b={reg['slope']:.3f}\n"
            f"SE(b)={reg['slope_se']:.3f}, R^2={reg['r2']:.3f}, n={int(reg['n'])}"
        )
        ax.text(
            0.98,
            0.98,
            reg_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#333333"},
        )

        ax.set_title(f"Method Improvement vs Average HHI ({loss_section.upper()}, h={horizon})")
        ax.set_xlabel("Average HHI")
        ax.set_ylabel("Average Improvement vs Best Individual (%)")
        ax.grid(alpha=0.25)
        fig.tight_layout()

        out_path = out_dir / f"{output_stem}_improvement_vs_hhi_{loss_section}_h{horizon}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def plot_individual_improvement_vs_hhi(detailed_rows: List[Dict[str, float]], out_dir: Path, output_stem: str) -> Path | None:
    groups: Dict[Tuple[str, int], List[Dict[str, float]]] = {}
    for r in detailed_rows:
        key = (str(r["loss_section"]), int(float(r["horizon"])))
        groups.setdefault(key, []).append(r)
    if not groups:
        return None

    ordered_groups = sorted(groups.items(), key=lambda x: (x[0][0], x[0][1]))
    methods = sorted({str(r["method"]) for r in detailed_rows})
    color_map = {m: plt.get_cmap("tab20")(i % 20) for i, m in enumerate(methods)}
    marker_map = {m: ("x" if is_concentration_penalized(m) else "o") for m in methods}

    n = len(ordered_groups)
    cols = min(3, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.2 * rows), squeeze=False)
    flat_axes = axes.ravel()
    legend_handles: Dict[str, object] = {}

    for ax, ((loss_section, horizon), grp) in zip(flat_axes, ordered_groups):
        for r in grp:
            hhi = float(r.get("avg_hhi", math.nan))
            method_obj = float(r.get("objective_value", math.nan))
            best_obj = float(r.get("best_individual_objective", math.nan))
            if not (np.isfinite(hhi) and np.isfinite(method_obj) and np.isfinite(best_obj) and abs(best_obj) > 1e-12):
                continue
            improvement = 100.0 * (best_obj - method_obj) / abs(best_obj)
            method = str(r["method"])
            sc = ax.scatter(
                hhi,
                improvement,
                s=28,
                alpha=0.65,
                color=color_map[method],
                marker=marker_map[method],
                linewidths=1.0,
            )
            if method not in legend_handles:
                legend_handles[method] = sc
        ax.set_title(f"{loss_section.upper()} | h={horizon}")
        ax.set_xlabel("Average HHI (per simulation)")
        ax.set_ylabel("Improvement vs Best Individual (%)")
        ax.grid(alpha=0.25)

    for ax in flat_axes[n:]:
        ax.axis("off")

    if legend_handles:
        ordered = [m for m in methods if m in legend_handles]
        fig.legend(
            [legend_handles[m] for m in ordered],
            ordered,
            loc="upper center",
            ncol=min(6, len(ordered)),
            fontsize=8,
            frameon=True,
            bbox_to_anchor=(0.5, 1.01),
        )
    fig.suptitle("Simulation-Level Improvement vs HHI (all methods, all simulations)", y=1.04)
    fig.tight_layout()
    out_path = out_dir / f"{output_stem}_individual_improvement_vs_hhi.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_csv(path: Path, rows: Iterable[Dict[str, float]]) -> None:
    rows_list = list(rows)
    if not rows_list:
        return
    fieldnames = sorted({k for r in rows_list for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_list:
            w.writerow(r)


def write_report(path: Path, summary_rows: List[Dict[str, float]], tuning_rows: List[Dict[str, float]], linex_a: float) -> None:
    lines: List[str] = []
    lines.append("# Dual-Loss Ensemble + RL Analysis")
    lines.append("")
    lines.append(f"- LINEX parameter `a`: {linex_a}")
    lines.append("")

    for loss_section in ["mse", "linex"]:
        lines.append(f"## {loss_section.upper()} Results")
        lines.append("")
        sub_tune = [r for r in tuning_rows if r["loss_section"] == loss_section]
        if sub_tune:
            lines.append("### Tuned Hyperparameters (Regular Methods)")
            for h in sorted({int(float(r["horizon"])) for r in sub_tune}):
                lines.append(f"- Horizon {h}:")
                hs = [r for r in sub_tune if int(float(r["horizon"])) == h]
                hs = sorted(hs, key=lambda x: float(x["tuning_objective"]))
                for r in hs:
                    lines.append(f"  - {r['method']}: params={r['best_params']} objective={r['tuning_objective']}")
            lines.append("")

        lines.append("### Summary (Lower Objective Is Better)")
        sub = [r for r in summary_rows if r["loss_section"] == loss_section]
        for h in sorted({int(float(r["horizon"])) for r in sub}):
            lines.append(f"- Horizon {h}:")
            hs = [r for r in sub if int(float(r["horizon"])) == h]
            hs = sorted(hs, key=lambda x: float(x["objective_mean"]))
            for r in hs:
                lines.append(
                    "  - "
                    f"{r['method']}: objective={r['objective_mean']:.4f} (std {r['objective_std']:.4f}), "
                    f"MSE={r['mse_mean']:.4f}, MAE={r['mae_mean']:.4f}, LINEX={r['linex_mean']:.4f}, "
                    f"avg HHI={r['avg_hhi_mean']:.4f}, excess vs best indiv={r['avg_excess_objective_vs_best_individual']:.4f}, "
                    f"improvement vs best indiv={r['avg_improvement_pct_vs_best_individual']:.2f}%"
                )
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dual-loss (MSE + LINEX) ensemble and RL comparison.")
    parser.add_argument("--scenario", type=str, default="discriminating")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--T", type=int, default=1400)
    parser.add_argument("--window", type=int, default=150)
    parser.add_argument("--tuning-seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[4, 5, 6, 7, 8, 9])
    parser.add_argument("--n-test-sims", type=int, default=0, help="If >0, override --test-seeds with a contiguous range.")
    parser.add_argument("--test-seed-start", type=int, default=4)
    parser.add_argument("--n-trials", type=int, default=35)
    parser.add_argument("--linex-a", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default="analyses/results")
    parser.add_argument("--output-stem", type=str, default="dual_loss_full", help="Filename stem for outputs.")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip Optuna tuning and use CSV/default params.")
    parser.add_argument("--tuned-params-csv", type=str, default="", help="Optional tuned parameter CSV to reuse.")
    args = parser.parse_args()

    loss_sections = [("mse", "squared", "mse"), ("linex", "linex", "linex")]
    methods = list(ot.DEFAULT_METHOD_PARAMS.keys())
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.n_test_sims > 0:
        test_seeds = list(range(args.test_seed_start, args.test_seed_start + args.n_test_sims))
    else:
        test_seeds = list(args.test_seeds)

    tuning_rows: List[Dict[str, float]] = []
    tuned_by_loss_h: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = {}

    if args.skip_tuning:
        csv_path = Path(args.tuned_params_csv).resolve() if args.tuned_params_csv else out_dir / f"{args.output_stem}_tuned_params.csv"
        tuning_rows, tuned_by_loss_h = load_tuned_params_from_csv(csv_path=csv_path, horizons=list(args.horizons), methods=methods)
        print(f"Using tuned params from: {csv_path}")
    else:
        for loss_section, ens_loss, objective_metric in loss_sections:
            tuned_by_h: Dict[int, Dict[str, Dict[str, float]]] = {}
            for h in args.horizons:
                slices_h = []
                for seed in args.tuning_seeds:
                    data_t, forecasts_t, _, s_unc_t = simulator.make_environment_and_forecasts(
                        T=args.T,
                        horizons=args.horizons,
                        window=args.window,
                        seed=seed,
                        include_oracle=False,
                        scenario=args.scenario,
                    )
                    y_t, F_t, s_t = align_for_horizon(data_t["pi"], forecasts_t[h], s_unc_t, h)
                    slices_h.append((y_t, F_t, s_t))
                try:
                    tuned_h = ot.tune_all_methods_optuna(
                        data_slices=slices_h,
                        methods=methods,
                        n_trials=args.n_trials,
                        seed=42 + 100 * h,
                        loss=ens_loss,
                        linex_a=args.linex_a,
                        objective_metric=objective_metric,
                    )
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        "Optuna is required for this script. Install dependencies, or rerun with --skip-tuning."
                    ) from exc
                tuned_by_h[h] = {m: dict(tuned_h[m].best_params) for m in methods}
                for m in methods:
                    tuning_rows.append(
                        {
                            "loss_section": loss_section,
                            "horizon": float(h),
                            "method": m,
                            "best_params": str(tuned_h[m].best_params),
                            "tuning_objective": float(tuned_h[m].best_value),
                        }
                    )
            tuned_by_loss_h[loss_section] = tuned_by_h

    all_rows: List[Dict[str, float]] = []
    all_diag_rows: List[Dict[str, float]] = []
    kappa_grid = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 8.0], dtype=float)

    for loss_section, _, _ in loss_sections:
        for seed in test_seeds:
            rows, diag_rows = evaluate_one_seed(
                seed=seed,
                horizons=args.horizons,
                T=args.T,
                window=args.window,
                scenario=args.scenario,
                loss_section=loss_section,
                linex_a=args.linex_a,
                params_map_by_h=tuned_by_loss_h[loss_section],
                kappa_grid=kappa_grid,
            )
            all_rows.extend(rows)
            all_diag_rows.extend(diag_rows)

    summary_rows = aggregate(all_rows)

    detailed_csv = out_dir / f"{args.output_stem}_detailed.csv"
    summary_csv = out_dir / f"{args.output_stem}_summary.csv"
    tuning_csv = out_dir / f"{args.output_stem}_tuned_params.csv"
    diag_csv = out_dir / f"{args.output_stem}_policy_diagnostics.csv"
    report_md = out_dir / f"{args.output_stem}_report.md"
    plot_paths = plot_improvement_vs_hhi(summary_rows, out_dir, output_stem=args.output_stem)
    indiv_plot = plot_individual_improvement_vs_hhi(all_rows, out_dir, output_stem=args.output_stem)

    write_csv(detailed_csv, all_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(tuning_csv, tuning_rows)
    write_csv(diag_csv, all_diag_rows)
    write_report(report_md, summary_rows, tuning_rows, linex_a=args.linex_a)

    print(f"Wrote: {detailed_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {tuning_csv}")
    print(f"Wrote: {diag_csv}")
    print(f"Wrote: {report_md}")
    for p in plot_paths:
        print(f"Wrote: {p}")
    if indiv_plot is not None:
        print(f"Wrote: {indiv_plot}")


if __name__ == "__main__":
    main()

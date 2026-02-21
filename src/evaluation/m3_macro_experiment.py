from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from src.data.cleaning import (
    align_m3_monthly_actuals_and_forecasts,
    build_m3_series_horizon_matrix,
    prepare_m3_monthly_data,
)
from src.data.loading import (
    load_or_download_m3_quarterly_actuals,
    load_or_download_m3_quarterly_forecasts,
)
from src.ensemblers import ensemblers
from src.ensemblers.rl import KappaBandit, RuleSelectionBandit
from src.evaluation import optuna_tuning as ot
from src.evaluation.evaluation_helpers import hhi_from_weights, linex_loss, mae, mse


META_COLS = {"origin_period", "target_period", "actual"}


def avg_hhi(weights: np.ndarray) -> float:
    hhi_t = np.asarray(hhi_from_weights(weights), dtype=float).reshape(-1)
    m = np.isfinite(hhi_t)
    if not np.any(m):
        return math.nan
    return float(np.mean(hhi_t[m]))


def avg_finite(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    m = np.isfinite(x)
    if not np.any(m):
        return math.nan
    return float(np.mean(x[m]))


def objective_fn(loss_section: str, linex_a: float):
    if loss_section == "mse":
        return mse
    if loss_section == "linex":
        return lambda y, yhat: linex_loss(y, yhat, a=linex_a)
    raise ValueError("loss_section must be 'mse' or 'linex'")


def build_context(F: np.ndarray, s: np.ndarray) -> np.ndarray:
    mean_fc = np.mean(F, axis=1)
    std_fc = np.std(F, axis=1)
    spread_fc = np.max(F, axis=1) - np.min(F, axis=1)
    return np.column_stack([np.ones(F.shape[0]), s, mean_fc, std_fc, spread_fc])


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


def build_models(
    params_map: Dict[str, Dict[str, float]],
    loss_name: str,
    linex_a: float,
) -> Dict[str, object]:
    ens_loss = "linex" if loss_name == "linex" else "squared"
    p = dict(params_map)
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


def _series_h_data(
    aligned_long,
    series_id: str,
    horizon: int,
    min_obs: int,
    min_forecasters: int,
    max_forecasters: int = 12,
):
    # Backward-compatible wrapper. Prefer _series_h_data_fixed_methods below.
    method_cols = sorted(
        aligned_long.loc[aligned_long["series_id"] == str(series_id), "method_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    return _series_h_data_fixed_methods(
        aligned_long=aligned_long,
        series_id=series_id,
        horizon=horizon,
        required_methods=method_cols,
        min_obs=min_obs,
        min_methods=min_forecasters,
    )


def _series_h_data_fixed_methods(
    aligned_long,
    series_id: str,
    horizon: int,
    required_methods: List[str],
    min_obs: int,
    min_methods: int = 3,
):
    mat = build_m3_series_horizon_matrix(
        aligned_df=aligned_long,
        series_id=series_id,
        horizon=horizon,
        require_actual=True,
    )
    method_cols = [c for c in mat.columns if c not in META_COLS]
    if not method_cols:
        return None

    # Use the requested method set directly (typically derived from M809).
    keep_methods = [m for m in required_methods if m in method_cols]
    if len(keep_methods) < int(min_methods):
        return None

    # Complete cases across the fixed method set (no look-ahead safe panel).
    mat2 = mat[["origin_period", "target_period", "actual", *keep_methods]].dropna().copy()
    if len(mat2) < int(min_obs):
        return None

    y = mat2["actual"].to_numpy(dtype=float)
    F = mat2[keep_methods].to_numpy(dtype=float)
    # State proxy from contemporaneous cross-method dispersion (no lookahead).
    s = np.std(F, axis=1)
    s = np.maximum(s, 1e-12)
    s = s / np.mean(s)

    return {
        "mat": mat2,
        "y": y,
        "F": F,
        "methods": keep_methods,
        "s": s,
    }


def series_h_data_fixed_methods(
    aligned_long,
    series_id: str,
    horizon: int,
    required_methods: List[str],
    min_obs: int,
    min_methods: int = 3,
):
    """Public wrapper for fixed-method panel construction."""
    return _series_h_data_fixed_methods(
        aligned_long=aligned_long,
        series_id=series_id,
        horizon=horizon,
        required_methods=required_methods,
        min_obs=min_obs,
        min_methods=min_methods,
    )


def series_online_data_fixed_methods(
    aligned_long,
    series_id: str,
    required_methods: List[str] | None = None,
    min_obs: int = 6,
    min_methods: int = 3,
):
    """
    Build a sequential panel for one series using horizons as the online time axis.

    This matches the M3 monthly competition layout where each series typically has a
    single forecast origin and horizons 1..18. The returned panel is one row per
    horizon with columns:
      horizon, target_period, actual, <method forecasts...>
    """
    d = aligned_long.loc[aligned_long["series_id"] == str(series_id)].copy()
    if d.empty:
        return None

    if "horizon_consistent" in d.columns:
        d = d.loc[d["horizon_consistent"]].copy()
    d = d.loc[d["actual"].notna()].copy()
    if d.empty:
        return None

    d["horizon"] = d["horizon"].astype(int)

    # Keep a single origin for coherent horizon-wise sequential evaluation.
    origin_counts = d.groupby("origin_period", as_index=False).size().sort_values("size", ascending=False)
    if origin_counts.empty:
        return None
    best_origin = origin_counts.iloc[0]["origin_period"]
    d = d.loc[d["origin_period"] == best_origin].copy()
    if d.empty:
        return None

    mat_fc = d.pivot_table(index="horizon", columns="method_id", values="forecast", aggfunc="last")
    mat_meta = (
        d.groupby("horizon", as_index=True)
        .agg(actual=("actual", "last"), target_period=("target_period", "last"))
        .sort_index()
    )
    mat = mat_meta.join(mat_fc, how="left").reset_index()

    method_cols = [c for c in mat.columns if c not in {"horizon", "target_period", "actual"}]
    if not method_cols:
        return None

    if required_methods is None:
        keep_methods = sorted(method_cols)
    else:
        keep_methods = [m for m in required_methods if m in method_cols]
    if len(keep_methods) < int(min_methods):
        return None

    mat2 = mat[["horizon", "target_period", "actual", *keep_methods]].dropna().copy()
    if len(mat2) < int(min_obs):
        return None

    y = mat2["actual"].to_numpy(dtype=float)
    F = mat2[keep_methods].to_numpy(dtype=float)
    s = np.std(F, axis=1)
    s = np.maximum(s, 1e-12)
    s = s / np.mean(s)

    return {
        "mat": mat2,
        "y": y,
        "F": F,
        "methods": keep_methods,
        "s": s,
        "origin_period": best_origin,
    }


def evaluate_series_horizon(
    series_id: str,
    horizon: int,
    y: np.ndarray,
    F: np.ndarray,
    s: np.ndarray,
    loss_section: str,
    linex_a: float,
    include_rl: bool,
    params_map: Dict[str, Dict[str, float]],
    kappa_grid: np.ndarray,
) -> tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    rows: List[Dict[str, float]] = []
    diag_rows: List[Dict[str, float]] = []

    obj = objective_fn(loss_section, linex_a)
    indiv_obj = np.array([obj(y, F[:, i]) for i in range(F.shape[1])], dtype=float)
    best_idx = int(np.argmin(indiv_obj))
    best_obj = float(indiv_obj[best_idx])
    best_mse = float(mse(y, F[:, best_idx]))
    best_linex = float(linex_loss(y, F[:, best_idx], a=linex_a))

    models = build_models(params_map=params_map, loss_name=loss_section, linex_a=linex_a)
    for name, model in models.items():
        needs_state = name in ot.STATE_METHODS
        result = model.run(F, y, s=s if needs_state else None)
        yhat = np.asarray(result.yhat, dtype=float)
        rows.append(
            {
                "series_id": series_id,
                "horizon": float(horizon),
                "loss_section": loss_section,
                "method": name,
                "n_obs": float(y.size),
                "n_forecasters": float(F.shape[1]),
                "objective_value": float(obj(y, yhat)),
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
        )

    if include_rl:
        X = build_context(F, s)
        rl_loss = "linex" if loss_section == "linex" else "mse"

        rb = RuleSelectionBandit(
            n_forecasters=F.shape[1],
            context_dim=X.shape[1],
            alpha=0.5,
            action_specs=tuned_rule_action_specs(params_map),
            loss=rl_loss,
            linex_a=linex_a,
            state_smoothing=ot.DEFAULT_STATE_SMOOTHING,
            lambda_min=ot.DEFAULT_LAMBDA_MIN,
        )
        rb_res = rb.run(F=F, y=y, X=X, s=s)
        actions = rb_res.meta["actions_t"].astype(int)
        valid_a = actions[actions >= 0]
        if valid_a.size:
            binc = np.bincount(valid_a, minlength=len(rb.action_names))
            dom_idx = int(np.argmax(binc))
            dom_share = float(binc[dom_idx] / np.sum(binc))
        else:
            dom_idx, dom_share = -1, math.nan

        rows.append(
            {
                "series_id": series_id,
                "horizon": float(horizon),
                "loss_section": loss_section,
                "method": "RLRuleBandit",
                "n_obs": float(y.size),
                "n_forecasters": float(F.shape[1]),
                "objective_value": float(obj(y, rb_res.yhat)),
                "mse": float(mse(y, rb_res.yhat)),
                "mae": float(mae(y, rb_res.yhat)),
                "linex": float(linex_loss(y, rb_res.yhat, a=linex_a)),
                "avg_hhi": float(avg_hhi(rb_res.weights)),
                "avg_lambda": math.nan,
                "best_individual_objective": best_obj,
                "best_individual_mse": best_mse,
                "best_individual_linex": best_linex,
                "best_individual_idx": float(best_idx),
                "diag_main": float(dom_idx),
                "diag_aux": dom_share,
            }
        )

        for t_idx in np.where(actions >= 0)[0]:
            diag_rows.append(
                {
                    "series_id": series_id,
                    "horizon": float(horizon),
                    "loss_section": loss_section,
                    "method": "RLRuleBandit",
                    "t": float(t_idx),
                    "action_idx": float(actions[t_idx]),
                    "action_name": rb.action_names[actions[t_idx]],
                    "kappa": math.nan,
                    "lambda_t": math.nan,
                    "hhi_t": float(rb_res.hhi_t[t_idx]),
                }
            )

        kb = KappaBandit(
            kappa_grid=kappa_grid,
            context_dim=X.shape[1],
            alpha=0.5,
            loss=rl_loss,
            linex_a=linex_a,
            state_smoothing=ot.DEFAULT_STATE_SMOOTHING,
            lambda_min=ot.DEFAULT_LAMBDA_MIN,
        )
        kb_res = kb.run(F=F, y=y, X=X, s=s)
        kappa_t = np.asarray(kb_res.meta["kappa_t"], dtype=float)
        high_thr = float(np.median(kappa_grid))

        rows.append(
            {
                "series_id": series_id,
                "horizon": float(horizon),
                "loss_section": loss_section,
                "method": "RLKappaBandit",
                "n_obs": float(y.size),
                "n_forecasters": float(F.shape[1]),
                "objective_value": float(obj(y, kb_res.yhat)),
                "mse": float(mse(y, kb_res.yhat)),
                "mae": float(mae(y, kb_res.yhat)),
                "linex": float(linex_loss(y, kb_res.yhat, a=linex_a)),
                "avg_hhi": float(avg_hhi(kb_res.weights)),
                "avg_lambda": float(avg_finite(kb_res.meta.get("lambda_t", np.array([math.nan])))),
                "best_individual_objective": best_obj,
                "best_individual_mse": best_mse,
                "best_individual_linex": best_linex,
                "best_individual_idx": float(best_idx),
                "diag_main": float(avg_finite(kappa_t)),
                "diag_aux": float(np.mean(kappa_t[np.isfinite(kappa_t)] >= high_thr)) if np.any(np.isfinite(kappa_t)) else math.nan,
            }
        )

        actions_k = np.asarray(kb_res.meta["actions_t"], dtype=int)
        lambda_t = np.asarray(kb_res.meta["lambda_t"], dtype=float)
        for t_idx in np.where(actions_k >= 0)[0]:
            diag_rows.append(
                {
                    "series_id": series_id,
                    "horizon": float(horizon),
                    "loss_section": loss_section,
                    "method": "RLKappaBandit",
                    "t": float(t_idx),
                    "action_idx": float(actions_k[t_idx]),
                    "action_name": "kappa_idx",
                    "kappa": float(kappa_t[t_idx]),
                    "lambda_t": float(lambda_t[t_idx]),
                    "hhi_t": float(kb_res.hhi_t[t_idx]),
                }
            )

    return rows, diag_rows


def aggregate(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[tuple[str, float, str], List[Dict[str, float]]] = {}
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
        improvement_pct = np.full(obj.shape, np.nan)
        m = np.isfinite(obj) & np.isfinite(bests) & (np.abs(bests) > 1e-12)
        improvement_pct[m] = 100.0 * (bests[m] - obj[m]) / np.abs(bests[m])

        out.append(
            {
                "loss_section": loss_section,
                "horizon": horizon,
                "method": method,
                "n_series": float(len(grp)),
                "objective_mean": float(np.mean(obj)),
                "objective_std": float(np.std(obj)),
                "mse_mean": float(np.mean(mses)),
                "mae_mean": float(np.mean(maes)),
                "linex_mean": float(np.mean(linexes)),
                "avg_hhi_mean": float(np.mean(hhis[np.isfinite(hhis)])) if np.any(np.isfinite(hhis)) else math.nan,
                "avg_lambda_mean": float(np.mean(lambdas[np.isfinite(lambdas)])) if np.any(np.isfinite(lambdas)) else math.nan,
                "avg_excess_objective_vs_best_individual": float(np.mean(obj - bests)),
                "avg_improvement_pct_vs_best_individual": float(np.mean(improvement_pct[np.isfinite(improvement_pct)])) if np.any(np.isfinite(improvement_pct)) else math.nan,
            }
        )
    return out


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


def write_report(path: Path, summary_rows: List[Dict[str, float]], detailed_rows: List[Dict[str, float]], linex_a: float) -> None:
    lines: List[str] = []
    lines.append("# M3 Monthly Macro Ensemble Experiment")
    lines.append("")
    lines.append(f"- LINEX parameter `a`: {linex_a}")
    lines.append(f"- Detailed rows: {len(detailed_rows)}")
    lines.append("")

    for loss_section in ["mse", "linex"]:
        lines.append(f"## {loss_section.upper()} Summary")
        lines.append("")
        sub = [r for r in summary_rows if r["loss_section"] == loss_section]
        if not sub:
            lines.append("No rows.")
            lines.append("")
            continue
        for h in sorted({int(float(r["horizon"])) for r in sub}):
            lines.append(f"### Horizon {h}")
            hs = sorted([r for r in sub if int(float(r["horizon"])) == h], key=lambda x: float(x["objective_mean"]))
            for r in hs:
                lines.append(
                    "- "
                    f"{r['method']}: objective={r['objective_mean']:.6f} (std {r['objective_std']:.6f}), "
                    f"MSE={r['mse_mean']:.6f}, MAE={r['mae_mean']:.6f}, LINEX={r['linex_mean']:.6f}, "
                    f"avg HHI={r['avg_hhi_mean']:.6f}, improvement vs best indiv={r['avg_improvement_pct_vs_best_individual']:.3f}% "
                    f"over {int(float(r['n_series']))} series"
                )
            lines.append("")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M3 monthly macro full ensembling experiment across macro series.")
    parser.add_argument("--out-dir", type=str, default="analyses/results/m3_macro")
    parser.add_argument("--output-stem", type=str, default="m3_macro_full")
    parser.add_argument("--linex-a", type=float, default=1.0)
    parser.add_argument("--loss-sections", type=str, nargs="+", default=["mse", "linex"])
    parser.add_argument("--horizons", type=int, nargs="+", default=[])
    parser.add_argument("--min-obs", type=int, default=24)
    parser.add_argument("--min-forecasters", type=int, default=3)
    parser.add_argument("--max-series", type=int, default=0, help="For debugging only; 0 means all macro series.")
    parser.add_argument("--skip-rl", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    actuals_raw = load_or_download_m3_quarterly_actuals()
    forecasts_raw = load_or_download_m3_quarterly_forecasts()

    actuals_macro, forecasts_macro, macro_series_ids = prepare_m3_monthly_data(
        actuals_raw,
        forecasts_raw,
        category="MACRO",
    )
    aligned_long = align_m3_monthly_actuals_and_forecasts(actuals_macro, forecasts_macro)

    all_horizons = sorted(aligned_long["horizon"].dropna().astype(int).unique().tolist())
    horizons = sorted(set(args.horizons)) if args.horizons else all_horizons

    series_ids = list(map(str, macro_series_ids.tolist()))
    if args.max_series > 0:
        series_ids = series_ids[: args.max_series]

    params_map = {k: dict(v) for k, v in ot.DEFAULT_METHOD_PARAMS.items()}
    kappa_grid = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 8.0], dtype=float)

    detailed_rows: List[Dict[str, float]] = []
    diag_rows: List[Dict[str, float]] = []

    for loss_section in args.loss_sections:
        for sid in series_ids:
            for h in horizons:
                d = _series_h_data(
                    aligned_long=aligned_long,
                    series_id=sid,
                    horizon=h,
                    min_obs=int(args.min_obs),
                    min_forecasters=int(args.min_forecasters),
                )
                if d is None:
                    continue
                rows_h, diag_h = evaluate_series_horizon(
                    series_id=sid,
                    horizon=int(h),
                    y=d["y"],
                    F=d["F"],
                    s=d["s"],
                    loss_section=loss_section,
                    linex_a=float(args.linex_a),
                    include_rl=not args.skip_rl,
                    params_map=params_map,
                    kappa_grid=kappa_grid,
                )
                detailed_rows.extend(rows_h)
                diag_rows.extend(diag_h)

    summary_rows = aggregate(detailed_rows)

    detailed_csv = out_dir / f"{args.output_stem}_detailed.csv"
    summary_csv = out_dir / f"{args.output_stem}_summary.csv"
    diag_csv = out_dir / f"{args.output_stem}_policy_diagnostics.csv"
    report_md = out_dir / f"{args.output_stem}_report.md"

    write_csv(detailed_csv, detailed_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(diag_csv, diag_rows)
    write_report(report_md, summary_rows, detailed_rows, linex_a=float(args.linex_a))

    print(f"Wrote: {detailed_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {diag_csv}")
    print(f"Wrote: {report_md}")
    print(f"Macro series evaluated: {len(set(r['series_id'] for r in detailed_rows)) if detailed_rows else 0}")


if __name__ == "__main__":
    main()

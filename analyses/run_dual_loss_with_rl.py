from __future__ import annotations

import argparse
import ast
import csv
import math
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from src.data import simulator
from src.ensemblers import ensemblers
from src.ensemblers.rl import KappaBandit, RuleSelectionBandit, SoftmaxSimplexBandit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "src" / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
import optuna_tuning as ot


def mse(y: np.ndarray, yhat: np.ndarray) -> float:
    e = y - yhat
    return float(np.mean(e * e))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def linex(y: np.ndarray, yhat: np.ndarray, a: float = 1.0) -> float:
    e = y - yhat
    return float(np.mean(np.exp(a * e) - a * e - 1.0))


def avg_hhi(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2:
        return math.nan
    valid = np.all(np.isfinite(w), axis=1)
    if not np.any(valid):
        return math.nan
    return float(np.mean(np.sum(w[valid] ** 2, axis=1)))


def avg_finite(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    m = np.isfinite(x)
    if not np.any(m):
        return math.nan
    return float(np.mean(x[m]))


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
        ),
        "OGDConcOnly": ensemblers.OGDConcentrationOnly(
            kappa=float(p["OGDConcOnly"].get("kappa", 0.80)),
            loss=ens_loss,
            linex_a=linex_a,
        ),
        "MWUMBothKL": ensemblers.MWUMBothKL(
            eta=float(p["MWUMBothKL"].get("eta", 0.30)),
            kappa=float(p["MWUMBothKL"].get("kappa", 0.80)),
            loss=ens_loss,
            linex_a=linex_a,
        ),
        "MWUMConcOnlyKL": ensemblers.MWUMConcentrationOnlyKL(
            kappa=float(p["MWUMConcOnlyKL"].get("kappa", 0.80)),
            loss=ens_loss,
            linex_a=linex_a,
        ),
    }


def objective_fn(loss_section: str, linex_a: float) -> Callable[[np.ndarray, np.ndarray], float]:
    if loss_section == "mse":
        return mse
    return lambda y, yhat: linex(y, yhat, a=linex_a)


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
    simplex_reg_kappa: float,
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
        best_linex = float(linex(y, F[:, best_idx], a=linex_a))
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
                "linex": float(linex(y, yhat, a=linex_a)),
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
                "linex": float(linex(y, rule_res.yhat, a=linex_a)),
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
                "linex": float(linex(y, kappa_res.yhat, a=linex_a)),
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

        simplex_bandit = SoftmaxSimplexBandit(
            n_forecasters=F.shape[1],
            context_dim=x.shape[1],
            lr=0.03,
            tau=1.0,
            loss=rl_loss,
            linex_a=linex_a,
            stochastic=False,
            reg_kappa=simplex_reg_kappa,
            use_state_as_reg=True,
            baseline_beta=0.95,
            seed=seed + 13 * h,
        )
        simplex_res = simplex_bandit.run(F=F, y=y, X=x, s=s, warmup=0)
        rows.append(
            {
                "loss_section": loss_section,
                "seed": float(seed),
                "horizon": float(h),
                "method": "RLSimplexBandit",
                "n_obs": float(y.size),
                "objective_value": float(obj(y, simplex_res.yhat)),
                "mse": float(mse(y, simplex_res.yhat)),
                "mae": float(mae(y, simplex_res.yhat)),
                "linex": float(linex(y, simplex_res.yhat, a=linex_a)),
                "avg_hhi": float(avg_hhi(simplex_res.weights)),
                "avg_lambda": float(avg_finite(simplex_res.meta.get("lambda_t", np.array([math.nan])))),
                "best_individual_objective": best_obj,
                "best_individual_mse": best_mse,
                "best_individual_linex": best_linex,
                "best_individual_idx": float(best_idx),
                "diag_main": float(avg_finite(simplex_res.meta.get("kl_t", np.array([math.nan])))),
                "diag_aux": math.nan,
            }
        )
        kl_t = np.asarray(simplex_res.meta.get("kl_t", np.full(y.size, np.nan)), dtype=float)
        slambda_t = np.asarray(simplex_res.meta.get("lambda_t", np.full(y.size, np.nan)), dtype=float)
        valid_idx = np.where(np.isfinite(simplex_res.hhi_t))[0]
        for t_idx in valid_idx:
            diag_rows.append(
                {
                    "loss_section": loss_section,
                    "seed": float(seed),
                    "horizon": float(h),
                    "method": "RLSimplexBandit",
                    "t": float(t_idx),
                    "action_idx": math.nan,
                    "action_name": "softmax_simplex",
                    "kappa": math.nan,
                    "lambda_t": float(slambda_t[t_idx]),
                    "hhi_t": float(simplex_res.hhi_t[t_idx]),
                    "kl_t": float(kl_t[t_idx]),
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
                "diag_main_mean": float(np.mean(diag_main[np.isfinite(diag_main)])) if np.any(np.isfinite(diag_main)) else math.nan,
                "diag_aux_mean": float(np.mean(diag_aux[np.isfinite(diag_aux)])) if np.any(np.isfinite(diag_aux)) else math.nan,
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


def parse_best_params(raw: str) -> Dict[str, float]:
    parsed = ast.literal_eval(raw)
    return {str(k): float(v) for k, v in dict(parsed).items()}


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
                    f"avg HHI={r['avg_hhi_mean']:.4f}, excess vs best indiv={r['avg_excess_objective_vs_best_individual']:.4f}"
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
    parser.add_argument("--n-trials", type=int, default=35)
    parser.add_argument("--linex-a", type=float, default=1.0)
    parser.add_argument("--simplex-reg-kappa", type=float, default=0.3)
    parser.add_argument("--out-dir", type=str, default="analyses/results")
    args = parser.parse_args()

    loss_sections = [("mse", "squared", "mse"), ("linex", "linex", "linex")]
    methods = list(ot.DEFAULT_METHOD_PARAMS.keys())
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tuning_rows: List[Dict[str, float]] = []
    tuned_by_loss_h: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = {}

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
                    "Optuna is required for this script. Install dependencies and rerun."
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
        for seed in args.test_seeds:
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
                simplex_reg_kappa=args.simplex_reg_kappa,
            )
            all_rows.extend(rows)
            all_diag_rows.extend(diag_rows)

    summary_rows = aggregate(all_rows)

    detailed_csv = out_dir / "dual_loss_full_detailed.csv"
    summary_csv = out_dir / "dual_loss_full_summary.csv"
    tuning_csv = out_dir / "dual_loss_full_tuned_params.csv"
    diag_csv = out_dir / "dual_loss_full_policy_diagnostics.csv"
    report_md = out_dir / "dual_loss_full_report.md"

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


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Block-bootstrap confidence intervals for SPF CPI3 ensemble improvement.
Reference: Politis & Romano (1994); Kunsch (1989) for MBB.
Run from the project root: python analyses/bootstrap_cpi3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.data.loading as ld
import src.data.cleaning as cln
import src.evaluation.spf_experiment as spfexp
import src.evaluation.optuna_tuning as ot
from src.evaluation.evaluation_helpers import loss_series
from src.evaluation.bootstrap_inference import (
    block_bootstrap_improvement_ci,
    block_size_sensitivity,
    dm_test,
    select_block_size,
)
from src.ensemblers import ensemblers

# ── Configuration ─────────────────────────────────────────────────────────────

HORIZON = "CPI3"
CPI3_OFFSET = 0          # offset found in spf_analysis.ipynb
LINEX_A = 0.5
MIN_OBS = 24
MIN_FORECASTERS = 3
ERROR_VOL_WINDOW = 8
N_BOOT = 4000
ALPHA = 0.05

RESULTS_DIR = PROJECT_ROOT / "analyses" / "results" / "spf"

# ── Step 1: Load raw data ─────────────────────────────────────────────────────

print("Loading SPF and VIX data...")
spf_raw = ld.load_or_download_csv(csv_path=ld.DEFAULT_CSV_PATH)
realiz_raw = ld.load_or_download_spf_realiz5_actuals(
    csv_path=ld.DEFAULT_SPF_CPI_REALIZ5_CSV_PATH,
    url=ld.SPF_CPI_REALIZ5_URL,
)
vix_raw = ld.load_or_download_vix_data(
    csv_path=ld.DEFAULT_VIX_CSV_PATH,
    url=ld.VIX_CSV_URL,
)

spf_clean, _ = cln.prepare_spf_data(spf_raw)

cpi_actuals_level = cln.clean_spf_realiz5_actuals(realiz_raw, realiz_col="Realiz5")
actual_inflation = cln.realiz5_to_actual_inflation(
    cpi_actuals_level, assume_already_annualized=True
)
actual_inflation = actual_inflation.rename(columns={"time": "period"})
actual_inflation["period"] = pd.PeriodIndex(
    pd.to_datetime(actual_inflation["period"]), freq="Q"
)

_, vix_quarterly = cln.prepare_vix_data(vix_raw)

# ── Step 2: Build CPI3 panel and horizon matrix ───────────────────────────────

print(f"Building {HORIZON} panel (offset={CPI3_OFFSET})...")
panel_long = cln.build_spf_cpi_panel_with_actuals(
    spf_clean=spf_clean,
    actual_inflation=actual_inflation,
    horizons=[HORIZON],
    horizon_offsets={HORIZON: CPI3_OFFSET},
)

window = cln.find_optimal_reporting_window(
    spf_clean=spf_clean,
    forecast_col=HORIZON,
    min_years=8,
)
print(
    f"  Window: {window['start_period']} → {window['end_period']}, "
    f"{window['n_ids']} forecasters, {window['n_periods']} periods"
)

mat_obj = cln.build_spf_horizon_matrix(
    panel_long=panel_long,
    horizon_name=HORIZON,
    start_period=window["start_period"],
    end_period=window["end_period"],
    forecaster_ids=window["ids"],
    min_obs=MIN_OBS,
    min_forecasters=MIN_FORECASTERS,
)
if mat_obj is None:
    raise RuntimeError(f"Could not build horizon matrix for {HORIZON}")

y = mat_obj["y"]
F = mat_obj["F"]
T, N = F.shape
print(f"  Matrix: T={T}, N={N} forecasters")

# ── Step 3: Build PCA state ────────────────────────────────────────────────────

components = spfexp.build_spf_state_components(
    mat=mat_obj["mat"],
    vix_quarterly=vix_quarterly,
    error_vol_window=ERROR_VOL_WINDOW,
)
state = spfexp.pca_state_from_components(components, expanding=True, min_train=12)

# ── Step 4: Load tuned parameters ─────────────────────────────────────────────

params_df = pd.read_csv(RESULTS_DIR / "spf_tuned_params.csv")
params_map = {k: dict(v) for k, v in ot.DEFAULT_METHOD_PARAMS.items()}
for _, row in params_df.iterrows():
    m = row["method"]
    if m in params_map:
        for col in ["eta", "kappa", "state_smoothing", "lambda_min"]:
            if col in row.index and pd.notna(row[col]):
                params_map[m][col] = float(row[col])

print("  Tuned params loaded from spf_tuned_params.csv")

# ── Step 5: Run ensemble models and collect per-period yhats ──────────────────

print("Running ensemble models...")

def run_model(name: str, params: dict, loss_section: str) -> np.ndarray:
    ens_loss = "linex" if loss_section == "linex" else "squared"
    p = params
    if name == "Mean":
        m = ensemblers.MeanEnsembler()
    elif name == "Median":
        m = ensemblers.MedianEnsembler()
    elif name == "OGDVanilla":
        m = ensemblers.OGDVanilla(eta=p.get("eta", 0.05), loss=ens_loss, linex_a=LINEX_A)
    elif name == "MWUMVanilla":
        m = ensemblers.MWUMVanilla(eta=p.get("eta", 0.30), loss=ens_loss, linex_a=LINEX_A)
    elif name == "OGDBoth":
        m = ensemblers.OGDConcentrationBoth(
            eta=p.get("eta", 0.05), kappa=p.get("kappa", 0.80),
            loss=ens_loss, linex_a=LINEX_A,
            state_smoothing=p.get("state_smoothing", ot.DEFAULT_STATE_SMOOTHING),
            lambda_min=p.get("lambda_min", ot.DEFAULT_LAMBDA_MIN),
        )
    elif name == "OGDConcOnly":
        m = ensemblers.OGDConcentrationOnly(
            eta=p.get("eta", 0.05), kappa=p.get("kappa", 0.80),
            loss=ens_loss, linex_a=LINEX_A,
            state_smoothing=p.get("state_smoothing", ot.DEFAULT_STATE_SMOOTHING),
            lambda_min=p.get("lambda_min", ot.DEFAULT_LAMBDA_MIN),
        )
    elif name == "MWUMBothKL":
        m = ensemblers.MWUMBothKL(
            eta=p.get("eta", 0.30), kappa=p.get("kappa", 0.80),
            loss=ens_loss, linex_a=LINEX_A,
            state_smoothing=p.get("state_smoothing", ot.DEFAULT_STATE_SMOOTHING),
            lambda_min=p.get("lambda_min", ot.DEFAULT_LAMBDA_MIN),
        )
    elif name == "MWUMConcOnlyKL":
        m = ensemblers.MWUMConcentrationOnlyKL(
            eta=p.get("eta", 0.30), kappa=p.get("kappa", 0.80),
            loss=ens_loss, linex_a=LINEX_A,
            state_smoothing=p.get("state_smoothing", ot.DEFAULT_STATE_SMOOTHING),
            lambda_min=p.get("lambda_min", ot.DEFAULT_LAMBDA_MIN),
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    needs_state = name in ot.STATE_METHODS
    result = m.run(F, y, s=state if needs_state else None)
    return np.asarray(result.yhat, dtype=float)


# Best individual (in-sample oracle benchmark, same as main experiment)
indiv_mse = np.array([float(np.mean((y - F[:, i]) ** 2)) for i in range(N)])
best_idx = int(np.argmin(indiv_mse))
yhat_bench = F[:, best_idx].copy()

model_names = ["Mean", "Median", "OGDVanilla", "MWUMVanilla",
               "OGDBoth", "OGDConcOnly", "MWUMBothKL", "MWUMConcOnlyKL"]

yhats: dict[str, np.ndarray] = {}
for loss_section in ["mse", "linex"]:
    for name in model_names:
        key = f"{name}|{loss_section}"
        yhats[key] = run_model(name, params_map.get(name, {}), loss_section)
        print(f"  {key}: MSE={float(np.mean((y - yhats[key])**2)):.4f}")

# ── Step 6: Compute per-period losses ─────────────────────────────────────────

bench_sq = loss_series(y, yhat_bench, loss="sq")
bench_linex = loss_series(y, yhat_bench, loss="linex", linex_a=LINEX_A)

# ── Step 7: Block-bootstrap CIs ───────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"Block-Bootstrap Inference — {HORIZON} (T={T} quarters)")
print(f"H0: ensemble not better than best individual (one-sided)")
print(f"Block size heuristic: l = ceil(T^(1/3)) = {select_block_size(T)}")
print(f"{'='*65}\n")

rows = []
for loss_section, bench_loss in [("mse", bench_sq), ("linex", bench_linex)]:
    loss_label = "sq" if loss_section == "mse" else "linex"
    for name in model_names:
        key = f"{name}|{loss_section}"
        ens_loss = loss_series(y, yhats[key], loss=loss_label, linex_a=LINEX_A)

        ci = block_bootstrap_improvement_ci(
            ens_loss, bench_loss, n_boot=N_BOOT, alpha=ALPHA,
            rng=np.random.default_rng(42),
        )
        dm = dm_test(ens_loss, bench_loss)

        rows.append({
            "method": name,
            "loss": loss_section,
            "improvement_pct": ci["estimate"],
            "ci_lower": ci["ci_lower"],
            "ci_upper": ci["ci_upper"],
            "mbb_p": ci["p_value"],
            "dm_stat": dm["stat"],
            "dm_p": dm["p_value"],
            "block_size": ci["block_size"],
            "T": ci["T"],
        })

df = pd.DataFrame(rows)

print(df.to_string(
    index=False,
    float_format=lambda x: f"{x:+.3f}" if abs(x) < 1000 else f"{x:.1f}",
))

# ── Step 8: Sensitivity to block size for the headline result ─────────────────

print(f"\n{'='*65}")
print(f"Block-size sensitivity — MWUMConcOnlyKL (MSE, {HORIZON})")
print(f"{'='*65}")

best_ens_loss = loss_series(y, yhats["MWUMConcOnlyKL|mse"], loss="sq")
sens = block_size_sensitivity(best_ens_loss, bench_sq, block_sizes=[2, 3, 4, 5, 6], n_boot=N_BOOT)
sens_df = pd.DataFrame([
    {
        "block_size": r["block_size"],
        "improvement_pct": r["estimate"],
        "95% CI": f"[{r['ci_lower']:+.2f}, {r['ci_upper']:+.2f}]",
        "mbb_p": r["p_value"],
    }
    for r in sens
])
print(sens_df.to_string(index=False))

# ── Step 9: Save results ───────────────────────────────────────────────────────

out_path = RESULTS_DIR / "spf_cpi3_bootstrap_inference.csv"
df.to_csv(out_path, index=False)
print(f"\nResults saved to {out_path.relative_to(PROJECT_ROOT)}")

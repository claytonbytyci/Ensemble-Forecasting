"""
Full M3 Monthly Macro ensemble experiment.

Each macro series has 1 forecast origin (Aug 1992) and 18 horizons.
We treat horizons as the sequential time axis (T=18 steps per series).

Key differences from the CLI in m3_macro_experiment.main():
  - Uses series_online_data_fixed_methods (horizons as time axis)
  - Runs across ALL 312 macro series
  - LINEX scaled to a=0.01 (appropriate for M3 level magnitudes)
  - Outputs to analyses/results/m3_macro/
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.cleaning import (
    align_m3_monthly_actuals_and_forecasts,
    prepare_m3_monthly_data,
)
from src.data.loading import (
    load_or_download_m3_quarterly_actuals,
    load_or_download_m3_quarterly_forecasts,
)
from src.evaluation import optuna_tuning as ot
from src.evaluation.m3_macro_experiment import (
    aggregate,
    evaluate_series_horizon,
    series_online_data_fixed_methods,
    write_csv,
    write_report,
)

# ── Parameters ────────────────────────────────────────────────────────────────
LOSS_SECTIONS = ["mse", "linex"]
LINEX_A = 0.01          # Scaled for M3 magnitudes (errors ~50-300 level)
INCLUDE_RL = True
MIN_OBS = 6             # Minimum horizons needed (T=18 so this is generous)
MIN_METHODS = 3

OUT_DIR = PROJECT_ROOT / "analyses" / "results" / "m3_macro"
OUT_STEM = "m3_macro_full"

KAPPA_GRID = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 8.0], dtype=float)
PARAMS_MAP = {k: dict(v) for k, v in ot.DEFAULT_METHOD_PARAMS.items()}


def main() -> None:
    print("Loading M3 monthly macro data...")
    actuals_raw = load_or_download_m3_quarterly_actuals()
    forecasts_raw = load_or_download_m3_quarterly_forecasts()

    actuals_macro, forecasts_macro, macro_series_ids = prepare_m3_monthly_data(
        actuals_raw, forecasts_raw, category="MACRO"
    )

    # Enforce M809..M1120 ordering
    macro_series_ids = pd.Index(
        sorted(
            [
                sid for sid in map(str, macro_series_ids.tolist())
                if sid.startswith("M") and sid[1:].isdigit() and 809 <= int(sid[1:]) <= 1120
            ],
            key=lambda s: int(s[1:]),
        ),
        name="series_id",
    )
    actuals_macro = actuals_macro.loc[actuals_macro["series_id"].isin(macro_series_ids)].copy()
    forecasts_macro = forecasts_macro.loc[forecasts_macro["series_id"].isin(macro_series_ids)].copy()

    aligned_long = align_m3_monthly_actuals_and_forecasts(actuals_macro, forecasts_macro)
    aligned_long = aligned_long.loc[aligned_long["series_id"].isin(macro_series_ids)].copy()

    series_ids = sorted(
        list(map(str, macro_series_ids.tolist())), key=lambda s: int(s[1:])
    )
    print(f"Macro series to evaluate: {len(series_ids)}")
    print(f"LINEX a = {LINEX_A}, Loss sections: {LOSS_SECTIONS}")
    print()

    detailed_rows: list[dict] = []
    diag_rows: list[dict] = []
    skip_counts = {"no_panel": 0}

    total = len(LOSS_SECTIONS) * len(series_ids)
    done = 0

    for loss_section in LOSS_SECTIONS:
        print(f"=== Loss section: {loss_section.upper()} ===")
        for sid in series_ids:
            sid_fc = forecasts_macro.loc[forecasts_macro["series_id"] == sid]
            methods_sid = sorted(
                sid_fc["method_id"].dropna().astype(str).unique().tolist()
            )

            d = series_online_data_fixed_methods(
                aligned_long=aligned_long,
                series_id=sid,
                required_methods=methods_sid,
                min_obs=MIN_OBS,
                min_methods=MIN_METHODS,
            )
            if d is None:
                skip_counts["no_panel"] += 1
                done += 1
                continue

            rows_h, diag_h = evaluate_series_horizon(
                series_id=sid,
                horizon=0,
                y=d["y"],
                F=d["F"],
                s=d["s"],
                loss_section=loss_section,
                linex_a=LINEX_A,
                include_rl=INCLUDE_RL,
                params_map=PARAMS_MAP,
                kappa_grid=KAPPA_GRID,
            )
            detailed_rows.extend(rows_h)
            diag_rows.extend(diag_h)
            done += 1

            if done % 100 == 0:
                print(f"  Progress: {done}/{total} ({100*done/total:.0f}%)")

    print()
    print(f"Detailed rows: {len(detailed_rows)}")
    print(f"Diagnostic rows: {len(diag_rows)}")
    print(f"Series evaluated: {len({r['series_id'] for r in detailed_rows})}")
    print(f"Skipped (no panel): {skip_counts['no_panel']}")

    summary_rows = aggregate(detailed_rows)
    summary_df = pd.DataFrame(summary_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detailed_csv = OUT_DIR / f"{OUT_STEM}_detailed.csv"
    summary_csv = OUT_DIR / f"{OUT_STEM}_summary.csv"
    diag_csv = OUT_DIR / f"{OUT_STEM}_policy_diagnostics.csv"
    report_md = OUT_DIR / f"{OUT_STEM}_report.md"

    write_csv(detailed_csv, detailed_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(diag_csv, diag_rows)
    write_report(report_md, summary_rows, detailed_rows, linex_a=LINEX_A)

    print()
    print(f"Wrote: {detailed_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {diag_csv}")
    print(f"Wrote: {report_md}")

    # ── Quick console summary ────────────────────────────────────────────────
    if not summary_df.empty:
        print()
        for ls in LOSS_SECTIONS:
            sub = summary_df[summary_df["loss_section"] == ls].sort_values("objective_mean")
            print(f"\n{'='*60}")
            print(f"  {ls.upper()} SUMMARY  (linex_a={LINEX_A})")
            print(f"{'='*60}")
            print(f"{'Method':<22} {'Obj Mean':>14} {'MSE Mean':>14} {'MAE Mean':>10} {'HHI':>7} {'Improve%':>10}")
            print("-" * 80)
            for _, r in sub.iterrows():
                imp = r["avg_improvement_pct_vs_best_individual"]
                imp_str = f"{imp:+.2f}" if np.isfinite(imp) else "  N/A"
                obj = r["objective_mean"]
                obj_str = f"{obj:.2f}" if np.isfinite(obj) else "  inf"
                hhi = r["avg_hhi_mean"]
                hhi_str = f"{hhi:.3f}" if np.isfinite(hhi) else "  N/A"
                print(
                    f"{r['method']:<22} {obj_str:>14} {r['mse_mean']:>14.2f} "
                    f"{r['mae_mean']:>10.2f} {hhi_str:>7} {imp_str:>10}"
                )


if __name__ == "__main__":
    main()

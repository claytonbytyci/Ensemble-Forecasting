import numpy as np
import pandas as pd

from src.data.cleaning import (
    build_spf_cpi_panel_with_actuals,
    build_spf_horizon_matrix,
    compute_annualized_cpi_inflation_from_level,
    extract_realtime_diagonal_actuals,
    prepare_vix_data,
)
from src.evaluation.spf_experiment import build_spf_state_components, pca_state_from_components


def test_extract_realtime_diagonal_actuals_basic():
    rt = pd.DataFrame(
        {
            "DATE": ["2020:01", "2020:04", "2020:07"],
            "CPI20Q1": [100.0, np.nan, np.nan],
            "CPI20Q2": [101.0, 102.0, np.nan],
            "CPI20Q3": [102.0, 103.0, 104.0],
        }
    )
    out = extract_realtime_diagonal_actuals(rt)
    assert list(out.columns) == ["period", "vintage_period", "cpi_level_rt"]
    assert len(out) == 3
    assert out.loc[0, "cpi_level_rt"] == 100.0
    assert out.loc[1, "cpi_level_rt"] == 102.0
    assert out.loc[2, "cpi_level_rt"] == 104.0


def test_spf_panel_and_matrix_build():
    diagonal = pd.DataFrame(
        {
            "period": [pd.Period("2020Q1"), pd.Period("2020Q2"), pd.Period("2020Q3"), pd.Period("2020Q4")],
            "vintage_period": [pd.Period("2020Q1"), pd.Period("2020Q2"), pd.Period("2020Q3"), pd.Period("2020Q4")],
            "cpi_level_rt": [100.0, 101.0, 102.0, 103.0],
        }
    )
    actual = compute_annualized_cpi_inflation_from_level(diagonal, method="compound")
    assert "actual_inflation_annualized" in actual.columns
    assert len(actual) == 3

    spf = pd.DataFrame(
        {
            "ID": [1, 2, 1, 2],
            "period": pd.PeriodIndex(["2019Q4", "2019Q4", "2020Q1", "2020Q1"], freq="Q").to_timestamp("Q"),
            "CPI1": [4.0, 4.5, 3.0, 3.5],
            "CPI2": [3.8, 4.2, 2.8, 3.2],
            "CPI3": [3.5, 3.7, 2.6, 2.9],
        }
    )
    panel = build_spf_cpi_panel_with_actuals(spf, actual, horizons=["CPI1", "CPI2"])
    assert {"ID", "origin_period", "target_period", "horizon_name", "forecast", "actual"}.issubset(panel.columns)

    m = build_spf_horizon_matrix(
        panel_long=panel,
        horizon_name="CPI1",
        min_obs=2,
        min_forecasters=2,
    )
    assert m is not None
    assert m["F"].shape[1] == 2
    assert m["y"].shape[0] >= 2


def test_state_components_and_pca_state():
    mat = pd.DataFrame(
        {
            "origin_period": pd.PeriodIndex(["2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q1"], freq="Q"),
            "target_period": pd.PeriodIndex(["2020Q2", "2020Q3", "2020Q4", "2021Q1", "2021Q2"], freq="Q"),
            "actual": [2.0, 2.2, 2.5, 2.1, 2.3],
            1: [1.8, 2.0, 2.7, 2.0, 2.2],
            2: [2.1, 2.3, 2.4, 2.2, 2.4],
            3: [2.2, 2.4, 2.6, 2.3, 2.5],
        }
    )
    vix_raw = pd.DataFrame(
        {
            "observation_date": pd.date_range("2020-01-01", periods=500, freq="D"),
            "VIXCLS": np.linspace(12.0, 30.0, 500),
        }
    )
    _, vix_q = prepare_vix_data(vix_raw)
    c = build_spf_state_components(mat=mat, vix_quarterly=vix_q, error_vol_window=3)
    assert {"vix_q_mean", "disagreement", "error_vol_recent"}.issubset(c.columns)

    s = pca_state_from_components(c, expanding=True, min_train=3)
    assert s.shape[0] == len(c)
    assert np.all(np.isfinite(s))
    assert np.mean(s) > 0.0

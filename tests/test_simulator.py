import numpy as np
import pytest

from src.data.simulator import make_environment_and_forecasts


def test_make_environment_and_forecasts_baseline_shapes():
    horizons = [1, 4]
    t = 180
    data, forecasts_by_h, names, s_unc = make_environment_and_forecasts(
        T=t,
        horizons=horizons,
        window=50,
        seed=123,
        include_oracle=True,
        scenario="baseline",
    )

    expected_len = data["pi"].shape[0]
    # Current simulator configs use burn_in=100, so returned length is T - burn_in.
    assert expected_len == t - 100
    for key in ["pi", "x", "i", "u", "regime", "sigma_pi", "sv_logvar"]:
        assert key in data
        assert data[key].shape == (expected_len,)

    assert len(names) > 0
    n = len(names)
    for h in horizons:
        assert h in forecasts_by_h
        assert forecasts_by_h[h].shape == (expected_len, n)

    assert s_unc.shape == (expected_len,)
    assert np.isclose(np.mean(s_unc), 1.0, atol=1e-10)
    assert np.all(s_unc > 0)


def test_make_environment_and_forecasts_discriminating_without_oracle():
    data, forecasts_by_h, names, s_unc = make_environment_and_forecasts(
        T=160,
        horizons=[1, 2, 8],
        window=40,
        seed=99,
        include_oracle=False,
        scenario="discriminating",
    )

    assert "ORACLE_STATE" not in names
    assert set(forecasts_by_h.keys()) == {1, 2, 8}
    assert data["pi"].shape == s_unc.shape


def test_make_environment_invalid_scenario_raises():
    with pytest.raises(ValueError, match="scenario must be"):
        make_environment_and_forecasts(T=80, horizons=[1], scenario="bad")

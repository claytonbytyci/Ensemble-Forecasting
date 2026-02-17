import numpy as np

from src.evaluation.evaluation_helpers import (
    best_forecaster_yhat,
    cumulative_loss,
    hhi_from_weights,
    linex_loss,
    mae,
    mse,
    rolling_mean,
)


def test_losses_ignore_non_finite_alignment():
    y = np.array([1.0, 2.0, np.nan, 4.0])
    yhat = np.array([1.5, np.nan, 3.0, 5.0])

    # valid pairs at indices 0 and 3 -> errors -0.5, -1.0
    assert np.isclose(mse(y, yhat), (0.25 + 1.0) / 2.0)
    assert np.isclose(mae(y, yhat), (0.5 + 1.0) / 2.0)

    lin = linex_loss(y, yhat, a=0.7)
    e = np.array([-0.5, -1.0])
    expected = np.mean(np.exp(0.7 * e) - 0.7 * e - 1.0)
    assert np.isclose(lin, expected)


def test_cumulative_and_rolling_behavior():
    x = np.array([np.nan, 1.0, 2.0, np.nan, 3.0])
    cum = cumulative_loss(x)
    assert np.isnan(cum[0])
    assert np.isclose(cum[1], 1.0)
    assert np.isclose(cum[2], 3.0)
    assert np.isclose(cum[3], 3.0)
    assert np.isclose(cum[4], 6.0)

    r = rolling_mean(np.array([1.0, 2.0, 3.0, 4.0]), window=2)
    assert np.isnan(r[0])
    assert np.isclose(r[1], 1.5)
    assert np.isclose(r[2], 2.5)
    assert np.isclose(r[3], 3.5)


def test_hhi_and_best_forecaster_helpers():
    w = np.array(
        [
            [0.5, 0.5],
            [0.2, 0.8],
            [np.nan, 1.0],
        ]
    )
    hhi = hhi_from_weights(w)
    assert np.isclose(hhi[0], 0.5)
    assert np.isclose(hhi[1], 0.68)
    assert np.isnan(hhi[2])

    f = np.array(
        [
            [1.0, 3.0, 2.0],
            [1.0, 2.0, 2.5],
            [1.0, 4.0, 1.5],
        ]
    )
    y = np.array([1.1, 1.2, 1.0])

    best_yhat, best_idx, best_score = best_forecaster_yhat(f, y, metric="mse")
    assert best_idx == 0
    assert best_yhat.shape == (3,)
    assert np.isclose(best_score, mse(y, f[:, 0]))

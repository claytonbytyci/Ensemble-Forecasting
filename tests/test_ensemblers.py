import numpy as np
import pytest

from src.ensemblers.ensemblers import (
    MeanEnsembler,
    MedianEnsembler,
    MWUMBothKL,
    MWUMConcentrationOnlyKL,
    MWUMVanilla,
    OGDConcentrationBoth,
    OGDConcentrationOnly,
    OGDVanilla,
    concentration_lambda,
    update_state_ema,
)


@pytest.fixture
def toy_data():
    t, n = 12, 4
    rng = np.random.default_rng(7)
    f = rng.normal(size=(t, n))
    y = rng.normal(size=t)
    s = np.linspace(0.5, 2.0, t)
    return f, y, s


def _assert_simplex_rows(w: np.ndarray) -> None:
    assert w.ndim == 2
    row_sums = np.sum(w, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-8)
    assert np.all(w >= -1e-12)


def test_mean_and_median_ensemblers(toy_data):
    f, y, _ = toy_data

    mean_res = MeanEnsembler().run(f, y)
    assert mean_res.yhat.shape == (f.shape[0],)
    assert mean_res.weights.shape == f.shape
    _assert_simplex_rows(mean_res.weights)

    med_res = MedianEnsembler().run(f, y)
    assert med_res.yhat.shape == (f.shape[0],)
    assert med_res.weights.shape == f.shape
    assert np.all(np.isnan(med_res.weights))


def test_ogd_vanilla_and_mwum_vanilla_shapes_and_weights(toy_data):
    f, y, _ = toy_data

    for model in [
        OGDVanilla(eta=0.05),
        OGDVanilla(eta=0.05, loss="linex", linex_a=0.7),
        MWUMVanilla(eta=0.3),
        MWUMVanilla(eta=0.3, loss="linex", linex_a=0.5),
    ]:
        res = model.run(f, y)
        assert res.yhat.shape == (f.shape[0],)
        assert res.weights.shape == f.shape
        _assert_simplex_rows(res.weights)


def test_concentration_models_require_state(toy_data):
    f, y, s = toy_data

    state_models = [
        OGDConcentrationBoth(eta=0.05, kappa=0.8),
        OGDConcentrationOnly(kappa=0.8),
        MWUMBothKL(eta=0.3, kappa=0.8),
        MWUMConcentrationOnlyKL(kappa=0.8),
    ]

    for model in state_models:
        with pytest.raises(ValueError, match="state series"):
            model.run(f, y, s=None)

        res = model.run(f, y, s=s)
        assert res.yhat.shape == (f.shape[0],)
        assert res.weights.shape == f.shape
        _assert_simplex_rows(res.weights)


def test_hyperparameter_validation_and_state_utils(toy_data):
    f, y, _ = toy_data

    with pytest.raises(ValueError, match="eta must be > 0"):
        OGDVanilla(eta=0.0).run(f, y)

    with pytest.raises(ValueError, match="eta must be > 0"):
        MWUMVanilla(eta=-0.1).run(f, y)

    with pytest.raises(ValueError, match="must satisfy 0 <= rho <= 1"):
        update_state_ema(None, 1.0, rho=1.1)

    assert update_state_ema(None, 2.0, rho=0.8) == 2.0
    assert np.isclose(update_state_ema(2.0, 4.0, rho=0.5), 3.0)

    lam = concentration_lambda(kappa=0.8, state_ema=2.0, lambda_min=1e-6)
    assert lam > 1e-6

    with pytest.raises(ValueError, match="kappa must be >= 0"):
        concentration_lambda(kappa=-0.1, state_ema=1.0)


def test_mwum_concentration_only_degenerate_lambda_path():
    # lambda_t = 0 branch is reachable when kappa=lambda_min=0 and s_t=0.
    f = np.array(
        [
            [0.2, -0.1, 1.0],
            [0.3, 0.1, 0.4],
            [0.5, 0.2, 0.0],
        ],
        dtype=float,
    )
    y = np.array([0.0, 0.0, 0.0], dtype=float)
    s = np.zeros(3, dtype=float)

    model = MWUMConcentrationOnlyKL(kappa=0.0, lambda_min=0.0)
    res = model.run(f, y, s=s)

    assert res.weights.shape == f.shape
    _assert_simplex_rows(res.weights)

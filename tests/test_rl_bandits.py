import numpy as np
import pytest

from src.ensemblers.rl import KappaBandit, RuleSelectionBandit


@pytest.fixture
def bandit_data():
    t, n, d = 20, 5, 4
    rng = np.random.default_rng(11)
    f = rng.normal(size=(t, n))
    y = rng.normal(size=t)
    x = rng.normal(size=(t, d))
    s = np.abs(rng.normal(size=t)) + 0.1
    return f, y, x, s


def test_rule_selection_bandit_shapes_and_consistency(bandit_data):
    f, y, x, s = bandit_data
    model = RuleSelectionBandit(n_forecasters=f.shape[1], context_dim=x.shape[1], alpha=0.3)
    res = model.run(F=f, y=y, X=x, s=s)

    assert res.yhat.shape == (f.shape[0],)
    assert res.weights.shape == f.shape
    assert res.loss_t.shape == (f.shape[0],)
    assert res.reward_t.shape == (f.shape[0],)
    assert res.hhi_t.shape == (f.shape[0],)

    actions = res.meta["actions_t"]
    assert actions.shape == (f.shape[0],)
    assert np.all((actions >= 0) & (actions < len(model.action_names)))
    assert np.allclose(res.reward_t, -res.loss_t)

    row_sums = np.sum(res.weights, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-8)


def test_rule_selection_bandit_skips_invalid_rows(bandit_data):
    f, y, x, s = bandit_data
    f = f.copy()
    x = x.copy()
    y = y.copy()
    f[3, 1] = np.nan
    x[7, 2] = np.nan
    y[10] = np.nan

    model = RuleSelectionBandit(n_forecasters=f.shape[1], context_dim=x.shape[1])
    res = model.run(F=f, y=y, X=x, s=s)

    for idx in [3, 7, 10]:
        assert np.isnan(res.yhat[idx])
        assert np.all(np.isnan(res.weights[idx]))
        assert res.meta["actions_t"][idx] == -1


def test_kappa_bandit_shapes_and_meta(bandit_data):
    f, y, x, s = bandit_data
    grid = np.array([0.01, 0.1, 1.0], dtype=float)
    model = KappaBandit(kappa_grid=grid, context_dim=x.shape[1], alpha=0.4)
    res = model.run(F=f, y=y, X=x, s=s)

    assert res.yhat.shape == (f.shape[0],)
    assert res.weights.shape == f.shape

    for key in ["kappa_t", "actions_t", "lambda_t", "state_ema_t"]:
        assert key in res.meta
        assert res.meta[key].shape == (f.shape[0],)

    actions = res.meta["actions_t"]
    assert np.all((actions >= 0) & (actions < grid.size))
    assert np.allclose(res.reward_t, -res.loss_t)

    valid_lambda = res.meta["lambda_t"][np.isfinite(res.meta["lambda_t"])]
    assert np.all(valid_lambda > 0)


def test_bandit_shape_validation_raises(bandit_data):
    f, y, x, s = bandit_data

    rsb = RuleSelectionBandit(n_forecasters=f.shape[1], context_dim=x.shape[1])
    with pytest.raises(ValueError, match="Need y"):
        rsb.run(F=f, y=y[:-1], X=x, s=s)

    kb = KappaBandit(kappa_grid=np.array([0.1, 0.3]), context_dim=x.shape[1])
    with pytest.raises(ValueError, match="Need y"):
        kb.run(F=f, y=y, X=x[:-1], s=s)

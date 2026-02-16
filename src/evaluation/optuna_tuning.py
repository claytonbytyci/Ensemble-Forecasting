from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.ensemblers import ensemblers


TunableMethod = str
DataSlice = Tuple[np.ndarray, np.ndarray, np.ndarray]  # (y, F, s)


DEFAULT_METHOD_PARAMS: Dict[TunableMethod, Dict[str, float]] = {
    "Mean": {},
    "Median": {},
    "OGDVanilla": {"eta": 0.05},
    "MWUMVanilla": {"eta": 0.30},
    "OGDBoth": {"eta": 0.05, "kappa": 0.80},
    "OGDConcOnly": {"kappa": 0.80},
    "MWUMBothKL": {"eta": 0.30, "kappa": 0.80},
    "MWUMConcOnlyKL": {"kappa": 0.80},
}

STATE_METHODS = {"OGDBoth", "OGDConcOnly", "MWUMBothKL", "MWUMConcOnlyKL"}


@dataclass
class TuneResult:
    method: str
    best_params: Dict[str, float]
    best_value: float


def _mse(y: np.ndarray, yhat: np.ndarray) -> float:
    e = np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)
    return float(np.mean(e * e))


def _linex(y: np.ndarray, yhat: np.ndarray, a: float) -> float:
    e = np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)
    return float(np.mean(np.exp(a * e) - a * e - 1.0))


def _score(y: np.ndarray, yhat: np.ndarray, metric: str, linex_a: float) -> float:
    if metric == "mse":
        return _mse(y, yhat)
    if metric == "linex":
        return _linex(y, yhat, a=linex_a)
    raise ValueError("metric must be 'mse' or 'linex'")


def _build_model(
    method: TunableMethod,
    params: Optional[Dict[str, float]] = None,
    loss: str = "squared",
    linex_a: float = 1.0,
):
    p = {} if params is None else dict(params)
    loss_name = "linex" if loss == "linex" else "squared"

    if method == "Mean":
        return ensemblers.MeanEnsembler()
    if method == "Median":
        return ensemblers.MedianEnsembler()
    if method == "OGDVanilla":
        return ensemblers.OGDVanilla(eta=float(p.get("eta", 0.05)), loss=loss_name, linex_a=linex_a)
    if method == "MWUMVanilla":
        return ensemblers.MWUMVanilla(eta=float(p.get("eta", 0.30)), loss=loss_name, linex_a=linex_a)
    if method == "OGDBoth":
        return ensemblers.OGDConcentrationBoth(
            eta=float(p.get("eta", 0.05)),
            kappa=float(p.get("kappa", 0.80)),
            loss=loss_name,
            linex_a=linex_a,
        )
    if method == "OGDConcOnly":
        return ensemblers.OGDConcentrationOnly(
            kappa=float(p.get("kappa", 0.80)),
            loss=loss_name,
            linex_a=linex_a,
        )
    if method == "MWUMBothKL":
        return ensemblers.MWUMBothKL(
            eta=float(p.get("eta", 0.30)),
            kappa=float(p.get("kappa", 0.80)),
            loss=loss_name,
            linex_a=linex_a,
        )
    if method == "MWUMConcOnlyKL":
        return ensemblers.MWUMConcentrationOnlyKL(
            kappa=float(p.get("kappa", 0.80)),
            loss=loss_name,
            linex_a=linex_a,
        )

    raise ValueError(f"Unknown method: {method}")


def _suggest_params(trial, method: TunableMethod) -> Dict[str, float]:
    if method in {"Mean", "Median"}:
        return {}

    if method == "OGDVanilla":
        return {"eta": trial.suggest_float("eta", 1e-3, 0.35, log=True)}

    if method == "MWUMVanilla":
        return {"eta": trial.suggest_float("eta", 1e-3, 3.0, log=True)}

    if method == "OGDBoth":
        return {
            "eta": trial.suggest_float("eta", 1e-3, 0.35, log=True),
            "kappa": trial.suggest_float("kappa", 1e-3, 8.0, log=True),
        }

    if method == "OGDConcOnly":
        return {"kappa": trial.suggest_float("kappa", 1e-3, 8.0, log=True)}

    if method == "MWUMBothKL":
        return {
            "eta": trial.suggest_float("eta", 1e-3, 3.0, log=True),
            "kappa": trial.suggest_float("kappa", 1e-3, 8.0, log=True),
        }

    if method == "MWUMConcOnlyKL":
        return {"kappa": trial.suggest_float("kappa", 1e-3, 8.0, log=True)}

    raise ValueError(f"Unknown method: {method}")


def tune_method_optuna(
    method: TunableMethod,
    data_slices: Iterable[DataSlice],
    n_trials: int = 40,
    seed: int = 0,
    loss: str = "squared",
    linex_a: float = 1.0,
    objective_metric: str = "mse",
):
    """
    Tune one method over provided slices using Optuna.

    data_slices: iterable of (y, F, s)
      - y: target series
      - F: (T,N) expert forecasts aligned with y
      - s: uncertainty state aligned with y
    """
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Optuna is required for tuning. Install with `pip install optuna` "
            "or add it to project dependencies."
        ) from exc

    slices = list(data_slices)
    if len(slices) == 0:
        raise ValueError("data_slices is empty; nothing to tune on.")

    if method in {"Mean", "Median"}:
        # No tunable hyperparameters.
        model = _build_model(method, {}, loss=loss, linex_a=linex_a)
        losses = []
        for y, F, s in slices:
            res = model.run(F, y, s=s if method in STATE_METHODS else None)
            losses.append(_score(y, res.yhat, metric=objective_metric, linex_a=linex_a))
        best_value = float(np.mean(losses))
        dummy = optuna.create_study(direction="minimize")
        return TuneResult(method=method, best_params={}, best_value=best_value), dummy

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial):
        params = _suggest_params(trial, method)
        model = _build_model(method, params, loss=loss, linex_a=linex_a)

        fold_losses = []
        for y, F, s in slices:
            res = model.run(F, y, s=s if method in STATE_METHODS else None)
            fold_losses.append(_score(y, res.yhat, metric=objective_metric, linex_a=linex_a))

        return float(np.mean(fold_losses))

    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)

    return (
        TuneResult(
            method=method,
            best_params={k: float(v) for k, v in study.best_params.items()},
            best_value=float(study.best_value),
        ),
        study,
    )


def tune_all_methods_optuna(
    data_slices: Iterable[DataSlice],
    methods: Optional[List[TunableMethod]] = None,
    n_trials: int = 40,
    seed: int = 0,
    loss: str = "squared",
    linex_a: float = 1.0,
    objective_metric: str = "mse",
) -> Dict[TunableMethod, TuneResult]:
    """Tune all requested methods and return best params + objective values."""
    if methods is None:
        methods = list(DEFAULT_METHOD_PARAMS.keys())

    results: Dict[TunableMethod, TuneResult] = {}
    for i, method in enumerate(methods):
        result, _ = tune_method_optuna(
            method=method,
            data_slices=data_slices,
            n_trials=n_trials,
            seed=seed + 17 * i,
            loss=loss,
            linex_a=linex_a,
            objective_metric=objective_metric,
        )
        results[method] = result

    return results

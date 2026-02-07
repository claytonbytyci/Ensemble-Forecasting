# Ensemble Forecasting

## Overview
This project develops online learning algorithms to optimally combine forecasts from multiple models. The core methods are:
- Online gradient descent over weights on forecasters.
- Multiplicative weights updates for fast, regret-minimizing reweighting.

The learning objective includes a KL-divergence penalty from equal weights to encode an inductive bias toward model averaging. This reflects a conservative, theory-driven prior inspired by Brainard conservatism: when model uncertainty is high, the system should adjust weights cautiously and remain close to a diversified baseline.

## Goals
- Implement weight-updating schemes for ensemble forecasts under streaming data.
- Compare OGD vs. multiplicative-weights under different loss functions.
- Evaluate the impact of KL regularization on stability, regret, and forecast accuracy.
- Provide simulation and empirical notebooks for benchmarking.

## Repository Layout
- `data/` : Datasets (raw and processed).
- `src/` : Core algorithms and utilities.
- `tests/` : Unit and integration tests.
- `analyses/` : Notebooks and experiment scripts.

## Environment
Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate ensemble
```

Optional dev tools:

```bash
pip install -e ".[dev]"
```

## Quick Start (Planned)
- Implement the `OnlineGradientDescent` and `MultiplicativeWeights` classes in `src/`.
- Add KL-regularized loss wrappers for weight updates.
- Run baseline experiments in `analyses/`.

## License
MIT License. See `LICENSE`.

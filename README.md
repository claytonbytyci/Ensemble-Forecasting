# Ensemble Forecasting

A research codebase for online forecast combination under model uncertainty, with concentration/diversification control and RL-based adaptive policy selection.

## 1. Motivation
Forecast combination is often more robust than committing to a single forecaster, but in online settings the ensemble must update weights sequentially as new outcomes arrive. This project studies that online weighting problem with:

- classical online learners (OGD, MWUM),
- explicit concentration penalties toward a baseline (uniform) portfolio,
- asymmetric loss support (LINEX),
- contextual bandits that adapt the weighting policy itself.

The concentration penalty is designed to encode a conservative prior: under higher uncertainty, keep the portfolio closer to a diversified baseline.

## 2. Mathematical Setup
At each time step `t`:

- `N` experts provide forecasts `f[i,t]` for target `y[t]`.
- A weight vector `w[t]` is used to combine forecasts.
- `w[t]` lies on the simplex:

```text
Delta^N = { w in R^N : w_i >= 0 and sum_i w_i = 1 }
```

Combined prediction:

```text
y_hat[t] = sum_i w[i,t] * f[i,t] = dot(w[t], f[t])
```

Error:

```text
e[t] = y[t] - y_hat[t]
```

### Losses
Implemented losses:

- Squared loss:

```text
L_sq(e) = e^2
```

- LINEX loss:

```text
L_linex(e; a) = exp(a*e) - a*e - 1
```

Derivatives used by gradient-based updates:

```text
dL_sq/de = 2e
dL_linex/de = a * (exp(a*e) - 1)
```

## 3. Core Algorithms (`src/ensemblers/ensemblers.py`)
The following online ensemblers are implemented.

### 3.1 Baselines
1. **MeanEnsembler**

```text
w[i,t] = 1/N
```

2. **MedianEnsembler**

```text
y_hat[t] = median_i f[i,t]
```

(Nonlinear aggregator; no meaningful simplex weight vector is returned.)

### 3.2 OGD Family
3. **OGDVanilla**

```text
w[t+1] = Proj_Delta( w[t] - eta * grad_w ell_t(w[t]) )
where ell_t(w[t]) = L( y[t] - dot(w[t], f[t]) )
```

4. **OGDConcentrationBoth**

```text
w[t+1] = argmin_{w in Delta}
         <grad ell_t(w[t]), w>
         + (1/(2*eta)) * ||w - w[t]||_2^2
         + lambda[t] * ||w - pi||_2^2
```

5. **OGDConcentrationOnly**

```text
w[t+1] = argmin_{w in Delta}
         <grad ell_t(w[t]), w>
         + lambda[t] * ||w - pi||_2^2
```

### 3.3 MWUM Family
6. **MWUMVanilla**
Per-expert losses `ell[i,t] = L(y[t] - f[i,t])`:

```text
w[i,t+1] proportional to w[i,t] * exp( -eta * ell[i,t] )
```

7. **MWUMBothKL**

```text
w[t+1] = argmin_{w in Delta}
         <w, ell_t>
         + (1/eta) * KL(w || w[t])
         + lambda[t] * KL(w || pi)
```

8. **MWUMConcentrationOnlyKL**

```text
w[t+1] = argmin_{w in Delta}
         <w, ell_t>
         + lambda[t] * KL(w || pi)
```

Closed form:

```text
w[i] proportional to pi[i] * exp( -ell[i,t] / lambda[t] )
```

### 3.4 Concentration Schedule
All concentration-penalized methods use:

1. EMA state smoothing:

```text
s_ema[t] = rho * s_ema[t-1] + (1-rho) * s[t]
```

2. State-dependent penalty:

```text
lambda[t] = lambda_min + kappa * log(1 + s_ema[t])
```

Here `s[t]` is an uncertainty proxy (in simulations: scaled inflation shock volatility).

## 4. RL Policy Layer (`src/ensemblers/rl.py`)
### 4.1 RuleSelectionBandit
A contextual LinUCB policy selects one weighting rule each period (e.g., Mean, OGD, MWUM, concentration variants).

- Context `x[t]` is built from forecast dispersion/statistics.
- Action reward is negative forecast loss.
- Selected action executes one online update step and yields `w[t]` and `y_hat[t]`.

### 4.2 KappaBandit
A contextual LinUCB policy selects `kappa[t]` from a discrete grid each period for concentration-only MWUM updates.

- Chosen `kappa[t]` induces `lambda[t]` via the same schedule above.
- Predict with current weights, observe reward, update bandit, then update portfolio weights.

Both RL modules log diagnostics including chosen actions, rewards, HHI, and (`KappaBandit`) `kappa[t]`, `lambda[t]`, and smoothed state.

## 5. Simulation Engine (`src/data/simulator.py`)
The synthetic environment is a regime-switching macro process with stochastic volatility.

State variables include inflation `pi[t]`, output gap `x[t]`, policy rate `i[t]`, and supply shock `u[t]`.

Regime dynamics are Markovian with transition matrix `P`, and equations are regime-dependent (schematic):

```text
pi[t] = alpha[s]*pi[t-1] + beta[s]*x[t-1] + gamma[s]*u[t] + eps_pi[t]
x[t]  = rho[s]*x[t-1] - phi[s]*(i[t-1]-pi[t-1]) + eps_x[t]
i[t]  = psi_pi[s]*pi[t] + psi_x[s]*x[t] + eps_i[t]
```

Two scenarios are provided:

- `baseline`
- `discriminating` (stronger regime contrast, noisier panel, clearer method separation)

The simulator also generates a heterogeneous forecaster panel (AR variants, Phillips-style models, heuristics, optional oracle forecaster).

## 6. Evaluation & Tuning
### 6.1 Evaluation Helpers (`src/evaluation/evaluation_helpers.py`)
Includes:

- loss utilities (`mse`, `mae`, `linex_loss`),
- best individual forecaster extraction,
- rolling/cumulative loss diagnostics,
- HHI computation and plotting utilities.

### 6.2 Optuna Tuning (`src/evaluation/optuna_tuning.py`)
Provides:

- default parameter map (`DEFAULT_METHOD_PARAMS`),
- per-method search spaces,
- tuning for one/all methods across data slices,
- objective metrics: `mse` or `linex`.

### 6.3 End-to-End Pipeline (`src/evaluation/dual_loss_with_rl.py`)
This module orchestrates:

1. data generation and horizon alignment,
2. optional hyperparameter tuning or CSV reuse,
3. per-seed/per-horizon evaluation of regular + RL methods,
4. aggregation and reporting,
5. output artifacts (CSV, report markdown, figures).

## 7. Repository Layout

```text
Ensemble-Forecasting/
  data/
    SPF_Individual_CPI.csv
  src/
    data/
      loading.py
      cleaning.py
      plots.py
      simulator.py
    ensemblers/
      ensemblers.py
      rl.py
    evaluation/
      evaluation_helpers.py
      optuna_tuning.py
      dual_loss_with_rl.py
  analyses/
    run_dual_loss_with_rl.py        # thin wrapper to src.evaluation.dual_loss_with_rl.main
    dual_loss_with_rl_analysis.ipynb
    simulation_ensemble_analysis.ipynb
    eda.ipynb
    theory_calculations.ipynb
    results/
  tests/
    test_ensemblers.py
    test_rl_bandits.py
    test_simulator.py
    test_evaluation_helpers.py
  pyproject.toml
  environment.yml
```

## 8. Installation
### Conda
```bash
conda env create -f environment.yml
conda activate ensemble
```

### Editable package + dev tools
```bash
python -m pip install -e ".[dev]"
```

If `pytest` resolves to a system interpreter, run tests as:

```bash
python -m pytest
```

## 9. Running Experiments
### 9.1 CLI via analysis wrapper
From repo root:

```bash
python analyses/run_dual_loss_with_rl.py \
  --scenario discriminating \
  --skip-tuning \
  --tuned-params-csv analyses/results/dual_loss_full_tuned_params.csv \
  --n-test-sims 40 \
  --test-seed-start 4 \
  --out-dir analyses/results/multi_sim_40 \
  --output-stem dual_loss_multi40
```

### 9.2 Direct module execution
```bash
python -m src.evaluation.dual_loss_with_rl --help
```

## 10. Output Artifacts
The dual-loss pipeline writes:

- `<stem>_detailed.csv`: simulation-level method metrics,
- `<stem>_summary.csv`: aggregated metrics (mean/std etc.),
- `<stem>_tuned_params.csv`: tuned regular-method params,
- `<stem>_policy_diagnostics.csv`: RL action/state diagnostics,
- `<stem>_report.md`: markdown summary report,
- `<stem>_improvement_vs_hhi_<loss>_h<h>.png`: group-level plots,
- `<stem>_individual_improvement_vs_hhi.png`: simulation-level scatter grid.

## 11. Testing
Current suite targets core invariants and edge paths:

- simplex and shape invariants for all ensemblers,
- concentration-state requirement and degenerate branches,
- RL bandit metadata/action bounds and invalid-row handling,
- simulator output contracts and scenario validation,
- evaluation helper correctness (finite masking, HHI, cumulative/rolling behavior).

Run:

```bash
python -m pytest tests -q
```

## 12. Notes on Data and Burn-In
In the simulator, returned effective sample size is `T - burn_in` because initial transients are discarded (default `burn_in=100`). Tests account for this explicitly.

## 13. License
MIT. See `LICENSE`.

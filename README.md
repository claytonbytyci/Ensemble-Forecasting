# Ensemble Forecasting

A research codebase for online forecast combination under model uncertainty, with concentration/diversification control and RL-based adaptive policy selection.

## 1. Motivation
Forecast combination is often more robust than committing to a single forecaster, but in online settings the ensemble must update weights sequentially as new outcomes arrive. This project studies that online weighting problem with:

- classical online learners (OGD, MWUM),
- explicit concentration penalties toward a baseline (uniform) portfolio,
- asymmetric loss support (LINEX),
- contextual bandits that adapt the weighting policy itself.

The concentration penalty encodes a conservative prior: under higher uncertainty, keep the portfolio closer to a diversified baseline.

## 2. Mathematical Setup
At each time step `t`:

- `N` experts provide forecasts for target `y_t`.
- A weight vector `w_t` combines forecasts.
- `w_t` lies on the simplex.

$$
\Delta^N = \{w \in \mathbb{R}^N : w_i \ge 0,\ \sum_i w_i = 1\}
$$

Combined prediction:

$$
\hat y_t = \sum_{i=1}^N w_{i,t} f_{i,t} = w_t^\top f_t
$$

Error:

$$
e_t = y_t - \hat y_t
$$

### Losses
Implemented losses:

Squared loss:

$$
L_{\text{sq}}(e_t) = e_t^2
$$

LINEX loss:

$$
L_{\text{linex}}(e_t; a) = \exp(a e_t) - a e_t - 1
$$

Derivatives used by gradient-based updates:

$$
\frac{\partial L_{\text{sq}}}{\partial e} = 2e
$$

$$
\frac{\partial L_{\text{linex}}}{\partial e} = a\left(\exp(ae) - 1\right)
$$

## 3. Core Algorithms (`src/ensemblers/ensemblers.py`)
The following online ensemblers are implemented.

### 3.1 Baselines
1. **MeanEnsembler**

$$
w_{i,t} = \frac{1}{N}
$$

2. **MedianEnsembler**

$$
\hat y_t = \operatorname{median}_i(f_{i,t})
$$

(Nonlinear aggregator; no meaningful simplex weight vector is returned.)

### 3.2 OGD Family
3. **OGDVanilla**

$$
w_{t+1} = \Pi_\Delta\left(w_t - \eta\nabla_w\ell_t(w_t)\right)
$$

$$
\ell_t(w_t) = L\left(y_t - w_t^\top f_t\right)
$$

4. **OGDConcentrationBoth**

$$
w_{t+1} = \arg\min_{w\in\Delta}
\left\langle \nabla\ell_t(w_t), w \right\rangle
+ \frac{1}{2\eta}\|w-w_t\|_2^2
+ \lambda_t\|w-\pi\|_2^2
$$

5. **OGDConcentrationOnly**

$$
w_{t+1} = \arg\min_{w\in\Delta}
\left\langle \nabla\ell_t(w_t), w \right\rangle
+ \lambda_t\|w-\pi\|_2^2
$$

### 3.3 MWUM Family
6. **MWUMVanilla**
Per-expert losses:

$$
\ell_{i,t} = L\left(y_t - f_{i,t}\right)
$$

Update:

$$
w_{i,t+1} \propto w_{i,t}\exp\left(-\eta\ell_{i,t}\right)
$$

7. **MWUMBothKL**

$$
w_{t+1} = \arg\min_{w\in\Delta}
\left\langle w,\ell_t \right\rangle
+ \frac{1}{\eta}\mathrm{KL}(w\|w_t)
+ \lambda_t\mathrm{KL}(w\|\pi)
$$

8. **MWUMConcentrationOnlyKL**

$$
w_{t+1} = \arg\min_{w\in\Delta}
\left\langle w,\ell_t \right\rangle
+ \lambda_t\mathrm{KL}(w\|\pi)
$$

Closed form:

$$
w_i \propto \pi_i\exp\left(-\frac{\ell_{i,t}}{\lambda_t}\right)
$$

### 3.4 Concentration Schedule
All concentration-penalized methods use:

1. EMA state smoothing:

$$
\tilde s_t = \rho\tilde s_{t-1} + (1-\rho)s_t
$$

2. State-dependent penalty:

$$
\lambda_t = \lambda_{\min} + \kappa\log\left(1+\tilde s_t\right)
$$

Here `s_t` is an uncertainty proxy (in simulations: scaled inflation shock volatility).

## 4. RL Policy Layer (`src/ensemblers/rl.py`)
### 4.1 RuleSelectionBandit
A contextual LinUCB policy selects one weighting rule each period (e.g., Mean, OGD, MWUM, concentration variants).

- Context `x_t` is built from forecast dispersion/statistics.
- Action reward is negative forecast loss.
- Selected action executes one online update step and yields `w_t` and `\hat y_t`.

### 4.2 KappaBandit
A contextual LinUCB policy selects `\kappa_t` from a discrete grid each period for concentration-only MWUM updates.

- Chosen `\kappa_t` induces `\lambda_t` via the schedule above.
- Predict with current weights, observe reward, update bandit, then update portfolio weights.

Both RL modules log diagnostics including chosen actions, rewards, HHI, and (`KappaBandit`) `\kappa_t`, `\lambda_t`, and smoothed state.

## 5. Simulation Engine (`src/data/simulator.py`)
The synthetic environment is a regime-switching macro process with stochastic volatility.

State variables include inflation, output gap, policy rate, and supply shock.

Schematic regime-dependent equations:

$$
\pi_t = \alpha_s\pi_{t-1} + \beta_s x_{t-1} + \gamma_s u_t + \varepsilon_t^{\pi}
$$

$$
x_t = \rho_s x_{t-1} - \phi_s\left(i_{t-1}-\pi_{t-1}\right) + \varepsilon_t^{x}
$$

$$
i_t = \psi_{\pi,s}\pi_t + \psi_{x,s}x_t + \varepsilon_t^{i}
$$

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

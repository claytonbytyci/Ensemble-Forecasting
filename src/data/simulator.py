import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal

# -----------------------------
# 1) Macro environment simulator
# -----------------------------

@dataclass
class MacroSimConfig:
    T: int = 600                       # length
    burn_in: int = 100                 # discard initial transient
    seed: int = 0

    # Regimes: 0=anchored, 1=unanchored/high persistence, 2=supply-shock/high pass-through
    P: np.ndarray = None               # transition matrix (3x3)

    # Regime-dependent inflation dynamics: pi_t = a*pi_{t-1} + b*x_{t-1} + g*u_t + eps_pi
    alpha: np.ndarray = None           # (3,)
    beta: np.ndarray = None            # (3,)
    gamma: np.ndarray = None           # (3,)

    # Output gap: x_t = rho*x_{t-1} - phi*(i_{t-1}-pi_{t-1}) + eps_x
    rho: np.ndarray = None             # (3,)
    phi: np.ndarray = None             # (3,)
    sigma_x: np.ndarray = None         # (3,)

    # Policy rule: i_t = psi_pi*pi_t + psi_x*x_t + eps_i
    psi_pi: np.ndarray = None          # (3,)
    psi_x: np.ndarray = None           # (3,)
    sigma_i: np.ndarray = None         # (3,)

    # Supply shock process u_t (AR(1))
    u_rho: float = 0.6
    sigma_u: float = 1.0

    # Stochastic volatility on inflation shock: log(sigma_pi^2) evolves with regime-dependent mean
    sv_phi: float = 0.97
    sv_tau: float = 0.10
    sv_mu: np.ndarray = None           # (3,) mean log-variance level
    base_sigma_pi: float = 0.3         # scales overall inflation shock size


def _default_config(T=600, seed=0) -> MacroSimConfig:
    cfg = MacroSimConfig(T=T, seed=seed)
    # Persistent regimes with occasional switching
    cfg.P = np.array([
        [0.96, 0.03, 0.01],
        [0.04, 0.94, 0.02],
        [0.03, 0.05, 0.92],
    ])
    # Inflation transmission differences
    cfg.alpha = np.array([0.40, 0.85, 0.55])   # persistence higher in regime 1
    cfg.beta  = np.array([0.10, 0.05, 0.08])   # Phillips slope flatter in regime 1
    cfg.gamma = np.array([0.10, 0.10, 0.45])   # supply pass-through high in regime 2

    # Output dynamics (demand propagation/policy transmission)
    cfg.rho     = np.array([0.85, 0.90, 0.80])
    cfg.phi     = np.array([0.15, 0.05, 0.10]) # weaker policy transmission in regime 1
    cfg.sigma_x = np.array([0.40, 0.60, 0.50])

    # Policy rule shifts
    cfg.psi_pi  = np.array([1.5, 1.1, 1.4])    # less hawkish in regime 1
    cfg.psi_x   = np.array([0.4, 0.2, 0.3])
    cfg.sigma_i = np.array([0.10, 0.15, 0.12])

    # SV mean levels (log variance): higher in regime 1 (high uncertainty) and 2 (supply shocks)
    cfg.sv_mu = np.array([-1.2, -0.3, -0.6])
    return cfg


def _discriminating_config(T=600, seed=0) -> MacroSimConfig:
    """
    Harder simulation with stronger regime contrast and more pronounced instability.
    This generally increases separability across forecasters/ensemblers.
    """
    cfg = MacroSimConfig(T=T, seed=seed)
    # More frequent switching than baseline, with persistent regimes.
    cfg.P = np.array([
        [0.88, 0.09, 0.03],
        [0.10, 0.82, 0.08],
        [0.06, 0.12, 0.82],
    ])

    # Sharper structural differences.
    cfg.alpha = np.array([0.15, 0.97, 0.45])   # very high persistence in regime 1
    cfg.beta = np.array([0.20, -0.05, 0.12])   # slope weak/negative in regime 1
    cfg.gamma = np.array([0.05, 0.25, 0.95])   # strong supply pass-through in regime 2

    cfg.rho = np.array([0.78, 0.94, 0.72])
    cfg.phi = np.array([0.25, 0.02, 0.08])     # policy transmission nearly broken in regime 1
    cfg.sigma_x = np.array([0.35, 0.95, 0.75])

    cfg.psi_pi = np.array([1.8, 0.7, 1.15])    # weak inflation response in regime 1
    cfg.psi_x = np.array([0.5, 0.1, 0.25])
    cfg.sigma_i = np.array([0.08, 0.25, 0.20])

    cfg.u_rho = 0.70
    cfg.sigma_u = 1.20

    cfg.sv_phi = 0.96
    cfg.sv_tau = 0.18
    cfg.sv_mu = np.array([-1.6, 0.35, 0.10])   # much higher volatility in regimes 1/2
    cfg.base_sigma_pi = 0.45
    return cfg


def simulate_macro_environment(cfg: Optional[MacroSimConfig] = None) -> Dict[str, np.ndarray]:
    """
    Simulates (pi_t, x_t, i_t, u_t) with regime switching and stochastic volatility.

    Returns dict with keys:
      pi, x, i, u, regime (s_t), sigma_pi (time-varying), sv_logvar
    """
    if cfg is None:
        cfg = _default_config()

    rng = np.random.default_rng(cfg.seed)
    T = cfg.T

    # allocate
    s = np.zeros(T, dtype=int)
    pi = np.zeros(T)
    x = np.zeros(T)
    i_rate = np.zeros(T)
    u = np.zeros(T)
    sv_logvar = np.zeros(T)
    sigma_pi = np.zeros(T)

    # init
    s[0] = 0
    sv_logvar[0] = cfg.sv_mu[s[0]]
    sigma_pi[0] = cfg.base_sigma_pi * np.exp(0.5 * sv_logvar[0])

    for t in range(1, T):
        # regime transition
        s[t] = rng.choice(3, p=cfg.P[s[t-1]])

        # supply shock
        u[t] = cfg.u_rho * u[t-1] + cfg.sigma_u * rng.normal()

        # stochastic volatility
        mu_s = cfg.sv_mu[s[t]]
        sv_logvar[t] = mu_s + cfg.sv_phi * (sv_logvar[t-1] - cfg.sv_mu[s[t-1]]) + cfg.sv_tau * rng.normal()
        sigma_pi[t] = cfg.base_sigma_pi * np.exp(0.5 * sv_logvar[t])

        # policy rule uses current pi,x contemporaneously (simple reduced-form)
        eps_i = cfg.sigma_i[s[t]] * rng.normal()

        # output gap shock
        eps_x = cfg.sigma_x[s[t]] * rng.normal()

        # output gap depends on lagged real rate proxy
        real_rate_lag = i_rate[t-1] - pi[t-1]
        x[t] = cfg.rho[s[t]] * x[t-1] - cfg.phi[s[t]] * real_rate_lag + eps_x

        # inflation shock
        eps_pi = sigma_pi[t] * rng.normal()
        pi[t] = cfg.alpha[s[t]] * pi[t-1] + cfg.beta[s[t]] * x[t-1] + cfg.gamma[s[t]] * u[t] + eps_pi

        # now policy rate given pi_t and x_t (contemporaneous)
        i_rate[t] = cfg.psi_pi[s[t]] * pi[t] + cfg.psi_x[s[t]] * x[t] + eps_i

    # burn-in
    b = cfg.burn_in
    out = {
        "pi": pi[b:],
        "x": x[b:],
        "i": i_rate[b:],
        "u": u[b:],
        "regime": s[b:],
        "sigma_pi": sigma_pi[b:],
        "sv_logvar": sv_logvar[b:]
    }
    return out


# -----------------------------
# 2) Forecaster panel generator
# -----------------------------

def _ols_fit(X: np.ndarray, y: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Stable ridge OLS: beta = (X'X + ridge I)^(-1) X'y."""
    XtX = X.T @ X
    k = XtX.shape[0]
    beta = np.linalg.solve(XtX + ridge * np.eye(k), X.T @ y)
    return beta


def _recursive_h_step_forecast(
    pi_hist: np.ndarray,
    x_hist: np.ndarray,
    u_hist: np.ndarray,
    i_hist: np.ndarray,
    model: str,
    h: int,
    params: Dict[str, np.ndarray],
    last_state: Optional[int] = None
) -> float:
    """
    Deterministic h-step forecast from a fitted linear reduced-form using recursion.

    model choices:
      - "AR1"                 pi_t = c + a*pi_{t-1}
      - "ARX"                 pi_t = c + a*pi_{t-1} + b*x_{t-1}
      - "ARU"                 pi_t = c + a*pi_{t-1} + g*u_t  (uses u_t as contemporaneous proxy)
      - "PHILLIPS"            pi_t = c + a*pi_{t-1} + b*x_{t-1} + g*u_t
      - "ORACLE_STATE"        uses regime-specific true parameters (requires last_state)
    """
    # latest observed values at time t
    pi_t = float(pi_hist[-1])
    x_t = float(x_hist[-1])
    u_t = float(u_hist[-1])
    i_t = float(i_hist[-1])

    if model == "ORACLE_STATE":
        if last_state is None:
            raise ValueError("ORACLE_STATE requires last_state.")
        # use true regime-specific parameters, hold regime fixed for h-step (simple oracle)
        a = float(params["alpha"][last_state])
        b = float(params["beta"][last_state])
        g = float(params["gamma"][last_state])
        # ignore future shocks (mean zero); hold x and u fixed at last observed values (simple)
        pi_fc = pi_t
        for _ in range(h):
            pi_fc = a * pi_fc + b * x_t + g * u_t
        return float(pi_fc)

    # fitted coefficients
    coef = params["coef"]
    # unpack depending on model
    # Always include intercept
    def one_step(pi_lag, x_lag, u_now):
        if model == "AR1":
            # [1, pi_{t-1}]
            return coef[0] + coef[1] * pi_lag
        if model == "ARX":
            # [1, pi_{t-1}, x_{t-1}]
            return coef[0] + coef[1] * pi_lag + coef[2] * x_lag
        if model == "ARU":
            # [1, pi_{t-1}, u_t]
            return coef[0] + coef[1] * pi_lag + coef[2] * u_now
        if model == "PHILLIPS":
            # [1, pi_{t-1}, x_{t-1}, u_t]
            return coef[0] + coef[1] * pi_lag + coef[2] * x_lag + coef[3] * u_now
        raise ValueError(f"Unknown model: {model}")

    # recursion: use forecasted pi as next lag; keep x,u fixed (simple and standard in forecast comps)
    pi_fc = pi_t
    for _ in range(h):
        pi_fc = one_step(pi_fc, x_t, u_t)
    return float(pi_fc)

def build_forecaster_panel(
    data: Dict[str, np.ndarray],
    horizons: List[int],
    window: int = 120,
    include_oracle: bool = True,
    seed: int = 0,
    forecaster_noise_std: float = 0.0,
    forecaster_bias_std: float = 0.0,
    oracle_params_cfg: Optional[MacroSimConfig] = None,
) -> Tuple[Dict[int, np.ndarray], List[str]]:
    """
    Creates a richer panel of N heterogeneous forecasters producing h-step inflation forecasts.

    Observables available to forecasters at time t:
      - inflation  pi_t
      - output gap x_t
      - supply     u_t
      - policy     i_t
      - (latent true regime s_t is *not* observed, except by oracle forecasters if enabled)

    Forecaster set (heterogeneous + deliberately mis-specified variants):
      Core linear families:
        1)  AR1                        pi_t ~ 1 + pi_{t-1}
        2)  AR2                        pi_t ~ 1 + pi_{t-1} + pi_{t-2}
        3)  ARX                        pi_t ~ 1 + pi_{t-1} + x_{t-1}
        4)  ARX2                       pi_t ~ 1 + pi_{t-1} + pi_{t-2} + x_{t-1}
        5)  ARU                        pi_t ~ 1 + pi_{t-1} + u_t
        6)  PHILLIPS                   pi_t ~ 1 + pi_{t-1} + x_{t-1} + u_t
        7)  TAYLORX                    pi_t ~ 1 + pi_{t-1} + x_{t-1} + (i_{t-1}-pi_{t-1})
        8)  FULL                       pi_t ~ 1 + pi_{t-1}+pi_{t-2} + x_{t-1} + u_t + (i_{t-1}-pi_{t-1})

      Window / regularisation heterogeneity:
        9)  AR1_short                  AR1 estimated on shorter window (more reactive)
        10) AR1_long                   AR1 estimated on longer window (more inertial)
        11) PHILLIPS_short             PHILLIPS on shorter window
        12) PHILLIPS_long              PHILLIPS on longer window
        13) FULL_ridge_strong           FULL with stronger ridge shrinkage
        14) ARX_ridge_strong            ARX with stronger ridge

      Naive / sticky / damped heuristics:
        15) LAST                        pi_{t+h|t} = pi_t
        16) MEAN_12                     pi_{t+h|t} = average(pi_{t-11:t})
        17) DRIFT_12                    pi_{t+h|t} = pi_t + h*(pi_t - mean(pi_{t-11:t}))/12
        18) EXP_SMOOTH                  pi_{t+h|t} = EWMA(pi)_t  (constant smoothing)

      Optional oracle upper bounds:
        19) ORACLE_STATE                uses true regime coefficients, holds regime fixed across horizon
        20) ORACLE_TVP                  (optional approximation) regime-aware + uses current x_t,u_t deterministically

    Returns:
      forecast_panel: dict {h: array(T, N)} with NaNs where not enough history or t+h out of range
      names: list of forecaster names (length N)

    Extra knobs to create more discriminating panels:
      - forecaster_noise_std: idiosyncratic forecast noise scale (per forecaster and horizon)
      - forecaster_bias_std: persistent forecaster bias scale (+ regime-loading bias)
    """
    rng = np.random.default_rng(seed)

    pi = data["pi"]
    x = data["x"]
    u = data["u"]
    i_rate = data["i"]
    regime = data["regime"]
    T = len(pi)

    # helper OLS
    def _ols_fit(X: np.ndarray, y: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
        XtX = X.T @ X
        k = XtX.shape[0]
        return np.linalg.solve(XtX + ridge * np.eye(k), X.T @ y)

    def _recursive_h_step_forecast_linear(coef: np.ndarray, model: str, h: int,
                                          pi_t: float, pi_tm1: float, x_tm1: float, u_t: float, rr_tm1: float) -> float:
        """
        Recursively forecast pi h-steps ahead, holding (x,u,rr) fixed at last observed values.
        """
        def one_step(p_lag, p_lag2):
            if model == "AR1":
                return coef[0] + coef[1] * p_lag
            if model == "AR2":
                return coef[0] + coef[1] * p_lag + coef[2] * p_lag2
            if model == "ARX":
                return coef[0] + coef[1] * p_lag + coef[2] * x_tm1
            if model == "ARX2":
                return coef[0] + coef[1] * p_lag + coef[2] * p_lag2 + coef[3] * x_tm1
            if model == "ARU":
                return coef[0] + coef[1] * p_lag + coef[2] * u_t
            if model == "PHILLIPS":
                return coef[0] + coef[1] * p_lag + coef[2] * x_tm1 + coef[3] * u_t
            if model == "TAYLORX":
                return coef[0] + coef[1] * p_lag + coef[2] * x_tm1 + coef[3] * rr_tm1
            if model == "FULL":
                return coef[0] + coef[1] * p_lag + coef[2] * p_lag2 + coef[3] * x_tm1 + coef[4] * u_t + coef[5] * rr_tm1
            raise ValueError(model)

        p1, p2 = pi_t, pi_tm1
        for _ in range(h):
            p_next = one_step(p1, p2)
            p2, p1 = p1, p_next
        return float(p1)

    # Oracle should use the same structural parameters as the simulated scenario.
    cfg_true = oracle_params_cfg if oracle_params_cfg is not None else _default_config(T=10, seed=0)
    true_params = {
        "alpha": cfg_true.alpha,
        "beta": cfg_true.beta,
        "gamma": cfg_true.gamma,
    }

    def _oracle_state_forecast(h: int, t: int) -> float:
        st = int(regime[t])
        a = float(true_params["alpha"][st])
        b = float(true_params["beta"][st])
        g = float(true_params["gamma"][st])
        # deterministic recursion holding x,u fixed
        p = float(pi[t])
        for _ in range(h):
            p = a * p + b * float(x[t]) + g * float(u[t])
        return p

    # heuristic forecasters
    def _ewma(series: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        out = np.zeros_like(series)
        out[0] = series[0]
        for t in range(1, len(series)):
            out[t] = alpha * series[t] + (1 - alpha) * out[t-1]
        return out

    ewma_pi = _ewma(pi, alpha=0.2)

    # Forecaster menu (name, type, params)
    forecasters = [
        ("AR1", "linear", {"model": "AR1", "window": window, "ridge": 1e-6}),
        ("AR2", "linear", {"model": "AR2", "window": window, "ridge": 1e-6}),
        ("ARX", "linear", {"model": "ARX", "window": window, "ridge": 1e-6}),
        ("ARX2", "linear", {"model": "ARX2", "window": window, "ridge": 1e-6}),
        ("ARU", "linear", {"model": "ARU", "window": window, "ridge": 1e-6}),
        ("PHILLIPS", "linear", {"model": "PHILLIPS", "window": window, "ridge": 1e-6}),
        ("TAYLORX", "linear", {"model": "TAYLORX", "window": window, "ridge": 1e-6}),
        ("FULL", "linear", {"model": "FULL", "window": window, "ridge": 1e-6}),

        ("AR1_short", "linear", {"model": "AR1", "window": max(30, window // 3), "ridge": 1e-6}),
        ("AR1_long", "linear", {"model": "AR1", "window": int(window * 1.8), "ridge": 1e-6}),
        ("PHILLIPS_short", "linear", {"model": "PHILLIPS", "window": max(30, window // 3), "ridge": 1e-6}),
        ("PHILLIPS_long", "linear", {"model": "PHILLIPS", "window": int(window * 1.8), "ridge": 1e-6}),
        ("FULL_ridge_strong", "linear", {"model": "FULL", "window": window, "ridge": 1e-2}),
        ("ARX_ridge_strong", "linear", {"model": "ARX", "window": window, "ridge": 1e-2}),

        ("LAST", "heur", {}),
        ("MEAN_12", "heur", {}),
        ("DRIFT_12", "heur", {}),
        ("EXP_SMOOTH", "heur", {}),
    ]

    if include_oracle:
        forecasters.append(("ORACLE_STATE", "oracle", {}))

    names = [n for n, _, _ in forecasters]
    N = len(names)
    forecast_panel: Dict[int, np.ndarray] = {h: np.full((T, N), np.nan) for h in horizons}
    forecaster_bias = rng.normal(0.0, forecaster_bias_std, size=N)
    regime_loading = rng.normal(0.0, 0.6 * forecaster_bias_std, size=N)

    # cache fitted coefs per t for each (model, window, ridge) to avoid refitting duplicates
    # key: (t, model, window, ridge)
    coef_cache: Dict[Tuple[int, str, int, float], np.ndarray] = {}

    for t in range(T):
        # need enough room for the largest horizon
        if t + max(horizons) >= T:
            continue

        # precompute heuristic components at time t
        pi_t = float(pi[t])
        pi_tm1 = float(pi[t-1]) if t >= 1 else pi_t
        x_tm1 = float(x[t-1]) if t >= 1 else float(x[t])
        u_t = float(u[t])
        rr_tm1 = float(i_rate[t-1] - pi[t-1]) if t >= 1 else float(i_rate[t] - pi[t])

        for j, (name, ftype, pars) in enumerate(forecasters):
            for h in horizons:
                if t + h >= T:
                    continue
                fc = None

                if ftype == "oracle":
                    fc = _oracle_state_forecast(h, t)
                elif ftype == "heur":
                    if name == "LAST":
                        fc = pi_t
                    elif name == "MEAN_12":
                        if t >= 11:
                            fc = float(np.mean(pi[t-11:t+1]))
                    elif name == "DRIFT_12":
                        if t >= 11:
                            m = float(np.mean(pi[t-11:t+1]))
                            fc = pi_t + h * (pi_t - m) / 12.0
                    elif name == "EXP_SMOOTH":
                        fc = float(ewma_pi[t])
                else:
                    # linear models: rolling OLS
                    model = pars["model"]
                    win = int(pars["window"])
                    ridge = float(pars["ridge"])

                    if t < max(win, 5) + 2:
                        continue
                    start = max(0, t - win)
                    # train to predict pi_{τ} for τ in (start+2,...,t) using lags aligned
                    # Build arrays for τ = start+2,...,t:
                    # y = pi_τ
                    # pi_{τ-1}, pi_{τ-2}, x_{τ-1}, u_τ, rr_{τ-1}
                    y_train = pi[start+2:t+1]
                    pi_l1 = pi[start+1:t]
                    pi_l2 = pi[start:t-1]
                    x_l1 = x[start+1:t]
                    u_now = u[start+2:t+1]
                    rr_l1 = (i_rate[start+1:t] - pi[start+1:t])

                    L = len(y_train)
                    if L < 20:
                        continue

                    key = (t, model, win, ridge)
                    if key not in coef_cache:
                        if model == "AR1":
                            X = np.column_stack([np.ones(L), pi_l1])
                        elif model == "AR2":
                            X = np.column_stack([np.ones(L), pi_l1, pi_l2])
                        elif model == "ARX":
                            X = np.column_stack([np.ones(L), pi_l1, x_l1])
                        elif model == "ARX2":
                            X = np.column_stack([np.ones(L), pi_l1, pi_l2, x_l1])
                        elif model == "ARU":
                            X = np.column_stack([np.ones(L), pi_l1, u_now])
                        elif model == "PHILLIPS":
                            X = np.column_stack([np.ones(L), pi_l1, x_l1, u_now])
                        elif model == "TAYLORX":
                            X = np.column_stack([np.ones(L), pi_l1, x_l1, rr_l1])
                        elif model == "FULL":
                            X = np.column_stack([np.ones(L), pi_l1, pi_l2, x_l1, u_now, rr_l1])
                        else:
                            raise ValueError(model)

                        coef_cache[key] = _ols_fit(X, y_train, ridge=ridge)

                    coef = coef_cache[key]
                    fc = _recursive_h_step_forecast_linear(
                        coef=coef,
                        model=model,
                        h=h,
                        pi_t=pi_t,
                        pi_tm1=pi_tm1,
                        x_tm1=x_tm1,
                        u_t=u_t,
                        rr_tm1=rr_tm1
                    )

                if fc is None:
                    continue

                # Optional bias/noise to create a more discriminating panel.
                if forecaster_bias_std > 0 and ftype != "oracle":
                    fc = fc + forecaster_bias[j] + regime_loading[j] * (float(regime[t]) - 1.0)
                if forecaster_noise_std > 0 and ftype != "oracle":
                    fc = fc + rng.normal(0.0, forecaster_noise_std * np.sqrt(float(h)))

                forecast_panel[h][t, j] = float(fc)

    return forecast_panel, names

# -----------------------------
# 3) End-to-end helper
# -----------------------------

def make_environment_and_forecasts(
    T: int = 600,
    horizons: List[int] = [1, 4, 8],
    window: int = 120,
    seed: int = 0,
    include_oracle: bool = True,
    scenario: Literal["baseline", "discriminating"] = "baseline",
    forecaster_noise_std: Optional[float] = None,
    forecaster_bias_std: Optional[float] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[int, np.ndarray], List[str], np.ndarray]:
    """
    Returns:
      data: macro series dict
      forecasts_by_h: dict {h: (T,N)} forecast matrices (with NaNs early/late)
      names: forecaster names (N,)
      state_uncertainty: series s_t you can feed into lambda_t = kappa * s_t
                         (here: sigma_pi, scaled to mean 1)
    """
    if scenario == "baseline":
        cfg = _default_config(T=T, seed=seed)
        if forecaster_noise_std is None:
            forecaster_noise_std = 0.0
        if forecaster_bias_std is None:
            forecaster_bias_std = 0.0
    elif scenario == "discriminating":
        cfg = _discriminating_config(T=T, seed=seed)
        if forecaster_noise_std is None:
            forecaster_noise_std = 0.08
        if forecaster_bias_std is None:
            forecaster_bias_std = 0.12
    else:
        raise ValueError("scenario must be 'baseline' or 'discriminating'")

    data = simulate_macro_environment(cfg)
    forecasts_by_h, names = build_forecaster_panel(
        data,
        horizons=horizons,
        window=window,
        include_oracle=include_oracle,
        seed=seed,
        forecaster_noise_std=float(forecaster_noise_std),
        forecaster_bias_std=float(forecaster_bias_std),
        oracle_params_cfg=cfg,
    )

    # uncertainty state: use sigma_pi scaled to mean 1 (so kappa interpretable)
    s_unc = data["sigma_pi"].copy()
    s_unc = s_unc / np.mean(s_unc)

    return data, forecasts_by_h, names, s_unc


# -----------------------------
# Example run
# -----------------------------
if __name__ == "__main__":
    data, forecasts_by_h, names, s_unc = make_environment_and_forecasts(
        T=800, horizons=[1, 4, 8], window=150, seed=42, include_oracle=True
    )
    print("Series:", {k: v.shape for k, v in data.items()})
    for h, F in forecasts_by_h.items():
        print(f"H={h}: forecasts matrix {F.shape}, NaN frac={np.isnan(F).mean():.3f}")
    print("Forecasters:", names)
    print("Uncertainty state s_unc:", s_unc.shape, "mean", s_unc.mean(), "min", s_unc.min(), "max", s_unc.max())

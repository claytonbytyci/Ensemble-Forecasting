from __future__ import annotations

import math
from typing import Optional

import numpy as np


def select_block_size(T: int) -> int:
    """Heuristic block size for MBB: ceil(T^(1/3)), capped at T//4."""
    return max(1, min(math.ceil(T ** (1.0 / 3.0)), T // 4))


def moving_block_bootstrap(
    d: np.ndarray,
    block_size: int,
    n_boot: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Moving block bootstrap for a 1-D time series of length T.

    Draws ceil(T/block_size) overlapping blocks of length block_size
    with replacement, trims back to length T.

    Returns array of shape (n_boot, T).
    """
    if rng is None:
        rng = np.random.default_rng()
    d = np.asarray(d, dtype=float)
    T = len(d)
    if not 1 <= block_size <= T:
        raise ValueError(f"block_size must be in [1, {T}], got {block_size}")

    n_blocks = math.ceil(T / block_size)
    n_starts = T - block_size + 1
    starts = rng.integers(0, n_starts, size=(n_boot, n_blocks))

    offsets = np.arange(block_size)
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, n_blocks * block_size)[:, :T]
    return d[indices]


def _newey_west_lrv(d: np.ndarray, max_lag: Optional[int] = None) -> float:
    """
    Newey-West long-run variance estimate for a scalar mean statistic.
    LRV = gamma_0 + 2 * sum_{k=1}^{L} (1 - k/(L+1)) * gamma_k
    """
    d = d[np.isfinite(d)]
    T = len(d)
    if T == 0:
        return np.nan
    if max_lag is None:
        max_lag = math.ceil(T ** (1.0 / 3.0))

    dc = d - np.mean(d)
    gamma0 = float(np.mean(dc ** 2))
    lrv = gamma0
    for k in range(1, min(max_lag + 1, T)):
        gamma_k = float(np.mean(dc[k:] * dc[:-k]))
        w = 1.0 - k / (max_lag + 1.0)
        lrv += 2.0 * w * gamma_k
    return max(lrv, 1e-16)


def dm_test(
    loss_ensemble: np.ndarray,
    loss_benchmark: np.ndarray,
    max_lag: Optional[int] = None,
) -> dict:
    """
    Diebold-Mariano test for H0: E[d_t] = 0, d_t = loss_benchmark - loss_ensemble.
    One-sided alternative: ensemble is better (E[d_t] > 0).

    Returns dict: stat, p_value, mean_d, T, se.
    """
    from scipy import stats as sp_stats

    d = np.asarray(loss_benchmark, dtype=float) - np.asarray(loss_ensemble, dtype=float)
    m = np.isfinite(d)
    d = d[m]
    T = len(d)
    if T == 0:
        return {"stat": np.nan, "p_value": np.nan, "mean_d": np.nan, "T": 0}

    lrv = _newey_west_lrv(d, max_lag=max_lag)
    se = math.sqrt(lrv / T)
    stat = float(np.mean(d)) / se
    p_value = float(sp_stats.norm.sf(stat))  # one-sided: P(Z > stat) under H0
    return {"stat": stat, "p_value": p_value, "mean_d": float(np.mean(d)), "T": T, "se": se}


def block_bootstrap_improvement_ci(
    loss_ensemble: np.ndarray,
    loss_benchmark: np.ndarray,
    block_size: Optional[int] = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Block-bootstrap CI for improvement% = 100 * mean(d_t) / mean(loss_benchmark_t),
    where d_t = loss_benchmark_t - loss_ensemble_t.

    Uses percentile bootstrap; p-value is the bootstrap fraction with mean(d*) <= 0
    (one-sided H0: ensemble is not better than benchmark).

    Returns dict: estimate, ci_lower, ci_upper, p_value, block_size, n_boot, T.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    l_ens = np.asarray(loss_ensemble, dtype=float)
    l_ben = np.asarray(loss_benchmark, dtype=float)
    m = np.isfinite(l_ens) & np.isfinite(l_ben)
    l_ens, l_ben = l_ens[m], l_ben[m]
    d = l_ben - l_ens
    T = len(d)

    if block_size is None:
        block_size = select_block_size(T)

    bench_mean = float(np.mean(l_ben))
    if abs(bench_mean) < 1e-12:
        raise ValueError("benchmark mean is near zero, cannot compute improvement %")

    estimate = 100.0 * float(np.mean(d)) / abs(bench_mean)

    boot_d = moving_block_bootstrap(d, block_size=block_size, n_boot=n_boot, rng=rng)
    boot_means = np.mean(boot_d, axis=1)
    boot_theta = 100.0 * boot_means / abs(bench_mean)

    ci_lower = float(np.percentile(boot_theta, 100.0 * alpha / 2))
    ci_upper = float(np.percentile(boot_theta, 100.0 * (1.0 - alpha / 2)))
    p_value = float(np.mean(boot_means <= 0.0))

    return {
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "block_size": block_size,
        "n_boot": n_boot,
        "T": T,
        "mean_d": float(np.mean(d)),
        "bench_mean": bench_mean,
    }


def block_size_sensitivity(
    loss_ensemble: np.ndarray,
    loss_benchmark: np.ndarray,
    block_sizes: Optional[list] = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> list[dict]:
    """Run block_bootstrap_improvement_ci for several block sizes and return list of results."""
    l_ens = np.asarray(loss_ensemble, dtype=float)
    l_ben = np.asarray(loss_benchmark, dtype=float)
    T = int(np.sum(np.isfinite(l_ens) & np.isfinite(l_ben)))

    if block_sizes is None:
        default = select_block_size(T)
        block_sizes = sorted({max(1, default - 1), default, default + 1, default + 2})

    rows = []
    for bs in block_sizes:
        rng = np.random.default_rng(seed)
        r = block_bootstrap_improvement_ci(
            l_ens, l_ben, block_size=bs, n_boot=n_boot, alpha=alpha, rng=rng
        )
        rows.append(r)
    return rows

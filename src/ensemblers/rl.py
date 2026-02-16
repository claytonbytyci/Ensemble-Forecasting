# rl_ensemblers.py

import numpy as np
from ensemblers import (
    MeanCombiner,
    MedianEnsembler,
    OGDCombiner,
    MWUMCombiner,
    OGDConcentrationCombiner,
    MWUMConcentrationCombiner,
)

class RuleSelectionBandit:
    """
    Contextual bandit over ensemble RULES.
    Each action is a full weighting method.
    """

    def __init__(self, n_forecasters, state_dim, alpha=0.5):
        self.alpha = alpha
        self.state_dim = state_dim

        # --- Define available actions ---
        self.actions = [
            MeanCombiner(n_forecasters),
            OGDCombiner(n_forecasters),
            MWUMCombiner(n_forecasters),
            OGDConcentrationCombiner(n_forecasters),
            MWUMConcentrationCombiner(n_forecasters),
        ]

        self.K = len(self.actions)

        # LinUCB parameters
        self.A = [np.eye(state_dim) for _ in range(self.K)]
        self.b = [np.zeros(state_dim) for _ in range(self.K)]

    def select_action(self, state):
        """Choose action using UCB rule."""
        scores = []
        for k in range(self.K):
            A_inv = np.linalg.inv(self.A[k])
            theta = A_inv @ self.b[k]

            mean = state @ theta
            uncertainty = self.alpha * np.sqrt(state @ A_inv @ state)

            scores.append(mean + uncertainty)

        return np.argmax(scores)

    def update(self, k, state, reward):
        """Update linear model for chosen action."""
        self.A[k] += np.outer(state, state)
        self.b[k] += reward * state

    def step(self, forecasts, y_true, state):
        k = self.select_action(state)

        weights = self.actions[k].update(forecasts, y_true)
        forecast = weights @ forecasts

        loss = (y_true - forecast) ** 2
        reward = -loss

        self.update(k, state, reward)

        return forecast, weights, k

class KappaBandit:
    """
    Contextual bandit over κ (degree of Brainard conservatism).
    """

    def __init__(self, kappa_grid, state_dim, alpha=0.5):
        self.kappa_grid = kappa_grid
        self.K = len(kappa_grid)
        self.alpha = alpha
        self.state_dim = state_dim

        self.A = [np.eye(state_dim) for _ in range(self.K)]
        self.b = [np.zeros(state_dim) for _ in range(self.K)]

    def softmax_weights(self, losses, lambda_t):
        scaled = -losses / lambda_t
        scaled -= np.max(scaled)
        exp_vals = np.exp(scaled)
        return exp_vals / np.sum(exp_vals)

    def select_action(self, state):
        scores = []
        for k in range(self.K):
            A_inv = np.linalg.inv(self.A[k])
            theta = A_inv @ self.b[k]

            mean = state @ theta
            uncertainty = self.alpha * np.sqrt(state @ A_inv @ state)

            scores.append(mean + uncertainty)

        return np.argmax(scores)

    def update_bandit(self, k, state, reward):
        self.A[k] += np.outer(state, state)
        self.b[k] += reward * state

    def step(self, forecasts, losses, y_true, state, s_t):
        k = self.select_action(state)
        kappa = self.kappa_grid[k]

        lambda_t = max(1e-6, kappa * s_t)

        weights = self.softmax_weights(losses, lambda_t)
        forecast = weights @ forecasts

        loss = (y_true - forecast) ** 2
        reward = -loss

        self.update_bandit(k, state, reward)

        return forecast, weights, kappa

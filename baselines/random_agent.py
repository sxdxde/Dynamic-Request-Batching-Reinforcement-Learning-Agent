"""
baselines/random_agent.py

Baseline agent that selects actions uniformly at random.
Useful as a lower-bound reference in comparative evaluations.
"""

import numpy as np


class RandomAgent:
    """Uniformly random action baseline.

    Parameters
    ----------
    seed : int | None
        RNG seed for reproducibility.
    """

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray) -> int:
        """Return a random action (0=Wait or 1=Serve) with equal probability.

        Parameters
        ----------
        obs : np.ndarray
            Observation from the environment (ignored by this agent).

        Returns
        -------
        int
            0 or 1, chosen uniformly at random.
        """
        return int(self._rng.integers(0, 2))  # {0, 1}

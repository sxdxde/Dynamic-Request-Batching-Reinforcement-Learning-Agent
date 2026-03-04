"""
baselines/greedy_agent.py

Greedy baseline: always choose action=1 (Serve) regardless of state.
This maximises throughput but ignores batching efficiency entirely,
serving every decision step even when the queue is empty.
"""

import numpy as np


class GreedyAgent:
    """Always-serve greedy baseline.

    Parameters
    ----------
    None — this agent is fully deterministic and stateless.
    """

    def predict(self, obs: np.ndarray) -> int:
        """Always return action=1 (Serve).

        Parameters
        ----------
        obs : np.ndarray
            Observation from the environment (ignored by this agent).

        Returns
        -------
        int
            Always 1 (Serve).
        """
        return 1

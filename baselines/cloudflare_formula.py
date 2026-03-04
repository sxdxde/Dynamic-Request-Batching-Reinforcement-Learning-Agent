"""
baselines/cloudflare_formula.py

Baseline agent inspired by Cloudflare's batching heuristic:

    p(serve) = exp(-lambda * remaining_time)

where
    lambda         = observed arrival rate (req/s)
    remaining_time = (max_latency_ms - oldest_wait_ms) / 1000  (seconds)

The agent serves if a uniform random draw < p(serve), giving a
probabilistic policy that becomes more aggressive as the SLA deadline
approaches.

Also provides evaluate_baseline() — a shared evaluation utility used
by all baseline agents.
"""

import math
import numpy as np

# Observation indices (must match BatchingEnv._get_obs())
_IDX_PENDING       = 0
_IDX_OLDEST_WAIT   = 1
_IDX_RATE          = 2
_IDX_SINCE_SERVE   = 3
_IDX_FILL_RATIO    = 4
_IDX_TIME_OF_DAY   = 5


class CloudflareBaseline:
    """Probabilistic serve baseline: p(serve) = exp(-λ · remaining_time).

    Parameters
    ----------
    max_latency_ms : float
        SLA deadline in milliseconds (default 500).
    seed : int | None
        RNG seed for the Bernoulli draw.
    """

    def __init__(self, max_latency_ms: float = 500.0, seed: int | None = None):
        self.max_latency_ms = max_latency_ms
        self._rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray) -> int:
        """Select action given observation vector.

        Parameters
        ----------
        obs : np.ndarray
            Shape (6,) observation from BatchingEnv.

        Returns
        -------
        int
            0 = Wait, 1 = Serve.
        """
        oldest_wait_ms = float(obs[_IDX_OLDEST_WAIT])
        rate_req_per_s = float(obs[_IDX_RATE])

        # If nothing is pending, always wait
        pending = int(obs[_IDX_PENDING])
        if pending == 0:
            return 0

        remaining_ms = self.max_latency_ms - oldest_wait_ms
        remaining_s  = remaining_ms / 1000.0

        # Clamp: if already past deadline, serve immediately
        if remaining_s <= 0:
            return 1

        p_serve = math.exp(-rate_req_per_s * remaining_s)

        # Stochastic draw
        return int(self._rng.random() < p_serve)


# ---------------------------------------------------------------------------
# Shared evaluation utility
# ---------------------------------------------------------------------------

def evaluate_baseline(
    baseline,
    env,
    n_episodes: int = 20,
    seed_offset: int = 0,
) -> tuple[float, float, float]:
    """Run a baseline for *n_episodes* and return aggregate metrics.

    Parameters
    ----------
    baseline :
        Any object with a ``predict(obs) -> int`` method.
    env :
        A BatchingEnv (or compatible gym.Env) instance.
    n_episodes : int
        Number of complete episodes to evaluate.
    seed_offset : int
        Added to the per-episode seed so different agents get the same
        traffic patterns when called with the same offset (fair comparison).

    Returns
    -------
    mean_reward : float
    std_reward  : float
    mean_latency_ms : float
    """
    episode_rewards: list[float] = []
    episode_latencies: list[float] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        total_reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action = baseline.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        episode_rewards.append(total_reward)
        episode_latencies.append(info.get("mean_latency_ms", 0.0))

    mean_reward   = float(np.mean(episode_rewards))
    std_reward    = float(np.std(episode_rewards))
    mean_latency  = float(np.mean(episode_latencies))
    return mean_reward, std_reward, mean_latency

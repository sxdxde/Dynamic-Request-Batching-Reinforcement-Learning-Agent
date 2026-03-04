"""
env/batching_env.py

Gymnasium environment for Dynamic Request Batching.

State space (6-dim Box):
    [0] pending_requests   – number of requests currently queued
    [1] oldest_wait_ms     – age (ms) of the oldest pending request
    [2] request_rate       – exponential moving average of req/s
    [3] time_since_last_serve_ms – ms elapsed since last Serve action
    [4] batch_fill_ratio   – pending_requests / max_batch_size  ∈ [0,1]
    [5] time_of_day        – fractional hour ∈ [0, 24)

Action space: Discrete(2)
    0 = Wait  – do not serve yet, keep accumulating
    1 = Serve – dispatch all pending requests as one batch

Reward:
    r = alpha * batch_size
      - beta  * oldest_wait_ms
      - gamma * idle_penalty          (if action==Wait and queue empty)
      - sla_penalty                   (if oldest_wait_ms > max_latency_ms)

Episode:
    Simulated time of episode_ms milliseconds.
    The agent makes a decision every decision_interval_ms.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import CONFIG
from env.traffic_generator import TrafficGenerator


class BatchingEnv(gym.Env):
    """Dynamic Request Batching Gymnasium environment.

    Parameters
    ----------
    config : dict | None
        Override selected config keys.  Defaults to the global CONFIG.
    render_mode : str | None
        Only ``"human"`` is supported (prints step info to stdout).
    seed : int | None
        RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: dict | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()

        # Merge caller config with defaults
        self.cfg = {**CONFIG, **(config or {})}

        self.render_mode = render_mode
        self._seed = seed

        # ------------------------------------------------------------------ #
        # Observation space
        # ------------------------------------------------------------------ #
        # [pending_requests, oldest_wait_ms, request_rate,
        #  time_since_last_serve_ms, batch_fill_ratio, time_of_day]
        obs_low  = np.array([0,    0,    0,   0,   0.0, 0.0],  dtype=np.float32)
        obs_high = np.array([
            self.cfg["max_batch_size"],          # pending_requests
            self.cfg["max_latency_ms"] * 2,      # oldest_wait_ms (allow overshoot)
            self.cfg["arrival_rate"] * self.cfg["peak_multiplier"] * 2,  # rate
            self.cfg["max_latency_ms"] * 2,      # time_since_last_serve_ms
            1.0,                                  # batch_fill_ratio
            24.0,                                 # time_of_day
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # ------------------------------------------------------------------ #
        # Action space: 0 = Wait, 1 = Serve
        # ------------------------------------------------------------------ #
        self.action_space = spaces.Discrete(2)

        # Internal state (initialised in reset)
        self._rng: np.random.Generator | None = None
        self._traffic: TrafficGenerator | None = None
        self._queue: list[float] = []          # arrival times of each request
        self._sim_time_ms: float = 0.0         # current simulated clock (ms)
        self._last_serve_ms: float = 0.0       # sim_time when last Serve happened
        self._episode_end_ms: float = 0.0
        self._ema_rate: float = 0.0            # EMA of observed req/s

        # Start time-of-day (randomised each episode for diversity)
        self._start_hour: float = 0.0

        # Metrics accumulated for _get_info()
        self._total_served: int = 0
        self._total_batches: int = 0
        self._latency_samples: list[float] = []

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ):
        super().reset(seed=seed)

        # Seeded RNG
        rng_seed = seed if seed is not None else self._seed
        self._rng = np.random.default_rng(rng_seed)

        # Traffic generator with the same RNG stream
        self._traffic = TrafficGenerator(
            base_rate=self.cfg["arrival_rate"],
            peak_hours=tuple(self.cfg["peak_hours"]),
            peak_multiplier=self.cfg["peak_multiplier"],
            offpeak_multiplier=self.cfg["offpeak_multiplier"],
            rng=self._rng,
        )

        # Simulate a random start hour so the agent sees varied time-of-day
        self._start_hour = float(self._rng.uniform(0, 24))
        self._sim_time_ms = 0.0
        self._last_serve_ms = 0.0
        self._episode_end_ms = float(self.cfg["episode_ms"])
        self._queue = []
        self._ema_rate = float(self.cfg["arrival_rate"])
        self._total_served = 0
        self._total_batches = 0
        self._latency_samples = []

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        dt_ms = float(self.cfg["decision_interval_ms"])
        alpha = self.cfg["alpha"]
        beta  = self.cfg["beta"]
        gamma = self.cfg["gamma"]
        sla_penalty = self.cfg["sla_penalty"]
        max_lat = self.cfg["max_latency_ms"]

        # --- 1. Advance simulation clock & generate new arrivals ---------- #
        hour = self._current_hour()
        new_arrivals = self._traffic.arrivals_in_window_ms(dt_ms, hour)

        for _ in range(new_arrivals):
            # Assign a random arrival time within [sim_time_ms, sim_time_ms+dt_ms)
            arrival_offset = self._rng.uniform(0, dt_ms)
            self._queue.append(self._sim_time_ms + arrival_offset)

        # Cap queue at max_batch_size (oldest requests are kept)
        if len(self._queue) > self.cfg["max_batch_size"]:
            self._queue = self._queue[-self.cfg["max_batch_size"]:]

        self._sim_time_ms += dt_ms

        # --- 2. Update EMA of arrival rate -------------------------------- #
        # rate_window = arrivals in this interval → req/s
        observed_rate = (new_arrivals / dt_ms) * 1000.0
        alpha_ema = 0.1  # EMA smoothing factor
        self._ema_rate = alpha_ema * observed_rate + (1 - alpha_ema) * self._ema_rate

        # --- 3. Compute reward -------------------------------------------- #
        reward = 0.0
        batch_size = 0
        oldest_wait = self._oldest_wait_ms()

        # SLA check BEFORE action (oldest request already waiting too long)
        sla_violated = oldest_wait > max_lat and len(self._queue) > 0

        if action == 1:  # Serve
            batch_size = len(self._queue)
            if batch_size > 0:
                # Record latencies
                for arrival_t in self._queue:
                    wait = self._sim_time_ms - arrival_t
                    self._latency_samples.append(wait)
                self._queue.clear()
                self._last_serve_ms = self._sim_time_ms
                self._total_served += batch_size
                self._total_batches += 1

            # Reward: batch efficiency − latency penalty
            reward = alpha * batch_size - beta * oldest_wait

        else:  # Wait
            idle_penalty = 1.0 if len(self._queue) == 0 else 0.0
            reward = -beta * oldest_wait - gamma * idle_penalty

        if sla_violated:
            reward += sla_penalty

        # --- 4. Termination ----------------------------------------------- #
        terminated = self._sim_time_ms >= self._episode_end_ms
        truncated  = False

        obs  = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_step(action, reward, batch_size, oldest_wait)

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(
                f"t={self._sim_time_ms:.0f}ms  queue={len(self._queue)}  "
                f"oldest={self._oldest_wait_ms():.1f}ms  "
                f"rate={self._ema_rate:.2f} req/s"
            )

    def close(self):
        pass

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _get_obs(self) -> np.ndarray:
        """Return the 6-dimensional observation vector."""
        pending  = float(len(self._queue))
        oldest   = self._oldest_wait_ms()
        rate     = self._ema_rate
        t_since  = self._sim_time_ms - self._last_serve_ms
        fill     = pending / self.cfg["max_batch_size"]
        tod      = self._current_hour()

        obs = np.array([pending, oldest, rate, t_since, fill, tod], dtype=np.float32)

        # Clip to observation space bounds
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def _get_info(self) -> dict:
        """Return auxiliary diagnostic information."""
        mean_lat = (
            float(np.mean(self._latency_samples))
            if self._latency_samples
            else 0.0
        )
        return {
            "sim_time_ms": self._sim_time_ms,
            "total_served": self._total_served,
            "total_batches": self._total_batches,
            "mean_latency_ms": mean_lat,
            "queue_length": len(self._queue),
        }

    def _oldest_wait_ms(self) -> float:
        """Age of the oldest pending request in milliseconds."""
        if not self._queue:
            return 0.0
        return float(self._sim_time_ms - self._queue[0])

    def _current_hour(self) -> float:
        """Return the simulated fractional hour of day."""
        elapsed_hours = self._sim_time_ms / (1000.0 * 3600.0)
        return (self._start_hour + elapsed_hours) % 24.0

    def _render_step(
        self, action: int, reward: float, batch_size: int, oldest_wait: float
    ):
        action_label = "SERVE" if action == 1 else "WAIT "
        print(
            f"[{self._sim_time_ms:8.0f} ms] {action_label}  "
            f"batch={batch_size:3d}  oldest={oldest_wait:6.1f}ms  "
            f"reward={reward:+.3f}  queue={len(self._queue):3d}"
        )

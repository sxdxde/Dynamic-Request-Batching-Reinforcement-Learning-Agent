# config.py
# Central configuration for the Dynamic Request Batching RL project

CONFIG = {
    # Environment
    "max_batch_size": 100,       # Maximum number of requests in a batch
    "max_latency_ms": 500,       # SLA: oldest request must not exceed this (ms)
    "arrival_rate": 10,          # Base Poisson arrival rate (requests/second)
    "episode_ms": 60_000,        # Episode length in milliseconds (60 seconds)
    "decision_interval_ms": 10,  # How often the agent makes a decision (ms)

    # Reward shaping
    "alpha": 1.0,    # Reward multiplier for batch size (efficiency)
    "beta": 0.01,    # Penalty multiplier for oldest_wait_ms (latency)
    "gamma": 0.1,    # Penalty for idle (no requests, action=Wait)

    # SLA
    "sla_penalty": -5.0,  # Penalty when oldest request exceeds max_latency_ms

    # Traffic variation (time-of-day)
    "peak_hours": (8, 18),       # 08:00–18:00 → peak traffic window
    "peak_multiplier": 2.5,      # Lambda multiplier during peak hours
    "offpeak_multiplier": 0.5,   # Lambda multiplier during off-peak hours
}


# ───────────────────────────────────────────────────────────────────────────
# Experiment presets — change one key to explore different traffic regimes.
#
# Usage example:
#   from config import EXPERIMENT_CONFIGS
#   env = BatchingEnv(config=EXPERIMENT_CONFIGS["high_load"])
# ───────────────────────────────────────────────────────────────────────────

EXPERIMENT_CONFIGS = {
    # Low-traffic scenario: sparse arrivals, agent must decide whether to wait
    # for a bigger batch or serve immediately to avoid idle-wait penalties.
    # Expected behaviour: agent learns to accumulate more before serving.
    "low_load": {
        **CONFIG,
        "arrival_rate": 5,           # half the default traffic
    },

    # Standard scenario: matches CONFIG exactly — the default training regime.
    "standard": {
        **CONFIG,
        "arrival_rate": 10,
    },

    # High-traffic scenario: backpressure stress-test.  The queue fills fast;
    # the agent must serve frequently to prevent SLA violations while still
    # batching enough to get meaningful efficiency rewards.
    "high_load": {
        **CONFIG,
        "arrival_rate": 50,          # 5× the default; queue saturates quickly
    },
}

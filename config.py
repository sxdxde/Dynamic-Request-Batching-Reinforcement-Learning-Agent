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
# DQN Hyperparameters
# Compare directly against PPO_KWARGS in agent/train.py
# ───────────────────────────────────────────────────────────────────────────

DQN_CONFIG = {
    # Learning
    "learning_rate": 1e-4,         # DQN is more sensitive to LR than PPO
    "gamma": 0.99,                 # Discount factor — same as PPO for fair comparison
    "batch_size": 64,              # Minibatch size for Q-network updates — same as PPO

    # Replay buffer
    "buffer_size": 100_000,        # How many past (s,a,r,s') transitions to store
                                   # Key DQN innovation — breaks temporal correlations
    "learning_starts": 10_000,     # Steps of random action BEFORE learning begins
                                   # Fills the replay buffer with diverse data first

    # Target network
    "target_update_interval": 1000, # Every N steps, copy Q-network → Target Q-network
                                    # Prevents the Q-value target from "chasing itself"
                                    # (main stability trick of DQN vs vanilla Q-learning)
    "tau": 1.0,                    # 1.0 = hard update (copy weights completely)
                                   # <1.0 = soft/polyak update (blend weights gradually)

    # Exploration (ε-greedy)
    "exploration_fraction": 0.2,   # Fraction of training spent decaying ε
    "exploration_initial_eps": 1.0, # Start fully random (100% explore)
    "exploration_final_eps": 0.05,  # End at 5% random exploration

    # Network
    "net_arch": [64, 64],          # Same hidden layer size as PPO for fair comparison

    # Training
    "train_freq": 4,               # Train the Q-network every N environment steps
    "gradient_steps": 1,           # Gradient update steps per training call

    # Logging
    "verbose": 1,
    "tensorboard_log": "tensorboard_logs/",
}


RPPO_CONFIG = {
    "learning_rate": 3e-4,
    "gamma": 0.99, 
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,

    "lstm_hidden_size": 64,
    "n_lstm_layers": 1,
    "enable_critic_lstm":True,
    "n_steps": 128,
    "batch_size": 64,
    "n_epochs":10,
    "net_arch": dict(pi=[64], vf=[64]),
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

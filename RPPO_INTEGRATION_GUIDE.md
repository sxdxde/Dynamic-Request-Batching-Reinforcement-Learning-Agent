# Integrating Recurrent PPO (LSTM-PPO) into the RL Batching Project

## A Step-by-Step Manual for Adding Recurrent PPO and Comparing Against PPO and DQN

---

## 1. Why Recurrent PPO? The Core Theory

### The Fundamental Limitation of Standard PPO and DQN in Your Project

Both PPO and DQN use a **feedforward** (non-recurrent) neural network as their policy/Q-network. This means at every 10ms decision tick, the network sees **only the current 6-dimensional observation** and produces an action. It has **zero memory** of what happened 100ms, 1 second, or 30 seconds ago.

This is a significant architectural limitation for your specific environment, because your traffic is **sequential and temporally correlated**:

- At t=0ms, `request_rate` = 8 req/s
- At t=100ms, `request_rate` = 12 req/s
- At t=200ms, `request_rate` = 18 req/s

A feedforward network sees each of these 3 observations independently. It cannot infer: *"the rate is rising sharply — a traffic spike is coming in the next 500ms"*. It can only react to the current value, not the trend.

### What Recurrent PPO Adds: LSTM Hidden State

Recurrent PPO replaces the feedforward policy layers with an **LSTM (Long Short-Term Memory)** cell. LSTM maintains two internal vectors between steps:

- **`h_t` (Hidden State):** Short-term working memory — what the network is "currently thinking about"
- **`c_t` (Cell State):** Long-term memory — important patterns from further back in the episode

The recurrent update at each step is:
```
(h_t, c_t)  =  LSTM(observation_t,  h_{t-1},  c_{t-1})
action_prob  =  Actor(h_t)
value        =  Critic(h_t)
```

The LSTM learns which signals to **remember** (traffic rising quickly), which to **forget** (irrelevant fluctuations), and which to **output** (the current decision).

### Why This Specifically Helps Dynamic Request Batching

| Temporal pattern | Standard PPO | Recurrent PPO |
|---|---|---|
| Traffic spike beginning | Can't see it — only sees current rate | h_t encodes "rising trend" |
| Post-peak cool-down | Doesn't know peak just ended | c_t remembers recent high rate |
| Time-of-day shift | Only sees `time_of_day` number | Can learn rhythm of daily cycles |
| Queue buildup rate | Sees current queue size only | Encodes how fast queue is growing |

### How It Differs From PPO and DQN

| Property | PPO | DQN | Recurrent PPO |
|---|---|---|---|
| Network type | Feedforward MLP | Feedforward MLP | LSTM + MLP |
| Memory | None — Markov assumption | None — Markov assumption | Hidden state per episode |
| On/Off-policy | On-policy | Off-policy | On-policy |
| Exploration | Entropy bonus | ε-greedy | Entropy bonus |
| Parallel envs | 4 envs | 1 env | 4 envs |
| Episode-state reset | N/A | N/A | LSTM state resets at episode end |

---

## 2. Understanding the Key Training Difference

This is the most important concept to understand before writing code.

### Standard PPO Training
In standard PPO, experiences `(s, a, r, s')` are collected independently. The network processes each observation on its own. You can shuffle minibatches any way you like.

### Recurrent PPO Training
In Recurrent PPO, the LSTM needs to process observations **in the exact order they happened within an episode**. You cannot shuffle individual time-steps — you must feed entire **sequences** of consecutive steps. SB3's RecurrentPPO handles this automatically using a technique called **sequence-padded minibatches**.

This means:
1. Rollout buffer stores entire episode segments in order
2. During minibatch sampling, full episode sequences are sampled (not individual steps)
3. The LSTM hidden state is reset at episode boundaries (`episode_start = True`)

**SB3 handles all of this for you** — but you need to be aware of it when you write the evaluation code, because you must manually manage the LSTM state during evaluation.

---

## 3. File Structure — What You Will Create

```
Rl project gooooo/
├── agent/
│   ├── train.py              ← already exists (PPO)
│   ├── train_dqn.py          ← already exists (DQN)
│   ├── train_rppo.py         ← NEW — Recurrent PPO training
│   └── evaluate.py           ← MODIFY — add RPPO wrapper + agent
├── config.py                 ← MODIFY — add RPPO config
└── RPPO_INTEGRATION_GUIDE.md ← this file
```

**1 new file, 2 targeted edits to existing files.**

---

## 4. Step 0 — Install `sb3-contrib`

Recurrent PPO is not in the main `stable-baselines3` package. It lives in the official companion package `sb3-contrib`. Run this in your terminal first:

```bash
pip install sb3-contrib
```

Then verify it installed correctly:
```bash
python3 -c "from sb3_contrib import RecurrentPPO; print('sb3-contrib OK')"
```

You should see `sb3-contrib OK`. If you see an error, make sure you are in the correct Python environment.

---

## 5. Step 1 — Add RPPO Config to `config.py`

Open `config.py` and add the following block **after** the existing `DQN_CONFIG` dict:

```python
# ─────────────────────────────────────────────────────────────────────────────
# Recurrent PPO (LSTM-PPO) Hyperparameters
# sb3-contrib RecurrentPPO with MlpLstmPolicy
# ─────────────────────────────────────────────────────────────────────────────

RPPO_CONFIG = {
    # Learning — same as PPO for fair comparison
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,        # Entropy bonus — same as PPO
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,

    # LSTM-specific — the key new parameters
    "lstm_hidden_size": 64,  # Size of the LSTM hidden state vector.
                             # This is the "working memory" of the agent.
                             # Must match your MLP hidden layers for
                             # a fair comparison; we use 64 to match [64,64].

    "n_lstm_layers": 1,      # Number of stacked LSTM cells.
                             # 1 is standard. 2+ adds depth but requires more
                             # data to train and risks vanishing gradients.

    "enable_critic_lstm": True,  # If True, the critic (value network) ALSO
                                 # gets its own LSTM. If False, only the actor
                                 # has LSTM memory. True gives better value
                                 # estimates but uses more memory.

    # Rollout — MUST be adapted for recurrent training
    # RecurrentPPO requires n_steps to be >= the longest episode segment
    # you want the LSTM to learn from. Larger = more temporal context,
    # but also more memory and slower training.
    "n_steps": 128,          # Steps per env per rollout update.
                             # Smaller than PPO's 2048 because RPPO processes
                             # full sequences — 128 * 4 envs = 512 steps total.
    "batch_size": 64,        # Minibatch size (in full sequences)

    "n_epochs": 10,          # Same as PPO

    # Network arch (the MLP layers AFTER the LSTM output)
    "net_arch": dict(pi=[64], vf=[64]),  # Shallower post-LSTM MLP — LSTM
                                         # already does heavy lifting, so one
                                         # layer of 64 is enough here.
}
```

**Why `n_steps=128` instead of PPO's `2048`?**
RecurrentPPO processes observations in temporal sequences. With 4 envs and 128 steps each, you get 512 total transitions per update. The sequences must fit in GPU/CPU memory simultaneously, so smaller is better. The LSTM itself can learn dependencies across hundreds of steps even with smaller rollout windows, because it accumulates context in its hidden state.

---

## 6. Step 2 — Create `agent/train_rppo.py`

Create a new file `agent/train_rppo.py` with the following content:

```python
"""
agent/train_rppo.py

Recurrent PPO (LSTM-PPO) training script for the Dynamic Request Batching agent.

Uses sb3-contrib RecurrentPPO with MlpLstmPolicy:
  - 4 parallel envs (Monitor-wrapped)
  - LSTM hidden state of size 64 — gives the agent sequential memory
  - Actor and Critic both have LSTM cells
  - n_steps=128 per env (sequences for LSTM training)
  - EvalCallback → saves best model to models/rppo_best/
  - CheckpointCallback → saves every 50k steps
  - TensorBoard logging → tensorboard_logs/RPPO_batching/

Key difference from train.py (standard PPO):
  This agent maintains a hidden state (h_t, c_t) between consecutive steps
  within an episode. This allows it to 'remember' past observations — for
  example, the fact that traffic was rising over the last 2 seconds — and
  make more informed batching decisions than a memoryless policy.

  Theoretical motivation: Batching revenue arrivals form a non-stationary
  Poisson process. An agent that can detect trend changes in arrival rate
  can make proactive serving decisions, rather than only reacting to the
  current queue size and oldest wait time.

Usage:
    python agent/train_rppo.py

TensorBoard:
    tensorboard --logdir tensorboard_logs/
    # Compare 'PPO_batching' vs 'DQN_batching' vs 'RPPO_batching'
"""

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from env.batching_env import BatchingEnv
from config import CONFIG, RPPO_CONFIG

# ── Directories ───────────────────────────────────────────────────────────────
TENSORBOARD_DIR        = os.path.join(ROOT, "tensorboard_logs")
MODELS_DIR             = os.path.join(ROOT, "models")
RPPO_BEST_MODEL_DIR    = os.path.join(MODELS_DIR, "rppo_best")
RPPO_CHECKPOINT_DIR    = os.path.join(MODELS_DIR, "rppo_checkpoints")
MONITOR_LOG_DIR        = os.path.join(ROOT, "monitor_logs")
RPPO_FINAL_MODEL_PATH  = os.path.join(MODELS_DIR, "rppo_batching_final")

for d in [RPPO_BEST_MODEL_DIR, RPPO_CHECKPOINT_DIR, MONITOR_LOG_DIR]:
    os.makedirs(d, exist_ok=True)

TOTAL_TIMESTEPS = 500_000
EVAL_FREQ       = 10_000
CHECKPOINT_FREQ = 50_000
N_ENVS          = 4


def train():
    print("=" * 65)
    print("  Dynamic Request Batching — Recurrent PPO (LSTM-PPO) Training")
    print("=" * 65)
    print(f"  Total timesteps     : {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel envs       : {N_ENVS}")
    print(f"  LSTM hidden size    : {RPPO_CONFIG['lstm_hidden_size']}")
    print(f"  LSTM layers         : {RPPO_CONFIG['n_lstm_layers']}")
    print(f"  Critic LSTM         : {RPPO_CONFIG['enable_critic_lstm']}")
    print(f"  n_steps per env     : {RPPO_CONFIG['n_steps']}")
    print(f"  TensorBoard         : {TENSORBOARD_DIR}")
    print("=" * 65)

    # ── Training envs (vectorised + Monitor) ──────────────────────────────────
    # RecurrentPPO works with make_vec_env just like standard PPO.
    # The key difference is INTERNAL: SB3 tracks LSTM states per-env in a
    # VecEnv buffer and resets them automatically when an episode ends.
    train_env = make_vec_env(
        env_id=BatchingEnv,
        n_envs=N_ENVS,
        seed=0,
        wrapper_class=Monitor,
        wrapper_kwargs={"filename": None},
        env_kwargs={},
    )

    # ── Eval env (single, deterministic) ──────────────────────────────────────
    eval_env = Monitor(BatchingEnv(seed=9999))

    # ── Recurrent PPO Model ───────────────────────────────────────────────────
    # MlpLstmPolicy — the new policy type. Under the hood:
    #   1. Observation passes through a small MLP feature extractor
    #   2. Features fed into LSTM cell → produces h_t
    #   3. h_t passed to separate Actor head (policy) and Critic head (value)
    # When enable_critic_lstm=True, the critic gets its own LSTM cell
    # with separate weights — it learns to estimate long-term value
    # using its own temporal context.
    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        learning_rate=RPPO_CONFIG["learning_rate"],
        n_steps=RPPO_CONFIG["n_steps"],
        batch_size=RPPO_CONFIG["batch_size"],
        n_epochs=RPPO_CONFIG["n_epochs"],
        gamma=RPPO_CONFIG["gamma"],
        gae_lambda=RPPO_CONFIG["gae_lambda"],
        clip_range=RPPO_CONFIG["clip_range"],
        ent_coef=RPPO_CONFIG["ent_coef"],
        vf_coef=RPPO_CONFIG["vf_coef"],
        max_grad_norm=RPPO_CONFIG["max_grad_norm"],
        policy_kwargs=dict(
            lstm_hidden_size=RPPO_CONFIG["lstm_hidden_size"],
            n_lstm_layers=RPPO_CONFIG["n_lstm_layers"],
            enable_critic_lstm=RPPO_CONFIG["enable_critic_lstm"],
            net_arch=RPPO_CONFIG["net_arch"],
        ),
        tensorboard_log=TENSORBOARD_DIR,
        verbose=1,
        seed=0,
    )

    print(f"\nRecurrent Policy architecture:\n{model.policy}\n")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=RPPO_BEST_MODEL_DIR,
        log_path=os.path.join(MODELS_DIR, "rppo_eval_logs"),
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path=RPPO_CHECKPOINT_DIR,
        name_prefix="rppo_batching",
        verbose=1,
    )
    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        reset_num_timesteps=True,
        tb_log_name="RPPO_batching",
        progress_bar=True,
    )
    elapsed = time.time() - t0

    # ── Save final model ───────────────────────────────────────────────────────
    model.save(RPPO_FINAL_MODEL_PATH)
    print(f"\n✓  RPPO final model saved → {RPPO_FINAL_MODEL_PATH}.zip")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Recurrent PPO Training Summary")
    print("=" * 65)
    print(f"  Training time     : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Total timesteps   : {TOTAL_TIMESTEPS:,}")
    print(f"  Throughput        : {TOTAL_TIMESTEPS / elapsed:,.0f} steps/s")
    print(f"  Final model       : {RPPO_FINAL_MODEL_PATH}.zip")
    print(f"  Best model        : {RPPO_BEST_MODEL_DIR}/best_model.zip")
    print(f"  TensorBoard log   : {TENSORBOARD_DIR}/RPPO_batching_*/")
    print("=" * 65)

    train_env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    train()
```

---

## 7. Step 3 — Modify `agent/evaluate.py`

Recurrent PPO requires a **special evaluation wrapper** because the `predict()` call has a different signature. During evaluation, you must manually carry the LSTM hidden state forward between steps, and reset it at episode boundaries.

### Edit 1: Add RecurrentPPO to imports (around line 35)

Find:
```python
from stable_baselines3 import PPO, DQN
```

Replace with:
```python
from stable_baselines3 import PPO, DQN
from sb3_contrib import RecurrentPPO
```

---

### Edit 2: Add `"RPPO"` to the colour palette and agent order (around line 58)

Find:
```python
AGENT_COLORS = {
    "PPO":        "#00e5ff",   # cyan
    "DQN":        "#76ff03",   # lime green
    "Greedy":     "#7c4dff",   # purple
    "Cloudflare": "#ff6f00",   # amber
    "Random":     "#f44336",   # red
}
AGENT_ORDER = ["PPO", "DQN", "Greedy", "Cloudflare", "Random"]
```

Replace with:
```python
AGENT_COLORS = {
    "PPO":        "#00e5ff",   # cyan
    "RPPO":       "#e040fb",   # magenta — LSTM-PPO gets a distinct vivid colour
    "DQN":        "#76ff03",   # lime green
    "Greedy":     "#7c4dff",   # purple
    "Cloudflare": "#ff6f00",   # amber
    "Random":     "#f44336",   # red
}
AGENT_ORDER = ["PPO", "RPPO", "DQN", "Greedy", "Cloudflare", "Random"]
```

---

### Edit 3: Add `RecurrentPPOWrapper` class below `DQNWrapper`

This is the most **critical** edit, because Recurrent PPO's `predict()` has a completely different signature compared to feedforward models.

Find the end of your `DQNWrapper` class and add the following right below it:

```python
class RecurrentPPOWrapper:
    """Wrapper for RecurrentPPO that correctly manages LSTM hidden states.

    This wrapper is fundamentally different from PPOWrapper and DQNWrapper.
    Standard feedforward models have a stateless predict() — you just pass
    the observation and get an action.

    RecurrentPPO is stateful: it requires you to track and pass the LSTM
    hidden state (lstm_states) between consecutive steps within the same
    episode, and explicitly tell it when a new episode begins
    (episode_start=True) so it can reset the hidden state to zeros.

    If you forget to pass lstm_states forward, the LSTM resets to zero at
    every step — equivalent to a memoryless policy, completely defeating
    the purpose of the recurrent architecture.

    If you forget to reset lstm_states at episode boundaries, the LSTM
    carries over stale memory from the previous episode into the new one,
    causing incorrect, context-polluted predictions.

    This wrapper handles all of that correctly so the evaluation loop in
    collect_episode_data() does not need to change at all.
    """

    def __init__(self, model: RecurrentPPO):
        self.model = model
        self.lstm_states = None        # Tuple of (h_t, c_t) — starts as None
        self.episode_started = True    # True at first step → resets LSTM to 0

    def reset(self):
        """Call this at the start of each new evaluation episode."""
        self.lstm_states = None
        self.episode_started = True

    def predict(self, obs: np.ndarray) -> int:
        """Predict action, carrying LSTM state forward between calls."""
        import numpy as np_inner

        # episode_starts shape must match n_envs — here n_envs=1 for eval.
        ep_start = np_inner.array([self.episode_started], dtype=bool)

        action, self.lstm_states = self.model.predict(
            obs,
            state=self.lstm_states,
            episode_start=ep_start,
            deterministic=True,
        )
        self.episode_started = False   # Subsequent steps in same episode
        return int(action)
```

---

### Edit 4: Update `collect_episode_data()` to reset RPPO state between episodes

The current `collect_episode_data()` function calls `env.reset()` at the start of each episode. For RPPO, we also need to reset the LSTM state. Add one line inside the episode loop:

Find this block near the top of `collect_episode_data()`:
```python
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        total_reward = 0.0
        terminated = truncated = False
        prev_served = 0
```

Replace with:
```python
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        total_reward = 0.0
        terminated = truncated = False
        prev_served = 0

        # Reset LSTM hidden state at episode boundaries for RecurrentPPO.
        # For standard PPOWrapper/DQNWrapper, this method doesn't exist,
        # so we use hasattr() to make it a no-op safely.
        if hasattr(agent, "reset"):
            agent.reset()
```

---

### Edit 5: Add RPPO to `run_all_agents()` alongside DQN

Find the `agents` dictionary in `run_all_agents()` and add RPPO:

```python
    # --- Resolve RPPO model path ---
    rppo_best  = os.path.join(ROOT, "models", "rppo_best", "best_model")
    rppo_final = os.path.join(ROOT, "models", "rppo_batching_final")
    if os.path.exists(rppo_best + ".zip") or os.path.exists(rppo_best):
        rppo_path = rppo_best
    elif os.path.exists(rppo_final + ".zip") or os.path.exists(rppo_final):
        rppo_path = rppo_final
    else:
        rppo_path = None
        print("  [WARN] RPPO model not found — run agent/train_rppo.py first.")

    agents = {
        "PPO":        PPOWrapper(PPO.load(model_path, env=env)),
        "RPPO":       RecurrentPPOWrapper(RecurrentPPO.load(rppo_path, env=env)) if rppo_path else RandomAgent(seed=1),
        "DQN":        DQNWrapper(DQN.load(dqn_path, env=env)) if dqn_path else RandomAgent(seed=0),
        "Greedy":     GreedyAgent(),
        "Cloudflare": CloudflareBaseline(
                          max_latency_ms=CONFIG["max_latency_ms"], seed=42),
        "Random":     RandomAgent(seed=42),
    }
```

---

## 8. Step 4 — Run RPPO Training

After making all edits above, train the RPPO agent:

```bash
python3 agent/train_rppo.py
```

**Estimated time:** Slightly slower than PPO due to LSTM forward-pass overhead, but uses 4 parallel envs. Expect **~10–20 min** for 500k steps on a MacBook CPU.

**Watch in TensorBoard (you already have it running!):**
```bash
# Your existing TensorBoard command already covers the new run:
tensorboard --logdir tensorboard_logs/
```

You will see a **third curve** `RPPO_batching` appear alongside `PPO_batching` and `DQN_batching`.

**Key TensorBoard metrics to watch for RPPO vs PPO:**
- Does RPPO converge faster in the early episodes? (It should, since traffic trends help)
- Is RPPO's `rollout/ep_rew_mean` higher at 500k steps?
- Is RPPO's reward **variance** (std of the curve) lower? Temporal memory should reduce the randomness of decisions.

---

## 9. Step 5 — Run Full Comparison

After training is done, run the updated evaluation:

```bash
python3 agent/evaluate.py
```

The output will now include RPPO (in magenta) alongside PPO, DQN, Greedy, Cloudflare, and Random in the `comparison_plots.png` comparison chart.

---

## 10. Understanding the RPPO Architecture Printed at Training Start

When you run `train_rppo.py`, it will print the full policy architecture. Here is what it looks like and what each part does:

```
RecurrentActorCriticPolicy(
  (features_extractor): FlattenExtractor(...)   # Raw obs → flat vector
  
  (lstm_actor): LSTM(6, 64, batch_first=True)   # Actor LSTM: 6 inputs, 64 hidden
  (lstm_critic): LSTM(6, 64, batch_first=True)  # Critic LSTM (separate weights)
  
  (mlp_extractor):
    (policy_net): Sequential(Linear(64→64), Tanh)  # Post-LSTM actor MLP
    (value_net):  Sequential(Linear(64→64), Tanh)  # Post-LSTM critic MLP
  
  (action_net): Linear(64, 2)    # Output: logits for [Wait, Serve]
  (value_net):  Linear(64, 1)    # Output: state value estimate V(s)
)
```

**The critical part:** `lstm_actor` processes the observation sequence `(o_1, o_2, ..., o_t)` and outputs `h_t` — a 64-dimensional summary of everything the agent has seen so far in the episode. The actor then maps `h_t → P(serve)`.

---

## 11. What to Expect in Results

### Expected Ranking (after 500k timesteps each)
```
RPPO ≥ PPO  >  DQN  >  Greedy  >  Random  >>  Cloudflare
```

### Why RPPO Should Edge Out PPO (or not — and why that's also interesting)

**Case A: RPPO wins**
- The LSTM successfully learnt to detect rising traffic trends
- You will see RPPO pre-emptively serving batches right before the rate peaks
- The decision heatmap will look "smoother" — less noisy threshold behaviour

**Case B: PPO ties or beats RPPO**
- This is also a valid and interesting result for your report
- It suggests the 6-dim observation already captures enough state that memory adds no value
- The `request_rate` EMA feature may already encode enough trend information that the LSTM has nothing extra to learn
- This is a genuine finding: "The engineered features sufficiently capture temporal dynamics, removing the need for recurrent memory in this problem"

Either outcome is academically publishable — it tests a real hypothesis about your environment.

### Metrics Table (fill in after running)

| Agent | Mean Reward | Std (variance) | p50 Latency | Avg Batch Size | Converges at |
|---|---|---|---|---|---|
| RPPO | ? | ? | ? ms | ? | ~? k steps |
| PPO | +607 | 495 | 6.0 ms | 1.12 | ~200k steps |
| DQN | ? | ? | ? ms | ? | ~150k steps |
| Greedy | +634 | 530 | 5.0 ms | 1.10 | N/A |

---

## 12. Full Terminal Command Reference

```bash
# 0. Install sb3-contrib (one time)
pip install sb3-contrib

# 1. Train all three RL agents
python3 agent/train.py          # Standard PPO  (~7 min)
python3 agent/train_dqn.py      # DQN           (~12 min)
python3 agent/train_rppo.py     # Recurrent PPO (~15 min)

# 2. Evaluate all agents + baselines
python3 agent/evaluate.py       # → results/comparison_plots.png

# 3. Watch all 3 training curves live
tensorboard --logdir tensorboard_logs/
# Open: http://localhost:6006
# You will see: PPO_batching / DQN_batching / RPPO_batching

# 4. (Optional) Ablation study on all models
python3 results/ablation_study.py
```

---

## 13. Suggested Report Section: RPPO vs PPO

If you are writing a course report, here is the exact hypothesis and analysis structure to use for the RPPO section:

**Hypothesis:**
> *"We hypothesise that the non-stationary Poisson arrival process creates temporal dependencies that a memoryless MLP policy cannot exploit. A recurrent LSTM-based policy should learn to detect traffic rate trends and make proactive serving decisions, improving the mean episode reward over standard PPO."*

**Experiment:**
> *"We trained Recurrent PPO (MlpLstmPolicy, hidden size 64) for 500k steps under identical hyperparameters to PPO. The only architectural difference was the LSTM layer replacing the first hidden MLP layer."*

**Result (write this after running):**
> *"RPPO achieved a mean episode reward of [X] vs PPO's [Y], a [positive/no] statistically significant difference (p=?, t-test over 30 episodes). [The LSTM's temporal memory allowed the agent to detect rising traffic and pre-emptively serve batches, reducing SLA violations by X% / PPO's engineered features were already sufficient to capture temporal dynamics, rendering LSTM memory redundant in this environment]."*

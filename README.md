# Dynamic Request Batching — Reinforcement Learning Project

A PPO (Proximal Policy Optimization) agent that learns *when* to dispatch queued inference requests to a GPU backend.  Every 10 ms the agent observes the current queue state and decides to **Wait** (keep accumulating) or **Serve** (dispatch the batch now), optimising the trade-off between GPU efficiency (large batches) and SLA compliance (low latency).

---

## Installation

```bash
pip install "stable-baselines3[extra]" gymnasium scipy matplotlib numpy
```

Python ≥ 3.12 required. No GPU needed for training or inference.

---

## Quick Start

```bash
# 1. Validate environment + baseline sanity check
python3 test_baselines.py

# 2. Train PPO agent  (~20-40 min on CPU, 1M steps)
python3 agent/train.py
tensorboard --logdir tensorboard_logs/   # monitor training curves

# 3. Evaluate and generate result plots
python3 agent/evaluate.py
# Outputs: results/comparison.png, results/decision_heatmap.png

# 4. (Optional) Smoke-test the deployment middleware
python3 deploy/middleware.py
```

---

## Problem Statement

An inference-serving node (GPU running a transformer model) receives a continuous stream of API requests at 2 000 req/s (peak: 5 000 req/s).  A batching layer must decide when to dispatch the accumulated queue:

| Decision | Consequence |
|---|---|
| Serve too early (tiny batch) | High dispatch overhead, poor GPU utilisation |
| Wait too long (large batch) | High latency, SLA violations |
| **Optimal** | Adapt threshold dynamically based on arrival rate, queue age, time-of-day |

---

## MDP Formulation

### State Space (7-dimensional, continuous)

| Index | Feature | Range | Purpose |
|---|---|---|---|
| 0 | `pending_requests` | [0, 512] | Current queue size |
| 1 | `oldest_wait_ms` | [0, 1000] | Age of head-of-queue request |
| 2 | `request_rate` | [0, 25 000] | EMA of arrival rate (req/s) |
| 3 | `since_serve_ms` | [0, 1000] | Time since last dispatch |
| 4 | `batch_fill_ratio` | [0, 1] | pending / max_batch_size |
| 5 | `time_of_day` | [0, 24) | Fractional hour (traffic predictor) |
| **6** | **`urgency_ratio`** | **[0, 3]** | **oldest_wait / effective_budget** — accounts for GPU processing time; >1.0 means SLA will be breached |

The 7th dimension (urgency_ratio) is new: it encodes the real-world constraint that **total latency = queue wait + GPU processing time**.  A request waiting 480 ms with a batch of 200 (processing ≈ 32 ms) would breach the 500 ms SLA.  This signal allows the agent to act *before* violations occur.

### Action Space

`Discrete(2)`:  `0 = Wait`,  `1 = Serve`

### Reward

```
Serve:  alpha × batch_size
      − beta  × oldest_wait_ms
      − dispatch_cost
      − sla_penalty_per_ms × Σ max(0, wait_i + processing_ms − SLA) per request

Wait:   −beta × oldest_wait_ms
      − idle_penalty   (only when queue is empty)
```

Key parameter: `dispatch_cost = 5.0`.  Serving a batch of fewer than 5 requests is unprofitable — the agent must learn to accumulate.

---

## GPU Processing Model

```
processing_time(n) = 8.0 + 0.12 × n   (ms)

  batch=1:    8.1 ms   (kernel launch dominates)
  batch=50:  14.0 ms
  batch=100: 20.0 ms
  batch=256: 38.7 ms   (memory-bandwidth bound)
```

The SLA is on **total client-perceived latency** (`queue_wait + processing_time`), not just queue wait.  This is the contractual metric production systems use.

---

## Algorithm: Proximal Policy Optimization (PPO)

**Why PPO over DQN / Recurrent PPO:**

| Criterion | PPO (chosen) | DQN | Recurrent PPO |
|---|---|---|---|
| State space | continuous 7-dim ✓ | continuous ✓ | continuous ✓ |
| Training stability | high (clipped objective) ✓ | moderate (requires careful LR tuning) | high but slower |
| Temporal correlation | handled via GAE ✓ | replay buffer breaks it | explicit via LSTM |
| Deployment | stateless predict() ✓ | stateless ✓ | stateful (LSTM) — harder to deploy |
| Convergence speed | 1M steps sufficient ✓ | 1M steps sufficient | 2M+ steps needed |

The 6 state features already capture recent history (EMA rate, since_serve_ms), making an explicit LSTM unnecessary.  PPO's entropy bonus (`ent_coef=0.005`) prevents the degenerate always-serve policy.

**Architecture:** MLP with `[128, 128]` hidden layers for both policy and value networks.  VecNormalize standardises observations online (critical: features span different scales — pending ∈ [0, 512] vs fill_ratio ∈ [0, 1]).

---

## Baselines

### Cloudflare (competitive target)

Production dual-threshold heuristic used by Cloudflare AI Gateway, NVIDIA Triton, and vLLM:

```
SERVE  if  oldest_wait ≥ 0.80 × (SLA − processing_time(batch))   [urgency]
SERVE  if  batch_size  ≥ arrival_rate × 0.050                     [efficiency]
WAIT   otherwise
```

At 2 000 req/s: serves when queue reaches ~100 requests OR oldest request has used 80% of its latency budget — whichever comes first.  This is adaptive: at 5 000 req/s the batch target automatically becomes ~250.

> **Why the original `exp(−λ·t)` formula was wrong**: At 2 000 req/s with 500 ms SLA, `exp(−2000 × 0.5) ≈ 0`.  The original implementation never dispatched, violated every SLA, and scored −6 000+ per episode.  The formula is designed for low-throughput web batching (λ ≈ 1–10 req/s) and does not transfer to inference serving.

### GreedyBatch (floor baseline)

Serves immediately whenever the queue reaches `min_batch_size = 8`.  Maximises dispatch frequency but wastes GPU capacity on tiny batches.

---

## Results

After training, run `python3 agent/evaluate.py` to regenerate:

| Metric | PPO | Cloudflare | GreedyBatch |
|---|---|---|---|
| Mean episode reward | see results/ | see results/ | see results/ |
| P95 total latency | see results/ | see results/ | see results/ |
| SLA violation rate | see results/ | see results/ | see results/ |
| Avg batch size | see results/ | see results/ | see results/ |

**What to look for in the results:**
- PPO should have **higher mean reward** and **lower SLA violation rate** than Cloudflare
- The improvement should be **meaningful but not trivial** (~10–25%): PPO adapts thresholds dynamically while Cloudflare uses fixed rules
- The decision heatmap (`decision_heatmap.png`) shows the **non-linear learned boundary** — curved rather than the two straight lines of the Cloudflare heuristic

---

## Deployment

```python
from deploy.middleware import BatchingMiddleware

# Startup (once)
mw = BatchingMiddleware(
    model_path  = "models/ppo_final",
    vecnorm_path= "models/ppo_vecnorm.pkl",
)

# Per 10ms tick
if mw.should_dispatch():
    batch = mw.flush()
    inference_backend.process(batch)
```

The middleware loads both the policy weights and the VecNormalize running statistics.  **Both files must be regenerated together** if you retrain — stale normalization statistics with a new model will produce incorrect predictions.

### Dynamic parameters

The agent adapts the following at runtime (no rule tuning required):
- Effective dispatch threshold based on `batch_fill_ratio` × `urgency_ratio`
- Aggressiveness at traffic spikes (via `ema_rate`)
- Time-of-day load prediction (via `time_of_day`)
- SLA urgency adjusted for GPU compute overhead (via `urgency_ratio`)

To deploy to a different SLA or GPU tier: update `config.py` and retrain.

---

## File Structure

```
config.py                  Central hyperparameters (env, GPU model, PPO, training)
env/
  batching_env.py          Gymnasium environment (7-dim state, GPU latency model)
  traffic_generator.py     Poisson arrivals with time-of-day variation
baselines/
  cloudflare_formula.py    Cloudflare dual-threshold baseline + GreedyBatch
agent/
  train.py                 PPO training (VecNormalize, 1M steps, checkpoints)
  evaluate.py              Comparison: PPO vs Cloudflare vs GreedyBatch
deploy/
  middleware.py            Production deployment class
models/
  ppo_final.zip            Trained policy weights
  ppo_vecnorm.pkl          VecNormalize statistics (required for deployment)
  best/best_model.zip      Best checkpoint by eval reward
results/
  comparison.png           6-panel comparison figure
  decision_heatmap.png     PPO P(Serve) policy heatmap
tensorboard_logs/          Training curves (PPO_batching)
```

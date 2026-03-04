"""
results/ablation_study.py

Feature-importance ablation study for the Dynamic Request Batching PPO agent.

For each of the 6 state features, we train a "crippled" variant of the agent
where that feature is permanently zeroed out.  The resulting mean reward is
compared against a full model baseline, and the drop (Δreward) is used as a
proxy for how much the agent *relies on* that feature.

Why ablation?
-------------
In RL, the policy is a black-box neural network.  We cannot simply inspect
weights to understand which observations matter most (unlike linear models).
Ablation is the standard practical alternative: you *remove* an input, retrain,
and measure the performance cost — features that hurt most when removed are
the most informative.

Usage:
    python results/ablation_study.py

Outputs:
    results/ablation.png              — bar chart of reward drop per feature
    Terminal                          — formatted table
"""

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from env.batching_env import BatchingEnv
from config import CONFIG

# ── Hyperparameters for the ablation runs ────────────────────────────────────
# We trade off accuracy for speed: 200k steps × 2 seeds = manageable runtime.
ABLATION_TIMESTEPS = 200_000
N_SEEDS            = 2
N_EVAL_EPISODES    = 10
N_ENVS             = 4

RESULTS_DIR = os.path.join(ROOT, "results")
PLOT_PATH   = os.path.join(RESULTS_DIR, "ablation.png")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Observation feature names (must match BatchingEnv._get_obs() order) ──────
FEATURE_NAMES = [
    "pending_requests",
    "oldest_wait_ms",
    "request_rate",
    "since_serve_ms",
    "fill_ratio",
    "time_of_day",
]

# ── Dark-theme colours ────────────────────────────────────────────────────────
BG_FIG    = "#0a0e1a"
BG_AX     = "#141c30"
COL_GRID  = "#2a3450"
COL_SPINE = "#2a3450"
COL_TEXT  = "#e8eaf6"
COL_BASE  = "#00e5ff"    # cyan — full model bar
COL_DROP  = "#ef5350"    # red  — ablated bars


# ─────────────────────────────────────────────────────────────────────────────
# Crippled environment wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ZeroFeatureWrapper(gym.ObservationWrapper):
    """Gymnasium wrapper that permanently zeros out one observation dimension.

    By zeroing a feature at the *environment* level (not the model level),
    we ensure the agent can never recover the information through correlated
    features — the ablation is clean.

    Parameters
    ----------
    env : gym.Env
        The wrapped environment.
    feature_idx : int
        Index of the observation dimension to zero out.
    """

    def __init__(self, env: gym.Env, feature_idx: int):
        super().__init__(env)
        self.feature_idx = feature_idx

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.copy()
        obs[self.feature_idx] = 0.0
        return obs


# ─────────────────────────────────────────────────────────────────────────────
# Training and evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_env_factory(feature_idx: int | None, seed: int):
    """Return a factory that builds a (possibly crippled) Monitor-wrapped env."""
    def _init():
        env = BatchingEnv(seed=seed)
        if feature_idx is not None:
            env = ZeroFeatureWrapper(env, feature_idx)
        env = Monitor(env, filename=None)
        return env
    return _init


def train_one(feature_idx: int | None, seed: int) -> float:
    """Train a PPO agent (possibly ablated) and return its mean eval reward.

    Parameters
    ----------
    feature_idx : int | None
        Feature to zero out.  None = full model (baseline).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    float
        Mean episode reward over N_EVAL_EPISODES deterministic rollouts.
    """
    from stable_baselines3.common.env_util import make_vec_env

    # Vectorised training env
    env_fns = [_make_env_factory(feature_idx, seed + i) for i in range(N_ENVS)]
    from stable_baselines3.common.vec_env import DummyVecEnv
    train_env = DummyVecEnv(env_fns)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,       # shorter for speed
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        verbose=0,
        seed=seed,
    )
    model.learn(total_timesteps=ABLATION_TIMESTEPS, progress_bar=False)
    train_env.close()

    # Deterministic evaluation
    rewards = []
    eval_env = BatchingEnv(seed=seed + 9999)
    if feature_idx is not None:
        eval_env = ZeroFeatureWrapper(eval_env, feature_idx)

    for ep in range(N_EVAL_EPISODES):
        obs, _ = eval_env.reset(seed=seed + 9999 + ep)
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(int(action))
            total += reward
        rewards.append(total)

    eval_env.close()
    return float(np.mean(rewards))


# ─────────────────────────────────────────────────────────────────────────────
# Main ablation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation() -> dict:
    """Train full model + one ablated model per feature; return results dict."""
    results = {}

    # ── Full model (no feature zeroed) ───────────────────────────────────────
    print("\n  Training FULL model (baseline)...")
    seed_rewards = []
    for s in range(N_SEEDS):
        t0 = time.time()
        r  = train_one(None, seed=s * 100)
        elapsed = time.time() - t0
        print(f"    seed {s}: mean_reward={r:+.1f}  ({elapsed:.0f}s)")
        seed_rewards.append(r)
    results["_full_"] = float(np.mean(seed_rewards))
    print(f"  → Full model mean reward: {results['_full_']:+.1f}\n")

    # ── One ablation per feature ──────────────────────────────────────────────
    for idx, name in enumerate(FEATURE_NAMES):
        print(f"  Ablating feature [{idx}] '{name}'...")
        seed_rewards = []
        for s in range(N_SEEDS):
            t0 = time.time()
            r  = train_one(idx, seed=s * 100)
            elapsed = time.time() - t0
            print(f"    seed {s}: mean_reward={r:+.1f}  ({elapsed:.0f}s)")
            seed_rewards.append(r)
        results[name] = float(np.mean(seed_rewards))
        drop = results["_full_"] - results[name]
        print(f"  → Mean reward: {results[name]:+.1f}   Δ = {drop:+.1f}\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation(results: dict) -> str:
    """Generate bar chart of reward drop and save to results/ablation.png."""
    full_r  = results["_full_"]
    names   = FEATURE_NAMES
    drops   = [full_r - results[n] for n in names]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color":  COL_TEXT,
    })

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_FIG)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.14)

    x     = np.arange(len(names))
    bars  = ax.bar(x, drops, color=COL_DROP, width=0.55, alpha=0.88, zorder=3)

    # Highlight the most important feature
    max_idx = int(np.argmax(drops))
    bars[max_idx].set_color(COL_BASE)
    bars[max_idx].set_alpha(1.0)

    # Value labels
    for bar, drop in zip(bars, drops):
        ax.annotate(
            f"Δ{drop:+.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, max(drop, 0)),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom",
            color=COL_TEXT, fontsize=8.5, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=22, ha="right", color=COL_TEXT, fontsize=9)
    ax.set_ylabel("Reward Drop  (Full − Ablated)", labelpad=8)
    ax.yaxis.label.set_color(COL_TEXT)
    ax.xaxis.label.set_color(COL_TEXT)
    ax.set_title(
        f"Feature Importance via Ablation\n"
        f"Full model reward: {full_r:+.0f}   |   "
        f"higher bar = more important feature",
        color=COL_TEXT, fontsize=11, fontweight="bold", pad=10,
    )

    ax.set_facecolor(BG_AX)
    for sp in ax.spines.values():
        sp.set_edgecolor(COL_SPINE)
    ax.tick_params(colors=COL_TEXT)
    ax.grid(True, axis="y", color=COL_GRID, linewidth=0.5, alpha=0.8)
    ax.axhline(0, color=COL_TEXT, linewidth=0.7, linestyle="--", alpha=0.4)

    # Legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=COL_BASE, label=f"Most important: {names[max_idx]}"),
        Patch(facecolor=COL_DROP, label="Other features"),
    ]
    ax.legend(handles=legend_elems, facecolor=BG_AX, edgecolor=COL_SPINE,
              labelcolor=COL_TEXT, fontsize=8)

    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig)
    return PLOT_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results: dict):
    full_r = results["_full_"]
    W = 60
    print("\n" + "=" * W)
    print(f"{'Feature':<22}  {'Ablated Reward':>14}  {'Δ Reward':>10}")
    print("-" * W)
    for name in FEATURE_NAMES:
        r    = results[name]
        drop = full_r - r
        bar  = "█" * max(0, int(drop / max(max(results[n] for n in FEATURE_NAMES), 1) * 20))
        print(f"  {name:<20}  {r:>+14.1f}  {drop:>+10.1f}  {bar}")
    print("=" * W)
    print(f"  Full model (baseline)    {full_r:>+14.1f}")
    print("=" * W)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Dynamic Request Batching — Ablation Study")
    print("=" * 60)
    print(f"  Features      : {len(FEATURE_NAMES)}")
    print(f"  Seeds/feature : {N_SEEDS}")
    print(f"  Timesteps     : {ABLATION_TIMESTEPS:,}")
    print(f"  Total runs    : {(len(FEATURE_NAMES) + 1) * N_SEEDS}")
    t_est = (len(FEATURE_NAMES) + 1) * N_SEEDS * (ABLATION_TIMESTEPS / 1200) / 60
    print(f"  Est. runtime  : ~{t_est:.0f} min on CPU")
    print("=" * 60)

    t0 = time.time()
    results = run_ablation()
    elapsed = time.time() - t0

    print_table(results)

    path = plot_ablation(results)
    print(f"\n✓  Ablation chart saved → {path}")
    print(f"✓  Total elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()

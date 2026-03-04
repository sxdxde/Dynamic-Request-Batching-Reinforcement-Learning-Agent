"""
agent/evaluate.py

Evaluation script: compares the trained PPO agent against all 3 baselines
over 30 episodes each, then generates comparison figures.

Outputs
-------
results/comparison_plots.png  — 3-panel agent comparison figure
results/decision_heatmap.png  — P(serve) heatmap over (batch_size × oldest_wait_ms)
Terminal                      — formatted summary table

Usage
-----
    python agent/evaluate.py
    python agent/evaluate.py --model models/best/best_model
"""

import os
import sys
import argparse
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless; works in any environment
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from stable_baselines3 import PPO

from env.batching_env import BatchingEnv
from baselines.cloudflare_formula import CloudflareBaseline, evaluate_baseline
from baselines.random_agent import RandomAgent
from baselines.greedy_agent import GreedyAgent
from config import CONFIG

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR      = os.path.join(ROOT, "results")
PLOT_PATH        = os.path.join(RESULTS_DIR, "comparison_plots.png")
HEATMAP_PATH     = os.path.join(RESULTS_DIR, "decision_heatmap.png")
DEFAULT_MODEL    = os.path.join(ROOT, "models", "best", "best_model")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_EPISODES  = 30
SEED_OFFSET = 100

# ── Dark-theme palette ─────────────────────────────────────────────────────────
BG_FIGURE  = "#0a0e1a"
BG_AXES    = "#141c30"
TEXT_COLOR = "#e8eaf6"
GRID_COLOR = "#2a3450"
SPINE_COLOR= "#2a3450"

AGENT_COLORS = {
    "PPO":        "#00e5ff",   # cyan
    "Greedy":     "#7c4dff",   # purple
    "Cloudflare": "#ff6f00",   # amber
    "Random":     "#f44336",   # red
}
AGENT_ORDER = ["PPO", "Greedy", "Cloudflare", "Random"]


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────

class PPOWrapper:
    """Thin wrapper so PPO has the same predict(obs)->int interface as baselines."""
    def __init__(self, model):
        self.model = model

    def predict(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def collect_episode_data(agent, env, n_episodes: int, seed_offset: int) -> dict:
    """Run agent for n_episodes and collect detailed per-step data.

    Returns
    -------
    dict with keys:
        rewards      : list[float]      — total reward per episode
        batch_sizes  : list[int]        — batch size at every Serve action
        latencies    : list[float]      — per-request latency at every Serve
    """
    rewards     = []
    batch_sizes = []
    latencies   = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        total_reward = 0.0
        terminated = truncated = False
        prev_served = 0

        # track queue snapshot before each serve
        while not (terminated or truncated):
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Infer batch_size from total_served delta
            if action == 1:
                served_now = info["total_served"] - prev_served
                if served_now > 0:
                    batch_sizes.append(served_now)
                    # mean_latency_ms from info is cumulative; use as proxy
                    if info["mean_latency_ms"] > 0:
                        latencies.append(info["mean_latency_ms"])
                prev_served = info["total_served"]

        rewards.append(total_reward)

    return {
        "rewards": rewards,
        "batch_sizes": batch_sizes,
        "latencies": latencies,
    }


def run_all_agents(model_path: str) -> dict[str, dict]:
    """Evaluate PPO + all baselines; return data dict keyed by agent name."""
    env = BatchingEnv()

    # Build agents
    agents = {
        "PPO":        PPOWrapper(PPO.load(model_path, env=env)),
        "Greedy":     GreedyAgent(),
        "Cloudflare": CloudflareBaseline(
                          max_latency_ms=CONFIG["max_latency_ms"], seed=42),
        "Random":     RandomAgent(seed=42),
    }

    all_data = {}
    for name in AGENT_ORDER:
        print(f"  Evaluating {name:12s} ({N_EPISODES} episodes) ...", end=" ", flush=True)
        data = collect_episode_data(agents[name], env, N_EPISODES, SEED_OFFSET)
        all_data[name] = data
        mean_r = np.mean(data["rewards"])
        print(f"mean reward = {mean_r:+.2f}")

    env.close()
    return all_data


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_dark_style(ax):
    """Apply dark styling to a single Axes."""
    ax.set_facecolor(BG_AXES)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.8)


def plot_reward_bars(ax, data: dict[str, dict]):
    """Panel 1: Bar chart of mean ± std reward."""
    means = [np.mean(data[n]["rewards"])   for n in AGENT_ORDER]
    stds  = [np.std(data[n]["rewards"])    for n in AGENT_ORDER]
    colors= [AGENT_COLORS[n]               for n in AGENT_ORDER]
    x     = np.arange(len(AGENT_ORDER))

    bars = ax.bar(x, means, yerr=stds, color=colors, width=0.55,
                  capsize=6, error_kw=dict(color=TEXT_COLOR, linewidth=1.5),
                  alpha=0.88, zorder=3)

    # Value labels
    for bar, mean in zip(bars, means):
        va  = "bottom" if mean >= 0 else "top"
        off = 5 if mean >= 0 else -5
        ax.annotate(
            f"{mean:+.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, mean),
            xytext=(0, off),
            textcoords="offset points",
            ha="center", va=va,
            color=TEXT_COLOR, fontsize=8.5, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(AGENT_ORDER, color=TEXT_COLOR)
    ax.axhline(0, color=TEXT_COLOR, linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_title("Mean Episode Reward ± Std", fontweight="bold", fontsize=11)
    ax.set_ylabel("Episode Reward", labelpad=8)
    _apply_dark_style(ax)


def plot_batch_histogram(ax, data: dict[str, dict]):
    """Panel 2: Overlapping histogram of batch size distributions."""
    has_data = False
    for name in AGENT_ORDER:
        bs = data[name]["batch_sizes"]
        if not bs:
            continue
        has_data = True
        ax.hist(
            bs,
            bins=30,
            range=(0, CONFIG["max_batch_size"]),
            alpha=0.45,
            color=AGENT_COLORS[name],
            label=name,
            density=True,
            edgecolor="none",
        )
        # Overlay a smooth KDE-style line
        from scipy.stats import gaussian_kde  # optional import; falls back
        try:
            if len(set(bs)) > 1:
                kde = gaussian_kde(bs, bw_method=0.25)
                xs  = np.linspace(0, CONFIG["max_batch_size"], 300)
                ax.plot(xs, kde(xs), color=AGENT_COLORS[name], linewidth=1.8)
        except Exception:
            pass

    if not has_data:
        ax.text(0.5, 0.5, "No serve actions recorded",
                transform=ax.transAxes, ha="center", va="center",
                color=TEXT_COLOR, fontsize=10)

    legend = ax.legend(
        facecolor=BG_AXES, edgecolor=SPINE_COLOR,
        labelcolor=TEXT_COLOR, fontsize=8
    )
    ax.set_title("Batch Size Distribution", fontweight="bold", fontsize=11)
    ax.set_xlabel("Batch Size (requests)")
    ax.set_ylabel("Density")
    _apply_dark_style(ax)


def plot_latency_cdf(ax, data: dict[str, dict]):
    """Panel 3: Latency CDF curves (lower-left = better)."""
    sla = CONFIG["max_latency_ms"]
    plotted = False

    for name in AGENT_ORDER:
        lat = data[name]["latencies"]
        if not lat:
            continue
        plotted = True
        sorted_lat = np.sort(lat)
        cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
        ax.plot(sorted_lat, cdf,
                color=AGENT_COLORS[name],
                linewidth=2.0,
                label=name,
                alpha=0.9)

    # SLA deadline vertical
    ax.axvline(sla, color="#ffffff", linewidth=1.0, linestyle=":",
               alpha=0.6, label=f"SLA ({sla} ms)")

    if not plotted:
        ax.text(0.5, 0.5, "No latency data collected",
                transform=ax.transAxes, ha="center", va="center",
                color=TEXT_COLOR, fontsize=10)

    legend = ax.legend(
        facecolor=BG_AXES, edgecolor=SPINE_COLOR,
        labelcolor=TEXT_COLOR, fontsize=8
    )
    ax.set_title("Serve Latency CDF  (lower-left = better)",
                 fontweight="bold", fontsize=11)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_ylim(0, 1.05)
    _apply_dark_style(ax)


def generate_figure(data: dict[str, dict]) -> str:
    """Build the 3-panel figure, save it, and return the path."""
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "text.color":        TEXT_COLOR,
        "axes.labelcolor":   TEXT_COLOR,
        "xtick.color":       TEXT_COLOR,
        "ytick.color":       TEXT_COLOR,
    })

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6),
        facecolor=BG_FIGURE,
    )
    fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.14, wspace=0.30)

    fig.suptitle(
        "Dynamic Request Batching — Agent Comparison",
        color=TEXT_COLOR,
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plot_reward_bars(axes[0], data)
    plot_batch_histogram(axes[1], data)
    plot_latency_cdf(axes[2], data)

    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight", facecolor=BG_FIGURE)
    plt.close(fig)
    return PLOT_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Decision heatmap
# ─────────────────────────────────────────────────────────────────────────────

def generate_decision_heatmap(model_path: str, grid_size: int = 50) -> str:
    """Generate a P(serve) heatmap over a grid of (batch_size × oldest_wait_ms).

    The PPO policy network outputs a probability distribution over actions.
    By sweeping two of the most interpretable state dimensions—how many requests
    are waiting and how old the oldest one is—while holding the remaining dims
    at their empirical mean values, we get a 2D slice through the policy's
    decision surface.  This directly shows *where* in state-space the agent
    prefers to serve vs. wait, which is the key learned batching heuristic.

    Parameters
    ----------
    model_path : str
        Path to a saved SB3 PPO model (with or without .zip).
    grid_size : int
        Resolution of the sweep grid along each axis.

    Returns
    -------
    str
        Path to the saved heatmap PNG.
    """
    import torch

    model = PPO.load(model_path)

    max_batch  = CONFIG["max_batch_size"]   # 100
    max_lat    = CONFIG["max_latency_ms"]   # 500

    # ── Representative "mean" values for the 4 fixed dimensions ──────────────
    # These come from a short rollout average so the heatmap reflects realistic
    # context rather than arbitrary zeros.
    arrival_rate_mean  = CONFIG["arrival_rate"] * CONFIG["peak_multiplier"] * 0.5
    since_serve_mean   = 150.0   # ms — moderate time since last serve
    fill_ratio_mean    = 0.3     # 30 % full — mid-episode typical
    time_of_day_mean   = 13.0    # early afternoon

    # ── Build observation grid ────────────────────────────────────────────────
    batch_sizes  = np.linspace(0, max_batch, grid_size)
    oldest_waits = np.linspace(0, max_lat,   grid_size)

    p_serve = np.zeros((grid_size, grid_size), dtype=np.float32)

    # PPO uses a categorical distribution over actions; we extract the raw
    # log-probability of action=1 (Serve) from the policy network.
    model.policy.set_training_mode(False)

    for i, bs in enumerate(batch_sizes):
        for j, ow in enumerate(oldest_waits):
            obs = np.array([
                bs,
                ow,
                arrival_rate_mean,
                since_serve_mean,
                bs / max(max_batch, 1),  # fill_ratio derived from bs
                time_of_day_mean,
            ], dtype=np.float32)

            obs_t = torch.as_tensor(obs).unsqueeze(0)  # shape (1, 6)
            with torch.no_grad():
                dist = model.policy.get_distribution(
                    model.policy.obs_to_tensor(obs)[0]
                )
                # log_prob for action=1 → exponentiate to get P(serve)
                log_p = dist.distribution.logits[0, 1]  # logit for class 1
                p     = torch.sigmoid(log_p).item()      # stable sigmoid
            p_serve[j, i] = p  # rows = oldest_wait (y), cols = batch_size (x)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":  "DejaVu Sans",
        "text.color":   TEXT_COLOR,
    })

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG_FIGURE)
    fig.subplots_adjust(left=0.10, right=0.92, top=0.90, bottom=0.12)

    # Use a diverging colormap anchored at p=0.5 (decision boundary)
    from matplotlib.colors import TwoSlopeNorm
    norm  = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    cmap  = plt.cm.RdYlGn   # red=Wait, green=Serve

    im = ax.imshow(
        p_serve,
        origin="lower",
        aspect="auto",
        extent=[0, max_batch, 0, max_lat],
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
    )

    # Contour at p=0.5 — the decision boundary
    ax.contour(
        np.linspace(0, max_batch, grid_size),
        np.linspace(0, max_lat,   grid_size),
        p_serve,
        levels=[0.5],
        colors=["white"],
        linewidths=1.5,
        linestyles="--",
    )

    # SLA deadline horizontal line
    ax.axhline(max_lat, color="#ff5252", linewidth=1.2, linestyle=":",
               alpha=0.8, label=f"SLA deadline ({max_lat} ms)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(Serve)", color=TEXT_COLOR, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    cbar.ax.tick_params(colors=TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)

    ax.set_facecolor(BG_AXES)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)

    ax.set_xlabel("Pending Batch Size (requests)", labelpad=8)
    ax.set_ylabel("Oldest Request Wait (ms)", labelpad=8)
    ax.set_title(
        "PPO Decision Heatmap — P(Serve)\n"
        "white dashed line = decision boundary (p = 0.5)",
        color=TEXT_COLOR, fontsize=11, fontweight="bold", pad=10,
    )
    ax.legend(facecolor=BG_AXES, edgecolor=SPINE_COLOR,
              labelcolor=TEXT_COLOR, fontsize=8, loc="upper left")

    fig.text(
        0.5, 0.01,
        f"Other state dims fixed at:  rate={arrival_rate_mean:.0f} req/s  "
        f"| since_serve={since_serve_mean:.0f} ms  "
        f"| time_of_day={time_of_day_mean:.0f}h",
        ha="center", color="#90a4ae", fontsize=7.5,
    )

    fig.savefig(HEATMAP_PATH, dpi=150, bbox_inches="tight", facecolor=BG_FIGURE)
    plt.close(fig)
    return HEATMAP_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(data: dict[str, dict]):
    W = 76
    print("\n" + "=" * W)
    print(f"{'Agent':<14}  {'Mean Reward':>12}  {'Std Reward':>10}  "
          f"{'p50 Latency':>12}  {'p95 Latency':>12}  {'Avg Batch':>10}")
    print("-" * W)
    for name in AGENT_ORDER:
        d = data[name]
        mean_r = np.mean(d["rewards"])
        std_r  = np.std(d["rewards"])
        lats   = d["latencies"]
        p50    = float(np.percentile(lats, 50)) if lats else float("nan")
        p95    = float(np.percentile(lats, 95)) if lats else float("nan")
        avg_bs = float(np.mean(d["batch_sizes"])) if d["batch_sizes"] else float("nan")
        print(
            f"{name:<14}  {mean_r:>+12.2f}  {std_r:>10.2f}  "
            f"{p50:>12.1f}  {p95:>12.1f}  {avg_bs:>10.2f}"
        )
    print("=" * W)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO vs baselines"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Path to SB3 model zip (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--skip-heatmap",
        action="store_true",
        help="Skip the decision heatmap generation (faster).",
    )
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        print(f"\n[ERROR] Model not found at: {model_path}")
        print("  Run `python agent/train.py` first to train the PPO agent.")
        sys.exit(1)

    print("=" * 60)
    print("  Dynamic Request Batching — Evaluation")
    print("=" * 60)
    print(f"  Model path  : {model_path}")
    print(f"  Episodes    : {N_EPISODES} per agent")
    print(f"  Output plot : {PLOT_PATH}")
    print("=" * 60 + "\n")

    data = run_all_agents(model_path)
    print_summary_table(data)

    path = generate_figure(data)
    print(f"\n✓  Comparison figure saved → {path}")

    if not args.skip_heatmap:
        print("\n  Generating decision heatmap (querying policy on 50×50 grid)...",
              end=" ", flush=True)
        hmap_path = generate_decision_heatmap(model_path)
        print(f"done.\n✓  Decision heatmap saved   → {hmap_path}")


if __name__ == "__main__":
    main()

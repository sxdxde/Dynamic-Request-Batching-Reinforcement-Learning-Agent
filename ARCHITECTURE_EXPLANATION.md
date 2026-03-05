# Project Architecture: Training, Evaluation, and Wrappers

This document explains how the different pieces of this Reinforcement Learning project fit together, specifically focusing on the separation between training and evaluation, and why we use "Wrappers".

## 1. The Training Scripts (`train.py`, `train_dqn.py`, `train_rppo.py`)

Think of the training scripts as the **"gym"** where the agents work out. 

* **Their single job:** To create a blank-slate neural network, drop it into the `BatchingEnv`, and let it practice for 500,000 steps. 
* As the agent practices, the Stable-Baselines3 algorithm (PPO, DQN, or RPPO) constantly tunes the neural network's weights to maximize the reward.
* When training is finished, the script takes the fully trained "brain" and saves it to your hard drive as a `.zip` file (e.g., `models/ppo_batching_final.zip`).
* **Once the 500k steps are done, the training script's job is completely over.** It does not generate any final comparison graphs or performance tables.

## 2. The Evaluation Script (`evaluate.py`)

Think of `evaluate.py` as the **"final exam"**. 

* **Its single job:** To test all agents fairly against each other.
* It loads the saved `.zip` brains from your hard drive. 
* It puts all of your agents (PPO, DQN, Recurrent PPO), plus the dummy baselines (Greedy, Cloudflare, Random), into the exact same environment under the exact same conditions.
* It forces them to play 30 games (episodes) using a set of fixed random seeds to ensure every agent faces the exact same traffic patterns.
* **No learning happens here.** The neural network weights are frozen. `evaluate.py` simply measures how well they play and draws the graphs (`results/comparison_plots.png`) to prove who is the best.

## 3. Why do we need the "Wrappers" (`PPOWrapper`, `DQNWrapper`, `RecurrentPPOWrapper`)?

In software engineering, this is an application of the **Adapter Pattern**. 

Inside `evaluate.py`, we want to evaluate every agent using a single, clean `while` loop:

```python
while not (terminated or truncated):
    action = agent.predict(obs)  # <--- We want this exact line to work for EVERY agent
    obs, reward, terminated, truncated, info = env.step(action)
```

The problem is that our different AI models and baselines give their answers in completely different formats:

* **RandomBaseline:** Its `predict(obs)` just spits out a simple integer `1` (Serve) or `0` (Wait).
* **Standard PPO/DQN (Stable-Baselines3):** Their `predict` function returns a tuple of numpy arrays: `(array([1]), None)`. If you try to pass `array([1])` directly into `env.step()`, the Gymnasium environment will crash or complain about incorrect types.
* **Recurrent PPO:** Its `predict` function requires you to track its hidden memory state and explicitly pass it back in: `action, new_memory = predict(obs, state=old_memory, episode_start=...)`.

If we didn't use wrappers, our core evaluation loop would look like a giant, messy `if/else` block:

```python
if agent_type == "PPO" or agent_type == "DQN":
    action_array, _ = agent.predict(obs, deterministic=True)
    action = int(action_array)
elif agent_type == "RPPO":
    ep_start = [episode_started]
    action_array, lstm_state = agent.predict(obs, state=lstm_state, episode_start=ep_start)
    action = int(action_array)
elif agent_type == "Random":
    action = agent.predict(obs)
```

**The Solution:** We create "Wrapper" classes. 

Wrappers act like translators. They take the weird, specific format of PPO, DQN, or RPPO, hide the complex logic inside, and provide a single, standard `agent.predict(obs)` function that just spits out a clean python `int` (`1` or `0`). 

This keeps the core evaluation code incredibly clean, readable, and easy to extend if we ever add a 10th algorithm in the future.

import os
import sys
import json
import argparse
import random
from collections import defaultdict

import yaml
import joblib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Allow importing from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim.taxi_env import TaxiDispatchEnv


def make_q_table(num_actions):
    return defaultdict(lambda: np.zeros(num_actions))


def choose_action(q_table, state, epsilon, num_actions):
    """
    Epsilon-greedy exploration.
    """
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    return int(np.argmax(q_table[state]))


def train(config):
    os.makedirs("policies", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    run_id = config["run_id"]

    env = TaxiDispatchEnv(
        grid_size=config["grid_size"],
        num_taxis=config["num_taxis"],
        max_steps=config["max_steps"],
        seed=42
    )

    num_actions = config["num_taxis"]

    episodes = config["episodes"]
    alpha = config["learning_rate"]
    gamma = config["gamma"]

    epsilon = config["epsilon"]
    epsilon_decay = config["epsilon_decay"]
    min_epsilon = config["min_epsilon"]

    q_table = make_q_table(num_actions)

    episode_rewards = []
    episode_waiting_times = []

    for episode in tqdm(range(episodes), desc=f"Training {run_id}"):
        state = env.reset()

        total_reward = 0
        total_waiting_time = 0
        done = False

        while not done:
            action = choose_action(q_table, state, epsilon, num_actions)

            next_state, reward, done, info = env.step(action)

            best_next_action_value = np.max(q_table[next_state])

            q_table[state][action] = q_table[state][action] + alpha * (
                reward + gamma * best_next_action_value - q_table[state][action]
            )

            state = next_state

            total_reward += reward
            total_waiting_time += info["waiting_time"]

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        episode_rewards.append(total_reward)
        episode_waiting_times.append(total_waiting_time / config["max_steps"])

    avg_reward = float(np.mean(episode_rewards[-50:]))
    avg_waiting_time = float(np.mean(episode_waiting_times[-50:]))

    # Save policy
    policy_data = {
        "q_table": {str(k): v.tolist() for k, v in q_table.items()},  # ← serialization-safe
        "num_actions": num_actions,
        "config": config
    }

    joblib.dump(policy_data, config["policy_path"])

    # Save JSON result
    results = {
        "run_id": run_id,
        "algorithm": "Q-learning",
        "episodes": episodes,
        "average_reward_last_50": avg_reward,
        "average_waiting_time_last_50": avg_waiting_time,
        "parameters": {
            "learning_rate": alpha,
            "gamma": gamma,
            "initial_epsilon": config["epsilon"],
            "epsilon_decay": epsilon_decay,
            "min_epsilon": min_epsilon
        },
        "state": "nearest_taxi_distance + passenger_pickup_x + passenger_pickup_y",
        "action": "choose taxi index to dispatch",
        "reward": "negative passenger waiting time",
        "policy_file": config["policy_path"],
        "reward_history": episode_rewards,
        "waiting_time_history": episode_waiting_times
    }

    with open(config["results_path"], "w") as f:
        json.dump(results, f, indent=4)

    # Save reward curve
    window = 20
    smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode="valid")

    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.3, color="steelblue", label="Raw reward")
    plt.plot(range(window - 1, len(episode_rewards)), smoothed,
            color="red", linewidth=2, label=f"{window}-ep moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Reward Curve - {run_id}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config["plot_path"])
    plt.close()

    print("\nTraining completed")
    print("Run ID:", run_id)
    print("Policy saved to:", config["policy_path"])
    print("Results saved to:", config["results_path"])
    print("Plot saved to:", config["plot_path"])
    print("Average reward last 50 episodes:", avg_reward)
    print("Average waiting time last 50 episodes:", avg_waiting_time)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    train(config)
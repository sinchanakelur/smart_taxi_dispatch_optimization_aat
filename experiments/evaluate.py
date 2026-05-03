import os
import sys
import json
import random
import joblib
import numpy as np

# allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim.taxi_env import TaxiDispatchEnv


def random_policy(num_actions):
    return random.randint(0, num_actions - 1)


def rl_policy(q_table, state):
    if state not in q_table:
        return 0
    return int(np.argmax(q_table[state]))


def run_episode(env, policy_fn, q_table=None):
    state = env.reset()

    total_wait = 0
    total_reward = 0
    done = False

    while not done:
        if q_table is None:
            action = policy_fn(env.num_taxis)
        else:
            action = policy_fn(q_table, state)

        next_state, reward, done, info = env.step(action)

        total_wait += info["waiting_time"]
        total_reward += reward

        state = next_state

    avg_wait = total_wait / env.max_steps
    return avg_wait, total_reward


def evaluate():
    env = TaxiDispatchEnv(grid_size=5, num_taxis=3, max_steps=50)

    # Load trained RL policy
    policy_data = joblib.load("policies/policy_v2_explored.pkl")
    q_table = policy_data["q_table"]

    episodes = 100

    random_waits = []
    rl_waits = []

    for _ in range(episodes):
        w, _ = run_episode(env, random_policy)
        random_waits.append(w)

    for _ in range(episodes):
        w, _ = run_episode(env, rl_policy, q_table)
        rl_waits.append(w)

    avg_random = float(np.mean(random_waits))
    avg_rl = float(np.mean(rl_waits))

    improvement = ((avg_random - avg_rl) / avg_random) * 100

    results = {
        "baseline": "Random Dispatch",
        "episodes": episodes,
        "random_avg_waiting_time": avg_random,
        "rl_avg_waiting_time": avg_rl,
        "improvement_percent": improvement
    }

    os.makedirs("results", exist_ok=True)

    with open("results/comparison.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n=== Evaluation Results ===")
    print("Random Policy Avg Wait:", avg_random)
    print("RL Policy Avg Wait:", avg_rl)
    print("Improvement (%):", improvement)


if __name__ == "__main__":
    evaluate()
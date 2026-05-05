import os
import sys
import json
import joblib
import numpy as np

# allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim.taxi_env import TaxiDispatchEnv


def nearest_taxi_policy(env, state):
    """
    Rule-based baseline: always dispatch the closest taxi to the passenger.
    state = (dist_taxi_0, dist_taxi_1, dist_taxi_2, pickup_x, pickup_y)
    """
    distances = state[:env.num_taxis]
    return int(np.argmin(distances))


def rl_policy(q_table, state):
    """
    RL policy: pick the action with the highest Q-value for this state.
    Falls back to taxi 0 if state was never seen during training.
    """
    key = str(state)
    if key not in q_table:
        return 0
    return int(np.argmax(q_table[key]))


def run_episode(env, policy_type, q_table=None):
    """
    Run a single episode.
    policy_type: "nearest" for rule-based baseline, "rl" for Q-learning policy.
    """
    state = env.reset()

    total_wait = 0
    total_reward = 0
    done = False

    while not done:
        if policy_type == "nearest":
            action = nearest_taxi_policy(env, state)
        else:
            action = rl_policy(q_table, state)

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

    baseline_waits = []
    rl_waits = []

    print("Running nearest-taxi baseline...")
    for _ in range(episodes):
        w, _ = run_episode(env, policy_type="nearest")
        baseline_waits.append(w)

    print("Running RL policy...")
    for _ in range(episodes):
        w, _ = run_episode(env, policy_type="rl", q_table=q_table)
        rl_waits.append(w)

    avg_baseline = float(np.mean(baseline_waits))
    avg_rl = float(np.mean(rl_waits))

    improvement = ((avg_baseline - avg_rl) / avg_baseline) * 100

    results = {
        "baseline": "Nearest-Taxi Heuristic",
        "episodes": episodes,
        "baseline_avg_waiting_time": avg_baseline,
        "rl_avg_waiting_time": avg_rl,
        "improvement_percent": improvement
    }

    os.makedirs("results", exist_ok=True)

    with open("results/comparison.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n=== Evaluation Results ===")
    print(f"Nearest-Taxi Baseline Avg Wait : {avg_baseline:.4f}")
    print(f"RL Policy Avg Wait             : {avg_rl:.4f}")
    print(f"Improvement (%)                : {improvement:.2f}%")


if __name__ == "__main__":
    evaluate()
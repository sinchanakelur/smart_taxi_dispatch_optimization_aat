import os
import json
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_reward_curve(result_path, output_path, title):
    data = load_json(result_path)

    rewards = data["reward_history"]

    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.savefig(output_path)
    plt.close()


def plot_waiting_time_comparison(comparison_path, output_path):
    data = load_json(comparison_path)

    labels = ["Random Dispatch", "RL Policy"]
    values = [
        data["random_avg_waiting_time"],
        data["rl_avg_waiting_time"]
    ]

    plt.figure()
    plt.bar(labels, values)
    plt.xlabel("Policy")
    plt.ylabel("Average Waiting Time")
    plt.title("Baseline vs RL Waiting Time Comparison")
    plt.savefig(output_path)
    plt.close()


def main():
    os.makedirs("plots", exist_ok=True)

    plot_reward_curve(
        "results/results_qlearning_v1.json",
        "plots/reward_curve_v1.png",
        "Reward Curve - Q-learning V1"
    )

    plot_reward_curve(
        "results/results_qlearning_v2_explored.json",
        "plots/reward_curve_v2.png",
        "Reward Curve - Q-learning V2 Explored"
    )

    plot_waiting_time_comparison(
        "results/comparison.json",
        "plots/wait_time_comparison.png"
    )

    print("Plots generated successfully:")
    print("plots/reward_curve_v1.png")
    print("plots/reward_curve_v2.png")
    print("plots/wait_time_comparison.png")


if __name__ == "__main__":
    main()
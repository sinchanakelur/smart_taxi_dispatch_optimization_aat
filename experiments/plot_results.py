import os
import json
import numpy as np
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_reward_curve(result_path, output_path, title):
    data = load_json(result_path)
    rewards = data["reward_history"]

    window = 20
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, color="steelblue", label="Raw reward")
    plt.plot(
        range(window - 1, len(rewards)),
        smoothed,
        color="red",
        linewidth=2,
        label=f"{window}-episode moving average",
    )
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_waiting_time_comparison(comparison_path, output_path):
    data = load_json(comparison_path)

    labels = ["Nearest-Taxi Baseline", "RL Policy"]
    values = [
        data["baseline_avg_waiting_time"],
        data["rl_avg_waiting_time"],
    ]

    improvement = data.get("improvement_percent", 0)

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values, color=["steelblue", "darkorange"], width=0.4)

    # Add value labels on top of each bar
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.xlabel("Policy")
    plt.ylabel("Average Waiting Time")
    plt.title(f"Baseline vs RL Waiting Time Comparison\n(Improvement: {improvement:.1f}%)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    os.makedirs("plots", exist_ok=True)

    plot_reward_curve(
        "results/results_qlearning_v1.json",
        "plots/reward_curve_v1.png",
        "Reward Curve - Q-learning V1",
    )

    plot_reward_curve(
        "results/results_qlearning_v2_explored.json",
        "plots/reward_curve_v2.png",
        "Reward Curve - Q-learning V2 Explored",
    )

    plot_waiting_time_comparison(
        "results/comparison.json",
        "plots/wait_time_comparison.png",
    )

    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()
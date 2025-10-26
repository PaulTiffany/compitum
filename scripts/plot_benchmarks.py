import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_results(case: str):
    """Loads benchmark data for a given case and generates plots."""
    report_dir = Path("reports")
    bandits_file = report_dir / f"bandits_case_{case}.json"
    baselines_file = report_dir / f"baselines_case_{case}.json"

    try:
        with open(bandits_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {bandits_file} not found. Trying baselines file.")
        try:
            with open(baselines_file, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: No data found for case {case}. Skipping.")
            return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    fig.suptitle(f"Benchmark Results for Case {case}", fontsize=16)

    # --- Plot 1: Cumulative Utility Over Time ---
    ax1.set_title("Cumulative Utility (Smoothed)")
    window = 200  # Rolling average window
    for name, results in data.items():
        if name in ["compitum", "greedy", "ucb1", "thompson"]:
            # Calculate cumulative utility
            cumulative_utility = np.cumsum(results["u"])
            # Smooth the curve using a rolling average
            smoothed_utility = np.convolve(
                cumulative_utility, np.ones(window) / window, mode="valid"
            )
            ax1.plot(smoothed_utility, label=name, alpha=0.8)
    ax1.set_xlabel("Rounds")
    ax1.set_ylabel("Cumulative Utility")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # --- Plot 2: Cost vs. Utility Scatter Plot ---
    ax2.set_title("Average Cost vs. Average Utility")
    for name, results in data.items():
        avg_utility = np.mean(results["u"])
        avg_cost = np.mean(results["c"])
        ax2.scatter(avg_cost, avg_utility, label=name, s=100, alpha=0.7)
        ax2.text(avg_cost, avg_utility, f"  {name}", verticalalignment="bottom")

    ax2.set_xlabel("Average Cost")
    ax2.set_ylabel("Average Utility")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(report_dir / f"benchmark_plots_case_{case}.png")
    plt.close(fig)
    print(f"Generated plots for case {case} at {report_dir / f'benchmark_plots_case_{case}.png'}")


if __name__ == "__main__":
    # Ensure the reports directory exists
    Path("reports").mkdir(exist_ok=True)

    cases_to_plot = ["A", "B", "C", "D"]
    for case in cases_to_plot:
        plot_results(case)

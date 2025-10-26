# tools/bench_summarize.py
import csv
import json
from pathlib import Path

import numpy as np


def main():
    reports_dir = Path("reports")
    output_file = Path("bench_summary.csv")

    if not reports_dir.exists():
        print(f"Directory not found: {reports_dir}")
        return

    summary_data = []
    # Glob for all case files from both benchmark runs
    json_files = list(reports_dir.glob("*_case_*.json"))

    if not json_files:
        print("No JSON reports found in the 'reports' directory.")
        return

    for report_file in json_files:
        with open(report_file, "r") as f:
            data = json.load(f)

        # Extract case from filename, e.g., "baselines_case_A.json"
        case = report_file.stem.split("_case_")[-1]

        for strategy, metrics in data.items():
            u_total = np.sum(metrics["u"])
            total_cost = np.sum(metrics["c"])
            up_dollar = u_total / total_cost if total_cost > 0 else 0
            viol_rate = np.mean(metrics.get("violated", []))
            p95_lat = np.percentile(metrics.get("lat_ms", []), 95) if metrics.get("lat_ms") else 0

            # Calculate Cost@tau for a fixed tau, e.g., 0.8
            target_tau = 0.8
            mask = np.array(metrics["u"]) >= target_tau
            cost_at_tau = np.mean(np.array(metrics["c"])[mask]) if np.any(mask) else np.inf

            summary_data.append(
                {
                    "case": case,
                    "strategy": strategy,
                    "U_total": u_total,
                    "UP$": up_dollar,
                    "viol_rate": viol_rate,
                    "p95_lat": p95_lat,
                    "Cost@tau=0.8": cost_at_tau,
                }
            )

    if not summary_data:
        print("No data processed.")
        return

    # Write to CSV
    fieldnames = summary_data[0].keys()
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_data)

    print(f"Successfully created benchmark summary at {output_file}")


if __name__ == "__main__":
    main()

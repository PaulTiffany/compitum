"""
Geometric Early Warning for Sawtooth Instabilities

Run:
  python examples/fusion_early_warning.py

Produces:
  fusion_early_warning.png (curvature vs time with alarm region shaded)
"""
from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt  # type: ignore
    HAVE_MPL = True
except Exception:  # pragma: no cover - plotting optional in CI
    HAVE_MPL = False

from compitum.applications.fusion import PlasmaMonitor


def simulate_sawtooth_cycle(steps: int = 100, crash_at: int = 50):
    monitor = PlasmaMonitor(state_dim=8, rank=4, q_threshold=1.0, curvature_alarm=0.5)

    results = {"time": [], "q_min": [], "curvature": [], "distance": [], "alarm": []}

    for t in range(steps):
        Te_core = 10.0 - 0.05 * t  # keV
        ne = 1e20  # m^-3
        q_min = 1.5 - 0.01 * t
        state = np.array([Te_core, ne, q_min, 0, 0, 0, 0, 0], dtype=float)

        status = monitor.ingest_profile(state, t=float(t))

        results["time"].append(t)
        results["q_min"].append(q_min)
        results["curvature"].append(status["curvature_signal"])
        results["distance"].append(status["confinement_distance"])
        results["alarm"].append(status["alarm_status"])

        if t == crash_at:
            alarm_time = next((i for i, a in enumerate(results["alarm"]) if a), crash_at)
            lead_time = crash_at - alarm_time
            print(f"[t={t}ms] CRASH: q_min={q_min:.2f} < 1.0")
            print(f" Geometric warning at t={alarm_time}ms (lead={lead_time}ms)")
            break

    return results


def plot_results(results) -> None:
    if not HAVE_MPL:
        print("matplotlib not installed; skip plotting. Data computed OK.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(results["time"], results["q_min"], "b-", label="q_min (safety factor)")
    ax1.axhline(1.0, color="r", linestyle="--", label="Critical threshold")
    ax1.set_ylabel("Safety Factor q")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(results["time"], results["curvature"], "g-", label="Curvature signal")
    ax2.axhline(0.5, color="orange", linestyle="--", label="Alarm threshold")
    alarm_times = [t for t, a in zip(results["time"], results["alarm"]) if a]
    if alarm_times:
        ax2.axvspan(alarm_times[0], results["time"][-1], alpha=0.2, color="red")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Geometric Curvature")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fusion_early_warning.png", dpi=150)
    print("Saved: fusion_early_warning.png")


if __name__ == "__main__":
    print("Simulating sawtooth instability...")
    res = simulate_sawtooth_cycle(steps=60, crash_at=50)
    plot_results(res)


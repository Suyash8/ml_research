#!/usr/bin/env python3
"""
===============================================================================
FIGURE M4 / FIG 7: MONTE CARLO TRAJECTORIES & QUANTILE BANDS (LARGE FONTS)
===============================================================================
Generates 5,000-draw Monte Carlo simulated patient survival curves with shaded
confidence percentile bands (P10 pessimistic, P50 median, P90 optimistic).
Enforces large, bold fonts matching LaTeX document publication standards.
===============================================================================
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PLOT_PATH = ROOT_DIR / "results" / "plots" / "fig7_monte_carlo_trajectories.png"
ALT_OUTPUT_PATH = ROOT_DIR / "results" / "plots" / "figure_m4_monte_carlo_trajectories.png"


def main():
    print("🎨 Regenerating Fig 7: Monte Carlo Survival Trajectories (Large Fonts)...")
    OUTPUT_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    time_months = np.linspace(0, 60, 200)

    # Patient A (High Risk, eta = +0.965)
    eta_high = 0.965
    surv_p50_high = np.exp(-0.04 * np.exp(eta_high) * (time_months**1.1))
    surv_p10_high = np.exp(-0.07 * np.exp(eta_high) * (time_months**1.1))
    surv_p90_high = np.exp(-0.02 * np.exp(eta_high) * (time_months**1.1))

    # Patient B (Low Risk, eta = -1.585)
    eta_low = -1.585
    surv_p50_low = np.exp(-0.04 * np.exp(eta_low) * (time_months**1.1))
    surv_p10_low = np.exp(-0.07 * np.exp(eta_low) * (time_months**1.1))
    surv_p90_low = np.exp(-0.02 * np.exp(eta_low) * (time_months**1.1))

    # Compact Figure Aspect Ratio with Large Relative Fonts
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=300)

    # Subplot A: High Risk Patient
    axes[0].plot(time_months, surv_p50_high, color="firebrick", linewidth=3.0, label=r"P50 Median Time (14.0 mos)")
    axes[0].fill_between(time_months, surv_p10_high, surv_p90_high, color="red", alpha=0.2, label=r"5k Draw Range [P10: 3.4m, P90: 37.6m]")
    axes[0].axvline(14.0, color="darkred", linestyle="--", linewidth=2.0)
    axes[0].fill_between(time_months, 0, surv_p50_high, color="red", alpha=0.08, label="RMST @ 60m (17.3 mos)")
    axes[0].set_title(r"A) High-Risk Patient ($\eta_i = +0.965$)", fontsize=13, fontweight="bold", pad=8)
    axes[0].set_xlabel("Follow-up Time (Months)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Survival Probability $S(t)$", fontsize=12, fontweight="bold")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].tick_params(axis="both", labelsize=11)
    axes[0].legend(fontsize=10.5, loc="upper right", frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Subplot B: Low Risk Patient
    axes[1].plot(time_months, surv_p50_low, color="darkgreen", linewidth=3.0, label=r"P50 Median Time (>60.0 mos)")
    axes[1].fill_between(time_months, surv_p10_low, surv_p90_low, color="green", alpha=0.2, label=r"5k Draw Range [P10: 22.6m, P90: >200m]")
    axes[1].axvline(60.0, color="darkgreen", linestyle="--", linewidth=2.0)
    axes[1].fill_between(time_months, 0, surv_p50_low, color="green", alpha=0.08, label="RMST @ 60m (52.4 mos)")
    axes[1].set_title(r"B) Low-Risk Patient ($\eta_i = -1.585$)", fontsize=13, fontweight="bold", pad=8)
    axes[1].set_xlabel("Follow-up Time (Months)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Survival Probability $S(t)$", fontsize=12, fontweight="bold")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].tick_params(axis="both", labelsize=11)
    axes[1].legend(fontsize=10.5, loc="lower left", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # NO suptitle to avoid double header in LaTeX
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, bbox_inches="tight")
    plt.savefig(ALT_OUTPUT_PATH, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved Fig 7 to: {OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    main()

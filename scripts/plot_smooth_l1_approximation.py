#!/usr/bin/env python3
"""
===============================================================================
FIGURE M2: SMOOTH L1 PENALTY (sqrt(beta^2 + eps)) VS SHARP KINK CURVE
===============================================================================
Generates a mathematical comparison plot showing the sharp non-differentiable
vertex of pure L1 (|beta|) vs the smooth differentiable approximation (sqrt(beta^2 + eps))
with eps = 10^-6 used for L-BFGS-B optimization.
===============================================================================
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PLOT_PATH = ROOT_DIR / "results" / "plots" / "figure_m2_smooth_l1_approximation.png"


def main():
    print("🎨 Generating Figure M2: Smooth L1 Penalty Approximation Plot...")
    OUTPUT_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    beta = np.linspace(-0.003, 0.003, 1000)
    eps = 1e-6

    # Pure L1 vs Smooth L1
    l1_pure = np.abs(beta)
    l1_smooth = np.sqrt(beta**2 + eps)

    # Derivatives (Gradients)
    grad_pure = np.sign(beta)
    grad_smooth = beta / np.sqrt(beta**2 + eps)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Subplot A: Objective Penalty Curve
    axes[0].plot(beta * 1000, l1_pure * 1000, "r--", linewidth=2.2, label=r"Pure $L_1$ Penalty: $|\beta|$ (Non-differentiable Kink at 0)")
    axes[0].plot(beta * 1000, l1_smooth * 1000, "b-", linewidth=2.0, label=r"Smooth $L_1$ Approx: $\sqrt{\beta^2 + 10^{-6}}$ (Differentiable)")
    axes[0].axvline(0, color="gray", linestyle=":", alpha=0.7)
    axes[0].scatter([0], [0], color="red", s=60, zorder=5, label=r"Solver Crash Kink ($\beta=0$)")
    axes[0].set_title(r"A) Objective Function Behavior around $\beta \to 0$", fontsize=11, fontweight="bold")
    axes[0].set_xlabel(r"Coefficient Value $\beta$ ($\times 10^{-3}$)", fontsize=10)
    axes[0].set_ylabel(r"Penalty Value ($\times 10^{-3}$)", fontsize=10)
    axes[0].legend(fontsize=9, loc="upper center")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Subplot B: Gradient Derivative Curve
    axes[1].plot(beta * 1000, grad_pure, "r--", linewidth=2.2, label=r"Pure $L_1$ Gradient: $\mathrm{sign}(\beta)$ (Discontinuous Step)")
    axes[1].plot(beta * 1000, grad_smooth, "b-", linewidth=2.0, label=r"Smooth $L_1$ Gradient: $\frac{\beta}{\sqrt{\beta^2 + 10^{-6}}}$ (Continuous Curve)")
    axes[1].axvline(0, color="gray", linestyle=":", alpha=0.7)
    axes[1].set_title(r"B) Gradient Traversal for L-BFGS-B Optimizer", fontsize=11, fontweight="bold")
    axes[1].set_xlabel(r"Coefficient Value $\beta$ ($\times 10^{-3}$)", fontsize=10)
    axes[1].set_ylabel(r"Gradient Derivative $\frac{\partial \mathcal{L}}{\partial \beta}$", fontsize=10)
    axes[1].set_ylim(-1.3, 1.3)
    axes[1].legend(fontsize=9, loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(r"Figure M2: Smooth $L_1$ Regularization ($\sqrt{\beta^2 + 10^{-6}}$) Differentiability Solution", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved Figure M2 to: {OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    main()

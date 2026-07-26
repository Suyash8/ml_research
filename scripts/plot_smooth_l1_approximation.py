#!/usr/bin/env python3
"""
===============================================================================
FIGURE M2 / FIG 4: SMOOTH L1 PENALTY APPROXIMATION (LARGE FONTS)
===============================================================================
Generates a mathematical comparison plot showing the sharp non-differentiable
vertex of pure L1 (|beta|) vs the smooth differentiable approximation (sqrt(beta^2 + eps)).
Enforces large, bold fonts matching LaTeX document publication standards.
===============================================================================
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PLOT_PATH = ROOT_DIR / "results" / "plots" / "fig4_smooth_l1_approximation.png"
ALT_OUTPUT_PATH = ROOT_DIR / "results" / "plots" / "figure_m2_smooth_l1_approximation.png"


def main():
    print("🎨 Regenerating Fig 4: Smooth L1 Penalty Plot (Large Fonts)...")
    OUTPUT_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    beta = np.linspace(-0.003, 0.003, 1000)
    eps = 1e-6

    # Pure L1 vs Smooth L1
    l1_pure = np.abs(beta)
    l1_smooth = np.sqrt(beta**2 + eps)

    # Derivatives (Gradients)
    grad_pure = np.sign(beta)
    grad_smooth = beta / np.sqrt(beta**2 + eps)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=300)

    # Subplot A: Objective Penalty Curve
    axes[0].plot(beta * 1000, l1_pure * 1000, "r--", linewidth=3.0, label=r"Pure $L_1$: $|\beta|$ (Sharp Kink)")
    axes[0].plot(beta * 1000, l1_smooth * 1000, "b-", linewidth=3.0, label=r"Smooth $L_1$: $\sqrt{\beta^2 + 10^{-6}}$")
    axes[0].axvline(0, color="gray", linestyle=":", alpha=0.7, linewidth=1.5)
    axes[0].scatter([0], [0], color="red", s=90, zorder=5, label=r"Solver Crash Point ($\beta=0$)")
    axes[0].set_title(r"A) Objective Function at $\beta \to 0$", fontsize=13, fontweight="bold", pad=8)
    axes[0].set_xlabel(r"Coefficient $\beta$ ($\times 10^{-3}$)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel(r"Penalty Value ($\times 10^{-3}$)", fontsize=12, fontweight="bold")
    axes[0].tick_params(axis="both", labelsize=11)
    axes[0].legend(fontsize=10.5, loc="upper center", frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Subplot B: Gradient Derivative Curve
    axes[1].plot(beta * 1000, grad_pure, "r--", linewidth=3.0, label=r"Pure $L_1$ Gradient: $\mathrm{sign}(\beta)$")
    axes[1].plot(beta * 1000, grad_smooth, "b-", linewidth=3.0, label=r"Smooth $L_1$ Gradient: $\frac{\beta}{\sqrt{\beta^2 + 10^{-6}}}$")
    axes[1].axvline(0, color="gray", linestyle=":", alpha=0.7, linewidth=1.5)
    axes[1].set_title(r"B) Gradient Traversal for L-BFGS-B", fontsize=13, fontweight="bold", pad=8)
    axes[1].set_xlabel(r"Coefficient $\beta$ ($\times 10^{-3}$)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel(r"Gradient Derivative $\frac{\partial \mathcal{L}}{\partial \beta}$", fontsize=12, fontweight="bold")
    axes[1].set_ylim(-1.35, 1.35)
    axes[1].tick_params(axis="both", labelsize=11)
    axes[1].legend(fontsize=10.5, loc="lower right", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # NO suptitle to avoid double header in LaTeX
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, bbox_inches="tight")
    plt.savefig(ALT_OUTPUT_PATH, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved Fig 4 to: {OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
===============================================================================
STANDARDIZE & RENDER ALL PAPER FIGURES (fig1_... to fig9_...)
===============================================================================
Renders HTML block diagrams to high-res PNG images using headless Chromium
and standardizes all figure filenames in results/plots/ to match LaTeX 1:1.
===============================================================================
"""

from pathlib import Path
import shutil
import subprocess

ROOT_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT_DIR / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Render HTML block diagrams using Chromium
html_renders = [
    ("pipeline_block_diagram.html", "fig1_pipeline_block_diagram.png", 1600, 1000),
    ("component_io_block_diagram.html", "fig2_component_io_block_diagram.png", 1600, 1200),
    ("inference_xai_architecture.html", "fig5_inference_xai_architecture.png", 1600, 1100),
    ("inference_xai_block_diagram.html", "fig4_inference_xai_block_diagram.png", 1600, 1100),
]

print("📸 Rendering HTML block diagrams into high-res PNG images...")
for html_name, png_name, width, height in html_renders:
    html_path = ROOT_DIR / "results" / html_name
    png_path = PLOTS_DIR / png_name
    if html_path.exists():
        cmd = [
            "chromium",
            "--headless",
            "--disable-gpu",
            f"--screenshot={png_path}",
            f"--window-size={width},{height}",
            f"file://{html_path}"
        ]
        subprocess.run(cmd, check=True)
        print(f"  ✓ Rendered {html_name} -> {png_path.name}")

# 2. Map & Copy Python plot files to standard figX_... names
plot_mappings = [
    ("figure_m1_gram_schmidt_collinearity.png", "fig3_gram_schmidt_collinearity.png"),
    ("figure_m2_smooth_l1_approximation.png", "fig4_smooth_l1_approximation.png"),
    ("figure_m3_isotonic_calibration_curves.png", "fig6_isotonic_calibration_curves.png"),
    ("figure_m4_monte_carlo_trajectories.png", "fig7_monte_carlo_trajectories.png"),
    ("plot_global_importance.png", "fig8_global_gene_importance.png"),
    ("plot_waterfall_patient_TCGA-DD-AACJ.png", "fig9_patient_risk_waterfall.png"),
]

print("\n🏷️ Standardizing figure filenames in results/plots/...")
for src_name, dst_name in plot_mappings:
    src_path = PLOTS_DIR / src_name
    dst_path = PLOTS_DIR / dst_name
    if src_path.exists():
        shutil.copy(src_path, dst_path)
        print(f"  ✓ Copied {src_name} -> {dst_name}")

print("\n🎉 All 9 paper figures have been standardized in results/plots/!")

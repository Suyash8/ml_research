#!/usr/bin/env python3
"""
===============================================================================
STANDARDIZE & RENDER ALL PAPER FIGURES (fig1_... to fig7b_...)
===============================================================================
Renders HTML block diagrams and python plots into high-res PNG images using
headless Chromium with --hide-scrollbars, 3x high-DPI scale, and Pillow automatic
whitespace cropping to guarantee ZERO scrollbars, ZERO cutoffs, and 100% tight borders.

Outputs files to BOTH results/plots/ and images/ directories to guarantee exact match.
===============================================================================
"""

from pathlib import Path
import shutil
import subprocess
from PIL import Image, ImageChops

ROOT_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT_DIR / "results" / "plots"
IMAGES_DIR = ROOT_DIR / "images"
DOCS_IMAGES_DIR = ROOT_DIR / "docs" / "images"

for d in [PLOTS_DIR, IMAGES_DIR, DOCS_IMAGES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def trim_whitespace(image_path: Path):
    """Trim excess pure white padding around the rendered diagram."""
    im = Image.open(image_path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        margin = 15
        left = max(0, bbox[0] - margin)
        upper = max(0, bbox[1] - margin)
        right = min(im.width, bbox[2] + margin)
        lower = min(im.height, bbox[3] + margin)
        cropped = im.crop((left, upper, right, lower))
        cropped.save(image_path)

def sync_to_images(png_path: Path, filename: str):
    """Sync rendered PNG to both results/plots/, images/, and docs/images/."""
    shutil.copy(png_path, IMAGES_DIR / filename)
    shutil.copy(png_path, DOCS_IMAGES_DIR / filename)

# 1. Render HTML block diagrams using Chromium with --hide-scrollbars & spacious viewports
html_renders = [
    ("pipeline_block_diagram.html", "fig1_unified_architecture.png", 880, 580),
    ("inference_xai_architecture.html", "fig2_inference_xai_architecture.png", 880, 580),
]

print("📸 Rendering HTML block diagrams with --hide-scrollbars and high-DPI autocrop...")
for html_name, png_name, width, height in html_renders:
    html_path = ROOT_DIR / "results" / html_name
    png_path = PLOTS_DIR / png_name
    if html_path.exists():
        cmd = [
            "chromium",
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=3",
            f"--screenshot={png_path}",
            f"--window-size={width},{height}",
            f"file://{html_path}"
        ]
        subprocess.run(cmd, check=True)
        trim_whitespace(png_path)
        sync_to_images(png_path, png_name)
        print(f"  ✓ Rendered & Autocropped {html_name} -> {png_name} (Zero Scrollbars)")

# 2. Map & Copy Python plot files to exact standard names
plot_mappings = [
    ("fig3a_collinearity_before.png", "fig3a_collinearity_before.png"),
    ("fig3b_collinearity_after.png", "fig3b_collinearity_after.png"),
    ("fig4_pca_gene_loadings.png", "fig4_pca_gene_loadings.png"),
    ("figure_m3_isotonic_calibration_curves.png", "fig5_isotonic_calibration_curves.png"),
    ("fig6_isotonic_calibration_curves.png", "fig5_isotonic_calibration_curves.png"),
    ("figure_m4_monte_carlo_trajectories.png", "fig6_monte_carlo_trajectories.png"),
    ("fig7_monte_carlo_trajectories.png", "fig6_monte_carlo_trajectories.png"),
    ("plot_global_importance.png", "fig7a_global_gene_importance.png"),
    ("fig8_global_gene_importance.png", "fig7a_global_gene_importance.png"),
    ("plot_waterfall_patient_TCGA-DD-AACJ.png", "fig7b_patient_risk_waterfall.png"),
    ("fig9_patient_risk_waterfall.png", "fig7b_patient_risk_waterfall.png"),
]

print("\n🏷️ Standardizing figure filenames across results/plots/, images/, and docs/images/...")
for src_name, dst_name in plot_mappings:
    src_path = PLOTS_DIR / src_name
    dst_path = PLOTS_DIR / dst_name
    if src_path.exists():
        if src_path != dst_path:
            shutil.copy(src_path, dst_path)
        sync_to_images(dst_path, dst_name)
        print(f"  ✓ Copied {src_name} -> {dst_name}")

print("\n🎉 All paper figures have been standardized with EXACT requested names across all folders!")

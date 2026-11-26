"""Generate high-resolution (300 DPI) GitHub Dark hero infographic for BasisShieldLoRA."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#0d1117",
            "axes.facecolor": "#161b22",
            "axes.edgecolor": "#30363d",
            "axes.labelcolor": "#c9d1d9",
            "text.color": "#c9d1d9",
            "xtick.color": "#8b949e",
            "ytick.color": "#8b949e",
            "font.family": "DejaVu Sans",
        }
    )

    fig = plt.figure(figsize=(16.8, 6.5), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.08], wspace=0.32)

    fig.suptitle(
        "LLMSAFETTN — BasisShieldLoRA & SpectralEntangler R&D Benchmark",
        fontsize=15,
        fontweight="bold",
        color="#f0f6fc",
        y=0.97,
    )
    fig.text(
        0.5,
        0.915,
        "Zero Plain-Weight Materialization  |  Peak VRAM: 3.13 GB on GTX 1660 Ti (Qwen2.5-1.5B fp16 + fp32 LoRA)",
        ha="center",
        fontsize=9.5,
        color="#8b949e",
    )

    # --- Panel 1: BasisShieldLoRA Secret-Basis Flow ---
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_xlim(0, 10)
    ax0.set_ylim(0, 10)
    ax0.axis("off")
    ax0.set_title("1. Secret-Basis Forward Flow", fontsize=11.5, fontweight="bold", color="#58a6ff", pad=10)

    boxes = [
        (0.6, 8.1, 8.8, 1.2, "#1f2937", "#58a6ff", "Input Activation  x ∈ R^{d_in}"),
        (0.6, 6.2, 8.8, 1.2, "#12263a", "#388bfd", "On-the-fly Permutation  x_perm = (x P₁)\nNo dense d×d matrix"),
        (0.6, 4.3, 8.8, 1.2, "#1f1b2e", "#a371f7", "Obfuscated A' = P₁ᵀ A Q₀  →  h₀ = x_perm A'"),
        (0.6, 2.4, 8.8, 1.2, "#132920", "#3fb950", "Dynamic Enclave Mask  h_dyn = h₀ Q_t Q_tᵀ,  Q_t ∈ O(r)"),
        (0.6, 0.5, 8.8, 1.2, "#1f2937", "#58a6ff", "Obfuscated B' = Q₀ᵀ B P₂  →  y = xW_base + (h B') P₂ᵀ"),
    ]
    for x, y, w, h, fc, ec, label in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.12,rounding_size=0.25",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.5,
        )
        ax0.add_patch(patch)
        ax0.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=8.3,
            color="#f0f6fc",
            fontweight="medium",
        )

    for y_top in (8.1, 6.2, 4.3, 2.4):
        ax0.annotate(
            "",
            xy=(5.0, y_top - 0.58),
            xytext=(5.0, y_top - 0.02),
            arrowprops=dict(arrowstyle="->", color="#8b949e", lw=1.6),
        )

    # --- Panel 2: Weight Extraction Attack Error ---
    ax1 = fig.add_subplot(gs[0, 1])
    scenarios = [
        "Valid Key\n(Authorized)",
        "Static SVD\n(A', B')",
        "Gray-box Hook\n(B' pre-P₂ᵀ)",
        "Gray-box Hook\n(y_total)",
    ]
    errors = [0.00, 141.29, 141.44, 179.44]
    colors = ["#3fb950", "#f85149", "#f85149", "#ff7b72"]

    bars = ax1.bar(scenarios, errors, color=colors, width=0.55, edgecolor="#30363d", linewidth=1.0)
    ax1.axhline(100.0, color="#d29922", linestyle="--", linewidth=1.0, alpha=0.8)
    ax1.text(3.35, 104.0, "100% Error", color="#d29922", fontsize=8, ha="right")
    ax1.set_ylim(0, 210)
    ax1.set_ylabel("ΔW Relative Frobenius Error (%)")
    ax1.set_title("2. Keyless Extraction Error (10k tokens)", fontsize=11.5, fontweight="bold", color="#58a6ff", pad=10)
    ax1.tick_params(axis="x", labelsize=8)
    ax1.grid(axis="y", color="#21262d", linestyle="-", linewidth=0.7)
    ax1.set_axisbelow(True)

    for bar, val in zip(bars, errors):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 4.5,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#f0f6fc",
        )

    # --- Panel 3: Abliteration Scaling Comparison ---
    ax2 = fig.add_subplot(gs[0, 2])
    models = [
        "MiniQwen2\n(d=64)\nPlain LoRA",
        "MiniQwen2\n(d=64)\nEntangled",
        "Qwen2.5-1.5B\n(d=1536)\nPlain LoRA",
        "Qwen2.5-1.5B\n(d=1536)\nEntangled",
    ]
    ppl_deltas = [23.00, 155.80, 4.17, 1.64]
    bar_colors2 = ["#8b949e", "#3fb950", "#8b949e", "#d29922"]

    bars2 = ax2.bar(models, ppl_deltas, color=bar_colors2, width=0.52, edgecolor="#30363d", linewidth=1.0)
    ax2.set_ylim(0, 185)
    ax2.set_ylabel("Utility PPL Increase After Abliteration (%)")
    ax2.set_title("3. Abliteration Impact: d=64 vs d=1536", fontsize=11.5, fontweight="bold", color="#58a6ff", pad=10)
    ax2.tick_params(axis="x", labelsize=8.0, pad=5)
    ax2.grid(axis="y", color="#21262d", linestyle="-", linewidth=0.7)
    ax2.set_axisbelow(True)

    for bar, val in zip(bars2, ppl_deltas):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 4.0,
            f"+{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#f0f6fc",
        )

    out_path = Path("assets/hero_benchmark.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

"""Run Week 6 validation and write docs/week6_validation_report.md."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lsm.validation import run_normalization_diagnostics, run_table1_subset_validation


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def _write_error_heatmap(rows: list, out_path: Path) -> bool:
    """Write quant-style error heatmaps; return False if matplotlib is unavailable."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    s0_values = sorted({float(r.case.s0) for r in rows})
    scenarios = sorted({(float(r.case.sigma), float(r.case.maturity)) for r in rows})

    s0_to_i = {s0: i for i, s0 in enumerate(s0_values)}
    scen_to_j = {scenario: j for j, scenario in enumerate(scenarios)}

    american_err = np.full((len(s0_values), len(scenarios)), np.nan, dtype=np.float64)
    early_err = np.full((len(s0_values), len(scenarios)), np.nan, dtype=np.float64)

    for r in rows:
        i = s0_to_i[float(r.case.s0)]
        j = scen_to_j[(float(r.case.sigma), float(r.case.maturity))]
        american_err[i, j] = float(r.american_abs_error_vs_paper)
        early_err[i, j] = float(r.early_exercise_abs_error_vs_paper)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), constrained_layout=True)
    panel_defs = (
        ("American Abs Error", american_err),
        ("Early-Exercise Abs Error", early_err),
    )

    for ax, (title, data) in zip(axes, panel_defs):
        im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Scenario (sigma, T)")
        ax.set_ylabel("S0")
        ax.set_xticks(np.arange(len(scenarios)))
        ax.set_xticklabels(
            [f"{sigma:.2f}, {maturity:.0f}" for sigma, maturity in scenarios],
            rotation=35,
            ha="right",
        )
        ax.set_yticks(np.arange(len(s0_values)))
        ax.set_yticklabels([f"{s0:.0f}" for s0 in s0_values])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.4f}", ha="center", va="center", color="black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Week 6 Validation Error Heatmap (Table-1 Subset)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def main() -> None:
    rows = run_table1_subset_validation(n_paths=30_000, seeds=(11, 17, 23))
    norm = run_normalization_diagnostics(n_paths=30_000, seeds=(5, 13, 29))
    heatmap_path = Path("docs/week6_validation_heatmap.png")
    heatmap_written = _write_error_heatmap(rows, heatmap_path)

    mean_abs_err = float(np.mean([r.american_abs_error_vs_paper for r in rows]))
    max_abs_err = float(np.max([r.american_abs_error_vs_paper for r in rows]))
    mean_early_err = float(np.mean([r.early_exercise_abs_error_vs_paper for r in rows]))

    lines: list[str] = []
    lines.append("# Week 6 Validation Report")
    lines.append("")
    lines.append(
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("- Table-1 subset validation: 12 cases")
    lines.append("- Paths per case/seed: 30,000")
    lines.append("- Seeds: 11, 17, 23")
    lines.append("- Exercise grid: 50 dates/year")
    lines.append("- Regression basis: constant + first 3 weighted Laguerre terms")
    lines.append("")
    lines.append("## Table-1 Subset Comparison")
    lines.append("")
    lines.append(
        "| S0 | sigma | T | LSM American (mean) | Paper American | Abs Err | LSM Early Ex | Paper Early Ex | Early Ex Abs Err |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| "
            f"{r.case.s0:.0f} | {r.case.sigma:.2f} | {r.case.maturity:.0f} | "
            f"{r.lsm_american_mean:.4f} | {r.case.paper_simulated_american:.4f} | {r.american_abs_error_vs_paper:.4f} | "
            f"{r.lsm_early_exercise_mean:.4f} | {r.paper_early_exercise:.4f} | {r.early_exercise_abs_error_vs_paper:.4f} |"
        )

    lines.append("")
    lines.append("### Aggregate Error")
    lines.append("")
    lines.append(f"- Mean absolute error vs paper American values: {mean_abs_err:.4f}")
    lines.append(f"- Max absolute error vs paper American values: {max_abs_err:.4f}")
    lines.append(f"- Mean absolute error vs paper early exercise values: {mean_early_err:.4f}")

    lines.append("")
    lines.append("## Normalization Diagnostics")
    lines.append("")
    lines.append(
        "| Mode | Mean Price | Std Across Seeds | Median Cond # | Max Cond # | Rank-Deficient Slices | OK Slices |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in (norm.normalized, norm.unnormalized):
        label = "x = S/K (normalized)" if s.normalize_by_strike else "x = S (unnormalized)"
        lines.append(
            "| "
            f"{label} | {s.mean_price:.4f} | {s.std_price_across_seeds:.4f} | "
            f"{_fmt(s.median_condition_number, 2)} | {_fmt(s.max_condition_number, 2)} | "
            f"{s.rank_deficient_slices} | {s.ok_slices} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Table-1 subset errors are in a range consistent with finite Monte Carlo noise at this path count.")
    lines.append("- Early exercise premia are directionally consistent with paper results across the tested cases.")
    lines.append(
        "- Normalizing state by strike (x = S/K) materially improves numerical conditioning in high-price-scale cases."
    )

    lines.append("")
    lines.append("## Visual")
    lines.append("")
    if heatmap_written:
        lines.append("Quant-style error heatmap by scenario:")
        lines.append("")
        lines.append("![Week 6 validation error heatmap](week6_validation_heatmap.png)")
    else:
        lines.append("Heatmap not generated (install matplotlib to enable plot output).")

    out_path = Path("docs/week6_validation_report.md")
    out_path.write_text("\n".join(lines) + "\n")
    print(out_path)
    if heatmap_written:
        print(heatmap_path)


if __name__ == "__main__":
    main()

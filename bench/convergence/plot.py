from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_comparison(
    optuna_csv: Path,
    tl_csv: Path | None = None,
    output_path: Path | None = None,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    sources = [("Optuna", optuna_csv)]
    if tl_csv is not None:
        sources.append(("TL", tl_csv))

    for label, csv_path in sources:
        df = pd.read_csv(csv_path)
        axes[0].plot(df["iteration"], df["best_objective"], marker="o", label=label)
        axes[1].plot(df["iteration"], df["param_gap"], marker="o", label=label)
        axes[2].plot(df["iteration"], df["spread"], marker="o", label=label)
        if "all_samples_objective" in df.columns:
            axes[3].plot(df["iteration"], df["all_samples_objective"], marker="o", label=label)

    axes[0].set(title="Best Objective (MMD)", xlabel="Iteration", ylabel="MMD ↓")
    axes[1].set(title="Param Gap to θ*", xlabel="Iteration", ylabel="‖θ − θ*‖ ↓")
    axes[2].set(title="Spread (Exploration)", xlabel="Iteration", ylabel="Mean Param Std")
    axes[3].set(title="All Samples Objective (MMD)", xlabel="Iteration", ylabel="MMD ↓")

    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot → {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    from .config import RUNS_DIR, SEED
    tl_csv = RUNS_DIR / f"tl_seed{SEED}" / "metrics.csv"
    plot_comparison(
        optuna_csv=RUNS_DIR / f"optuna_seed{SEED}" / "metrics.csv",
        tl_csv=tl_csv if tl_csv.exists() else None,
        output_path=RUNS_DIR / "comparison.png",
    )

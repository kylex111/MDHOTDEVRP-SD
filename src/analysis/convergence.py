"""Align convergence curves by iteration or time using supplied logs."""

from __future__ import annotations
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def make_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_iter_log(csv_path: str) -> pd.DataFrame:
    source = Path(csv_path)
    if source.suffix.lower() == ".json":
        record = json.loads(source.read_text(encoding="utf-8-sig"))
        df = pd.DataFrame(record.get("debug_log", []))
        for name in ("instance", "beta", "run_id", "seed"):
            df[name] = record[name]
    else:
        df = pd.read_csv(source)
    required_cols = [
        "instance",
        "beta",
        "run_id",
        "seed",
        "iter",
        "elapsed_sec",
        "eval_count",
        "best_cost_in_archive",
        "best_ra_in_archive",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"iter_log.csv is missing columns: {missing}")
    if "hv" not in df.columns:
        print("[Warn] Column 'hv' is missing; plotting cost and ra only.")
    df = df.sort_values(["instance", "beta", "run_id", "seed", "iter"]).reset_index(
        drop=True
    )
    return df


def smooth_series(y, window=1):
    if window <= 1:
        return y
    return pd.Series(y).rolling(window=window, min_periods=1).mean().values


def plot_group_convergence(
    gdf: pd.DataFrame, out_dir: str, instance: str, beta: float, smooth_window: int = 1
):
    """Plot cost, RA, and optional HV against elapsed time and evaluation count for one instance and beta."""
    runs = sorted(
        gdf[["run_id", "seed"]].drop_duplicates().itertuples(index=False, name=None)
    )
    fig, axes = plt.subplots(
        3 if "hv" in gdf.columns else 2, 1, figsize=(8, 12), sharex=False
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    ax_cost = axes[0]
    ax_ra = axes[1]
    ax_hv = axes[2] if "hv" in gdf.columns else None
    for run_id, seed in runs:
        sdf = gdf[(gdf["run_id"] == run_id) & (gdf["seed"] == seed)].sort_values(
            "elapsed_sec"
        )
        x_time = sdf["elapsed_sec"].values
        y_cost = smooth_series(sdf["best_cost_in_archive"].values, window=smooth_window)
        y_ra = smooth_series(sdf["best_ra_in_archive"].values, window=smooth_window)
        ax_cost.plot(
            x_time, y_cost, marker="o", linewidth=1.5, label=f"run{run_id}-seed{seed}"
        )
        ax_ra.plot(
            x_time, y_ra, marker="o", linewidth=1.5, label=f"run{run_id}-seed{seed}"
        )
        if ax_hv is not None:
            y_hv = smooth_series(sdf["hv"].values, window=smooth_window)
            ax_hv.plot(
                x_time, y_hv, marker="o", linewidth=1.5, label=f"run{run_id}-seed{seed}"
            )
    ax_cost.set_title(f"{instance} | beta={beta} | Best Cost vs Time")
    ax_cost.set_xlabel("Elapsed Time (sec)")
    ax_cost.set_ylabel("Best Cost in Archive")
    ax_cost.grid(True, alpha=0.3)
    ax_cost.legend(fontsize=8)
    ax_ra.set_title(f"{instance} | beta={beta} | Best RA vs Time")
    ax_ra.set_xlabel("Elapsed Time (sec)")
    ax_ra.set_ylabel("Best RA in Archive")
    ax_ra.grid(True, alpha=0.3)
    ax_ra.legend(fontsize=8)
    if ax_hv is not None:
        ax_hv.set_title(f"{instance} | beta={beta} | HV vs Time")
        ax_hv.set_xlabel("Elapsed Time (sec)")
        ax_hv.set_ylabel("HV")
        ax_hv.grid(True, alpha=0.3)
        ax_hv.legend(fontsize=8)
    plt.tight_layout()
    save_path = os.path.join(
        out_dir, f"{instance}_beta{beta:.3f}_convergence_by_time.png"
    )
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    fig, axes = plt.subplots(
        3 if "hv" in gdf.columns else 2, 1, figsize=(8, 12), sharex=False
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    ax_cost = axes[0]
    ax_ra = axes[1]
    ax_hv = axes[2] if "hv" in gdf.columns else None
    for run_id, seed in runs:
        sdf = gdf[(gdf["run_id"] == run_id) & (gdf["seed"] == seed)].sort_values(
            "eval_count"
        )
        x_eval = sdf["eval_count"].values
        y_cost = smooth_series(sdf["best_cost_in_archive"].values, window=smooth_window)
        y_ra = smooth_series(sdf["best_ra_in_archive"].values, window=smooth_window)
        ax_cost.plot(
            x_eval, y_cost, marker="o", linewidth=1.5, label=f"run{run_id}-seed{seed}"
        )
        ax_ra.plot(
            x_eval, y_ra, marker="o", linewidth=1.5, label=f"run{run_id}-seed{seed}"
        )
        if ax_hv is not None:
            y_hv = smooth_series(sdf["hv"].values, window=smooth_window)
            ax_hv.plot(
                x_eval, y_hv, marker="o", linewidth=1.5, label=f"run{run_id}-seed{seed}"
            )
    ax_cost.set_title(f"{instance} | beta={beta} | Best Cost vs Eval Count")
    ax_cost.set_xlabel("Evaluation Count")
    ax_cost.set_ylabel("Best Cost in Archive")
    ax_cost.grid(True, alpha=0.3)
    ax_cost.legend(fontsize=8)
    ax_ra.set_title(f"{instance} | beta={beta} | Best RA vs Eval Count")
    ax_ra.set_xlabel("Evaluation Count")
    ax_ra.set_ylabel("Best RA in Archive")
    ax_ra.grid(True, alpha=0.3)
    ax_ra.legend(fontsize=8)
    if ax_hv is not None:
        ax_hv.set_title(f"{instance} | beta={beta} | HV vs Eval Count")
        ax_hv.set_xlabel("Evaluation Count")
        ax_hv.set_ylabel("HV")
        ax_hv.grid(True, alpha=0.3)
        ax_hv.legend(fontsize=8)
    plt.tight_layout()
    save_path = os.path.join(
        out_dir, f"{instance}_beta{beta:.3f}_convergence_by_eval.png"
    )
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def build_final_iter_table(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the last iteration for each instance, beta, run_id, and seed to compare final results."""
    idx = df.groupby(["instance", "beta", "run_id", "seed"])["iter"].idxmax()
    final_df = (
        df.loc[idx]
        .sort_values(["instance", "beta", "run_id", "seed"])
        .reset_index(drop=True)
    )
    return final_df


def summarize_final_table(final_df: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "elapsed_sec": ["mean", "std", "max"],
        "eval_count": ["mean", "std", "max"],
        "best_cost_in_archive": ["mean", "std", "min"],
        "best_ra_in_archive": ["mean", "std", "min"],
    }
    if "hv" in final_df.columns:
        agg["hv"] = ["mean", "std", "max"]
    summary = final_df.groupby(["instance", "beta"]).agg(agg)
    summary.columns = ["_".join(col) for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    return summary


def align_run_to_time_grid(
    run_df: pd.DataFrame, time_grid: np.ndarray, metrics: list[str]
) -> pd.DataFrame:
    """Align a run to a common time grid by carrying forward its latest observation."""
    run_df = run_df.sort_values("elapsed_sec").copy()
    base = pd.DataFrame({"elapsed_sec": time_grid})
    merged = pd.merge_asof(
        base, run_df[["elapsed_sec"] + metrics], on="elapsed_sec", direction="backward"
    )
    for col in metrics:
        if merged[col].isna().all():
            continue
        merged[col] = merged[col].bfill()
    return merged


def build_time_aligned_summary(
    df: pd.DataFrame,
    instance: str,
    beta: float,
    time_step: int = 200,
    max_time: int | None = None,
):
    """Align runs for an instance and beta to a common time grid and compute means and standard deviations."""
    gdf = df[(df["instance"] == instance) & (df["beta"] == beta)].copy()
    if gdf.empty:
        return (None, None)
    if max_time is None:
        max_time = int(np.ceil(gdf["elapsed_sec"].max() / time_step) * time_step)
    time_grid = np.arange(0, max_time + time_step, time_step)
    metrics = ["best_cost_in_archive", "best_ra_in_archive"]
    if "hv" in gdf.columns:
        metrics.append("hv")
    aligned_runs = []
    run_keys = sorted(
        gdf[["run_id", "seed"]].drop_duplicates().itertuples(index=False, name=None)
    )
    for run_id, seed in run_keys:
        rdf = gdf[(gdf["run_id"] == run_id) & (gdf["seed"] == seed)].copy()
        aligned = align_run_to_time_grid(rdf, time_grid, metrics)
        aligned["run_id"] = run_id
        aligned["seed"] = seed
        aligned_runs.append(aligned)
    aligned_df = pd.concat(aligned_runs, ignore_index=True)
    agg_dict = {}
    for m in metrics:
        agg_dict[m] = ["mean", "std", "median", "min", "max"]
    summary = aligned_df.groupby("elapsed_sec").agg(agg_dict)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    return (aligned_df, summary)


def plot_time_aligned_mean_curve(
    summary: pd.DataFrame, out_path: str, title_prefix: str
):
    has_hv = "hv_mean" in summary.columns
    nrows = 3 if has_hv else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(8, 11))
    if nrows == 1:
        axes = [axes]
    x = summary["elapsed_sec"].values
    ax = axes[0]
    y = summary["best_cost_in_archive_mean"].values
    s = summary["best_cost_in_archive_std"].fillna(0).values
    ax.plot(x, y, linewidth=2, label="Mean")
    ax.fill_between(x, y - s, y + s, alpha=0.25, label="Mean ± Std")
    ax.set_title(f"{title_prefix} | Best Cost vs Time")
    ax.set_xlabel("Elapsed Time (sec)")
    ax.set_ylabel("Best Cost")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax = axes[1]
    y = summary["best_ra_in_archive_mean"].values
    s = summary["best_ra_in_archive_std"].fillna(0).values
    ax.plot(x, y, linewidth=2, label="Mean")
    ax.fill_between(x, y - s, y + s, alpha=0.25, label="Mean ± Std")
    ax.set_title(f"{title_prefix} | Best RA vs Time")
    ax.set_xlabel("Elapsed Time (sec)")
    ax.set_ylabel("Best RA")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if has_hv:
        ax = axes[2]
        y = summary["hv_mean"].values
        s = summary["hv_std"].fillna(0).values
        ax.plot(x, y, linewidth=2, label="Mean")
        ax.fill_between(x, y - s, y + s, alpha=0.25, label="Mean ± Std")
        ax.set_title(f"{title_prefix} | HV vs Time")
        ax.set_xlabel("Elapsed Time (sec)")
        ax.set_ylabel("HV")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

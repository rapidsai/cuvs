#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Plot CAGRA filter benchmark CSV (bitset vs bloom_filter line charts)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FILTER_LABELS = {
    "bitset": "Bitset",
    "bloom_filter": "Bloom",
}

FILTER_ORDER = ["Bitset", "Bloom"]
FILTER_COLORS = {"Bitset": "#0173b2", "Bloom": "#de8f05"}
FILTER_MARKERS = {"Bitset": "o", "Bloom": "s"}

SEARCH_LABELS = {
    10000: "10k queries",
    25000: "25k queries",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate line charts from cagra_filter_benchmark CSV output.",
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default="filter_bench.csv",
        help="Path to benchmark CSV (default: filter_bench.csv)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="filter_bench_plots",
        help="Directory for PNG output (default: filter_bench_plots)",
    )
    parser.add_argument(
        "--with-recall",
        action="store_true",
        help="Also plot recall charts when non-NaN values are present",
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["recall"] = pd.to_numeric(df["recall"], errors="coerce")
    df["filter_label"] = (
        df["filter_type"].map(FILTER_LABELS).fillna(df["filter_type"])
    )
    df["search_label"] = (
        df["search_n_rows"]
        .map(SEARCH_LABELS)
        .fillna(df["search_n_rows"].astype(str) + " queries")
    )
    return df


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.subplots_adjust(left=0.11, bottom=0.08, right=0.98, top=0.94)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_panel_bars(
    ax: plt.Axes,
    panel: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> list:
    """Grouped bars for two filters at each x value — visible even when latencies are close."""
    x_values = sorted(panel[x_col].unique())
    x_idx = np.arange(len(x_values))
    bar_width = 0.36
    handles = []

    for series_idx, label in enumerate(FILTER_ORDER):
        heights = []
        for x_val in x_values:
            row = panel[
                (panel["filter_label"] == label) & (panel[x_col] == x_val)
            ]
            heights.append(row[y_col].iloc[0] if not row.empty else 0.0)
        offset = (series_idx - 0.5) * bar_width
        bars = ax.bar(
            x_idx + offset,
            heights,
            width=bar_width,
            label=label,
            color=FILTER_COLORS[label],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )
        handles.append(bars[0])

    ax.set_xticks(x_idx)
    if x_col == "build_n_rows":
        ax.set_xticklabels([f"{int(v):,}" for v in x_values])
    else:
        ax.set_xticklabels(
            [
                str(int(v)) if float(v).is_integer() else str(v)
                for v in x_values
            ]
        )
    return handles


def plot_panel_lines(
    ax: plt.Axes,
    panel: pd.DataFrame,
    x_col: str,
    y_col: str,
    log_x: bool = False,
) -> list:
    """Draw bitset vs bloom as two explicit series (no seaborn hue/style mashup)."""
    handles = []
    x_values = sorted(panel[x_col].unique())
    if log_x:
        ax.set_xscale("log")
        ax.set_xticks(x_values)
        ax.set_xticklabels([f"{int(v):,}" for v in x_values])
        ax.minorticks_off()

    for label in FILTER_ORDER:
        series = panel[panel["filter_label"] == label].sort_values(x_col)
        if series.empty:
            continue
        (line,) = ax.plot(
            series[x_col],
            series[y_col],
            label=label,
            color=FILTER_COLORS[label],
            marker=FILTER_MARKERS[label],
            markersize=7,
            linewidth=2.0,
            markerfacecolor="white",
            markeredgewidth=1.5,
        )
        handles.append(line)

    return handles


def add_row_band_label(
    fig: plt.Figure, axes: np.ndarray, row_idx: int, text: str
) -> None:
    ax = axes[row_idx, 0]
    pos = ax.get_position()
    fig.text(
        0.02,
        (pos.y0 + pos.y1) / 2,
        text,
        ha="left",
        va="center",
        rotation=90,
        fontsize=10,
        fontweight="bold",
    )


def plot_latency_grid_for_valid_pct(
    df: pd.DataFrame,
    out_dir: Path,
    valid_pct: int,
) -> None:
    """One PNG per valid_pct: 6×3 grid.

    Row bands (top to bottom): 10k queries × dims 128/512/1024, then 25k × dims.
    Columns: k = 64 / 256 / 1024.
    Each panel has only two lines (bitset vs bloom) over index size.
    """
    sub = df[df["valid_pct"] == valid_pct].copy()
    if sub.empty:
        print(f"valid_pct={valid_pct}%: no rows, skipping")
        return

    k_values = sorted(sub["k"].unique())
    col_values = sorted(sub["build_n_cols"].unique())
    search_values = sorted(sub["search_n_rows"].unique())

    row_specs = [
        (search, dims) for search in search_values for dims in col_values
    ]
    n_rows = len(row_specs)
    n_cols = len(k_values)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols + 0.8, 2.4 * n_rows),
        squeeze=False,
    )

    legend_handles = None
    for row_idx, (search_n_rows, build_n_cols) in enumerate(row_specs):
        for col_idx, k in enumerate(k_values):
            ax = axes[row_idx, col_idx]
            panel = sub[
                (sub["build_n_cols"] == build_n_cols)
                & (sub["k"] == k)
                & (sub["search_n_rows"] == search_n_rows)
            ]
            if panel.empty:
                ax.set_visible(False)
                continue

            handles = plot_panel_bars(
                ax,
                panel,
                x_col="build_n_rows",
                y_col="avg_latency_per_query_ms",
            )
            if legend_handles is None and handles:
                legend_handles = handles

            ax.set_title(f"k={k}", fontsize=10)
            ax.set_xlabel("index rows")
            if col_idx == 0:
                ax.set_ylabel(f"dims={build_n_cols}\nlatency / query (ms)")
            else:
                ax.set_ylabel("")

    for search_idx, search_n_rows in enumerate(search_values):
        band_row = search_idx * len(col_values)
        add_row_band_label(
            fig,
            axes,
            band_row,
            SEARCH_LABELS.get(search_n_rows, f"{search_n_rows:,} queries"),
        )

    if legend_handles:
        fig.legend(
            legend_handles,
            FILTER_ORDER,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            frameon=False,
            fontsize=11,
        )

    fig.suptitle(
        f"Per-query search latency  —  {valid_pct}% rows valid",
        fontsize=13,
        y=1.01,
    )
    path = out_dir / f"latency_valid_{valid_pct}pct.png"
    save_figure(fig, path)
    print(f"wrote {path}")


def plot_all_valid_pct_grids(df: pd.DataFrame, out_dir: Path) -> None:
    for valid_pct in sorted(df["valid_pct"].unique()):
        plot_latency_grid_for_valid_pct(df, out_dir, int(valid_pct))


def plot_valid_pct_overview(
    df: pd.DataFrame,
    out_dir: Path,
    build_n_rows: int,
    search_n_rows: int,
) -> None:
    """valid_pct on x-axis at one build/search point; 3×3 grid (dims × k)."""
    sub = df[
        (df["build_n_rows"] == build_n_rows)
        & (df["search_n_rows"] == search_n_rows)
    ].copy()
    if sub.empty:
        print(
            "valid_pct overview: no rows for selected build/search slice, skipping"
        )
        return

    k_values = sorted(sub["k"].unique())
    col_values = sorted(sub["build_n_cols"].unique())

    fig, axes = plt.subplots(
        len(col_values),
        len(k_values),
        figsize=(4.0 * len(k_values), 2.8 * len(col_values)),
        squeeze=False,
    )

    legend_handles = None
    for row_idx, build_n_cols in enumerate(col_values):
        for col_idx, k in enumerate(k_values):
            ax = axes[row_idx, col_idx]
            panel = sub[
                (sub["build_n_cols"] == build_n_cols) & (sub["k"] == k)
            ]
            if panel.empty:
                ax.set_visible(False)
                continue

            handles = plot_panel_bars(
                ax,
                panel,
                x_col="valid_pct",
                y_col="avg_latency_per_query_ms",
            )
            if legend_handles is None and handles:
                legend_handles = handles

            ax.set_title(f"k={k}", fontsize=10)
            ax.set_xlabel("valid rows (%)")
            if col_idx == 0:
                ax.set_ylabel(f"dims={build_n_cols}\nlatency / query (ms)")
            else:
                ax.set_ylabel("")

    if legend_handles:
        fig.legend(
            legend_handles,
            FILTER_ORDER,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            frameon=False,
            fontsize=11,
        )

    fig.suptitle(
        f"Latency vs filter selectivity  —  "
        f"build={build_n_rows:,} rows  search={search_n_rows:,} queries",
        fontsize=12,
        y=1.01,
    )
    path = out_dir / "overview_valid_pct_sweep.png"
    save_figure(fig, path)
    print(f"wrote {path}")


def plot_recall_grid_for_valid_pct(
    df: pd.DataFrame,
    out_dir: Path,
    valid_pct: int,
) -> None:
    sub = df[(df["valid_pct"] == valid_pct) & df["recall"].notna()].copy()
    if sub.empty:
        return

    k_values = sorted(sub["k"].unique())
    col_values = sorted(sub["build_n_cols"].unique())
    search_values = sorted(sub["search_n_rows"].unique())
    row_specs = [
        (search, dims) for search in search_values for dims in col_values
    ]

    fig, axes = plt.subplots(
        len(row_specs),
        len(k_values),
        figsize=(4.0 * len(k_values) + 0.8, 2.4 * len(row_specs)),
        squeeze=False,
    )

    legend_handles = None
    for row_idx, (search_n_rows, build_n_cols) in enumerate(row_specs):
        for col_idx, k in enumerate(k_values):
            ax = axes[row_idx, col_idx]
            panel = sub[
                (sub["build_n_cols"] == build_n_cols)
                & (sub["k"] == k)
                & (sub["search_n_rows"] == search_n_rows)
            ]
            if panel.empty:
                ax.set_visible(False)
                continue

            handles = plot_panel_bars(
                ax,
                panel,
                x_col="build_n_rows",
                y_col="recall",
            )
            if legend_handles is None and handles:
                legend_handles = handles

            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"k={k}", fontsize=10)
            ax.set_xlabel("index rows")
            if col_idx == 0:
                ax.set_ylabel(f"dims={build_n_cols}\nrecall@k")
            else:
                ax.set_ylabel("")

    for search_idx, search_n_rows in enumerate(search_values):
        band_row = search_idx * len(col_values)
        add_row_band_label(
            fig,
            axes,
            band_row,
            SEARCH_LABELS.get(search_n_rows, f"{search_n_rows:,} queries"),
        )

    if legend_handles:
        fig.legend(
            legend_handles,
            FILTER_ORDER,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            frameon=False,
            fontsize=11,
        )

    fig.suptitle(f"Recall@k  —  {valid_pct}% rows valid", fontsize=13, y=1.01)
    path = out_dir / f"recall_valid_{valid_pct}pct.png"
    save_figure(fig, path)
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = load_csv(csv_path)
    print(f"loaded {len(df)} rows from {csv_path}")

    sns.set_theme(style="whitegrid", context="notebook")

    plot_all_valid_pct_grids(df, out_dir)

    default_build = int(df["build_n_rows"].min())
    default_search = int(df["search_n_rows"].min())
    plot_valid_pct_overview(df, out_dir, default_build, default_search)

    if args.with_recall and df["recall"].notna().any():
        for valid_pct in sorted(df["valid_pct"].unique()):
            plot_recall_grid_for_valid_pct(df, out_dir, int(valid_pct))
    elif args.with_recall:
        print(
            "recall charts skipped: CSV has no recall values (run benchmark with --ground-truth)"
        )
    else:
        print("recall charts skipped (default; pass --with-recall to enable)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot UDF benchmark results from CSV output.

Usage:
    python plot_udf_benchmark.py udf_benchmark_results.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_benchmark_results(csv_file: str):
    # Read data
    df = pd.read_csv(csv_file)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "IVF-Flat UDF Benchmark: Built-in vs Macro UDF vs Raw UDF\n(1M vectors, 512 dims, 100 queries)",
        fontsize=14,
        fontweight="bold",
    )

    colors = {"float32": "#2ecc71", "int8": "#3498db"}

    # =========================================================================
    # Plot 1: Median search time comparison (float32)
    # =========================================================================
    ax1 = axes[0, 0]

    data_f32 = df[df["dtype"] == "float32"]
    x = np.arange(len(data_f32))
    width = 0.25

    ax1.bar(
        x - width,
        data_f32["median_builtin_ms"],
        width,
        label="Built-in",
        color=colors["float32"],
        alpha=0.9,
    )
    ax1.bar(
        x,
        data_f32["median_udf_ms"],
        width,
        label="Macro UDF",
        color=colors["float32"],
        alpha=0.5,
        hatch="//",
    )
    ax1.bar(
        x + width,
        data_f32["median_raw_ms"],
        width,
        label="Raw UDF",
        color=colors["float32"],
        alpha=0.3,
        hatch="\\\\",
    )

    ax1.set_xlabel("k (neighbors)")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Float32: Median Search Time")
    ax1.set_xticks(x)
    ax1.set_xticklabels(data_f32["k"])
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", alpha=0.3)

    # =========================================================================
    # Plot 2: Median search time comparison (int8)
    # =========================================================================
    ax2 = axes[0, 1]

    data_int8 = df[df["dtype"] == "int8"]
    x = np.arange(len(data_int8))

    ax2.bar(
        x - width,
        data_int8["median_builtin_ms"],
        width,
        label="Built-in",
        color=colors["int8"],
        alpha=0.9,
    )
    ax2.bar(
        x,
        data_int8["median_udf_ms"],
        width,
        label="Macro UDF",
        color=colors["int8"],
        alpha=0.5,
        hatch="//",
    )
    ax2.bar(
        x + width,
        data_int8["median_raw_ms"],
        width,
        label="Raw UDF",
        color=colors["int8"],
        alpha=0.3,
        hatch="\\\\",
    )

    ax2.set_xlabel("k (neighbors)")
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("Int8: Median Search Time")
    ax2.set_xticks(x)
    ax2.set_xticklabels(data_int8["k"])
    ax2.legend(loc="upper left")
    ax2.grid(axis="y", alpha=0.3)

    # =========================================================================
    # Plot 3: UDF/Built-in ratio comparison
    # =========================================================================
    ax3 = axes[1, 0]

    for dtype in ["float32", "int8"]:
        data = df[df["dtype"] == dtype]
        ax3.plot(
            data["k"],
            data["udf_ratio"],
            "o-",
            label=f"{dtype} Macro UDF",
            color=colors[dtype],
            linewidth=2,
            markersize=8,
        )
        ax3.plot(
            data["k"],
            data["raw_ratio"],
            "s--",
            label=f"{dtype} Raw UDF",
            color=colors[dtype],
            linewidth=2,
            markersize=8,
            alpha=0.7,
        )

    ax3.axhline(
        y=1.0,
        color="red",
        linestyle="--",
        alpha=0.5,
        label="1.0x (no overhead)",
    )
    ax3.set_xlabel("k (neighbors)")
    ax3.set_ylabel("UDF / Built-in Ratio")
    ax3.set_title("Performance Ratio (closer to 1.0 = better)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(alpha=0.3)
    ax3.set_xscale("log", base=2)

    # =========================================================================
    # Plot 4: Summary bar chart
    # =========================================================================
    ax4 = axes[1, 1]

    # Average ratios
    categories = ["Float32\nMacro", "Float32\nRaw", "Int8\nMacro", "Int8\nRaw"]
    ratios = [
        df[df["dtype"] == "float32"]["udf_ratio"].mean(),
        df[df["dtype"] == "float32"]["raw_ratio"].mean(),
        df[df["dtype"] == "int8"]["udf_ratio"].mean(),
        df[df["dtype"] == "int8"]["raw_ratio"].mean(),
    ]
    bar_colors = [
        colors["float32"],
        colors["float32"],
        colors["int8"],
        colors["int8"],
    ]
    alphas = [0.7, 0.4, 0.7, 0.4]

    bars = ax4.bar(categories, ratios, color=bar_colors, alpha=0.7)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    ax4.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax4.set_ylabel("Average UDF / Built-in Ratio")
    ax4.set_title("Average Overhead Summary")
    ax4.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, ratio in zip(bars, ratios):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{ratio:.2f}x",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()

    # Save figure
    output_file = csv_file.replace(".csv", ".png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")


def print_summary(csv_file: str):
    """Print a summary table of results."""
    df = pd.read_csv(csv_file)

    print("\n" + "=" * 100)
    print("UDF Benchmark Summary")
    print("=" * 100)
    print(
        f"\n{'dtype':<10} {'k':<6} {'Built-in (ms)':<15} {'Macro UDF (ms)':<15} {'Raw UDF (ms)':<15} {'Macro Ratio':<12} {'Raw Ratio':<12}"
    )
    print("-" * 100)

    for _, row in df.iterrows():
        print(
            f"{row['dtype']:<10} {row['k']:<6} {row['median_builtin_ms']:<15.2f} {row['median_udf_ms']:<15.2f} {row['median_raw_ms']:<15.2f} {row['udf_ratio']:<12.3f} {row['raw_ratio']:<12.3f}"
        )

    print("\n" + "=" * 100)
    print("Key Observations:")
    print(
        f"  - Float32 Macro UDF avg ratio: {df[df['dtype'] == 'float32']['udf_ratio'].mean():.3f}x"
    )
    print(
        f"  - Float32 Raw UDF avg ratio:   {df[df['dtype'] == 'float32']['raw_ratio'].mean():.3f}x"
    )
    print(
        f"  - Int8 Macro UDF avg ratio:    {df[df['dtype'] == 'int8']['udf_ratio'].mean():.3f}x"
    )
    print(
        f"  - Int8 Raw UDF avg ratio:      {df[df['dtype'] == 'int8']['raw_ratio'].mean():.3f}x"
    )
    print("=" * 100 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_udf_benchmark.py <csv_file>")
        print(
            "Example: python plot_udf_benchmark.py udf_benchmark_results.csv"
        )
        sys.exit(1)

    csv_file = sys.argv[1]
    print_summary(csv_file)
    plot_benchmark_results(csv_file)

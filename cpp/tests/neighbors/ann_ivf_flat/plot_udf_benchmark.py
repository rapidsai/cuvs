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
        "IVF-Flat UDF Benchmark: Built-in L2 vs Custom UDF L2\n(1M vectors, 512 dims, 100 queries)",
        fontsize=14,
        fontweight="bold",
    )

    colors = {"float32": "#2ecc71", "int8": "#3498db"}

    # =========================================================================
    # Plot 1: First search time (JIT compilation cost)
    # =========================================================================
    ax1 = axes[0, 0]

    for dtype in ["float32", "int8"]:
        data = df[df["dtype"] == dtype]
        x = np.arange(len(data))
        width = 0.35
        offset = -width / 2 if dtype == "float32" else width / 2

        ax1.bar(
            x + offset,
            data["first_builtin_ms"],
            width,
            label=f"{dtype} Built-in",
            color=colors[dtype],
            alpha=0.7,
        )
        ax1.bar(
            x + offset,
            data["first_udf_ms"] - data["first_builtin_ms"],
            width,
            bottom=data["first_builtin_ms"],
            label=f"{dtype} UDF overhead",
            color=colors[dtype],
            alpha=0.4,
            hatch="//",
        )

    ax1.set_xlabel("k (neighbors)")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("First Search Time (includes JIT compilation)")
    ax1.set_xticks(np.arange(len(df[df["dtype"] == "float32"])))
    ax1.set_xticklabels(df[df["dtype"] == "float32"]["k"])
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", alpha=0.3)

    # =========================================================================
    # Plot 2: JIT overhead
    # =========================================================================
    ax2 = axes[0, 1]

    for dtype in ["float32", "int8"]:
        data = df[df["dtype"] == dtype]
        ax2.plot(
            data["k"],
            data["jit_overhead_ms"],
            "o-",
            label=dtype,
            color=colors[dtype],
            linewidth=2,
            markersize=8,
        )

    ax2.set_xlabel("k (neighbors)")
    ax2.set_ylabel("JIT Overhead (ms)")
    ax2.set_title("UDF JIT Compilation Overhead\n(First UDF - First Built-in)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xscale("log", base=2)

    # =========================================================================
    # Plot 3: Median search time (cached)
    # =========================================================================
    ax3 = axes[1, 0]

    width = 0.35
    for i, dtype in enumerate(["float32", "int8"]):
        data = df[df["dtype"] == dtype]
        x = np.arange(len(data))
        offset = (i - 0.5) * width

        ax3.bar(
            x + offset - width / 4,
            data["median_builtin_ms"],
            width / 2,
            label=f"{dtype} Built-in",
            color=colors[dtype],
            alpha=0.8,
        )
        ax3.bar(
            x + offset + width / 4,
            data["median_udf_ms"],
            width / 2,
            label=f"{dtype} UDF",
            color=colors[dtype],
            alpha=0.4,
            hatch="//",
        )

    ax3.set_xlabel("k (neighbors)")
    ax3.set_ylabel("Time (ms)")
    ax3.set_title("Median Search Time (JIT cached, 20 iterations)")
    ax3.set_xticks(np.arange(len(df[df["dtype"] == "float32"])))
    ax3.set_xticklabels(df[df["dtype"] == "float32"]["k"])
    ax3.legend(loc="upper left")
    ax3.grid(axis="y", alpha=0.3)

    # =========================================================================
    # Plot 4: UDF/Built-in ratio
    # =========================================================================
    ax4 = axes[1, 1]

    for dtype in ["float32", "int8"]:
        data = df[df["dtype"] == dtype]
        ax4.plot(
            data["k"],
            data["udf_builtin_ratio"],
            "o-",
            label=dtype,
            color=colors[dtype],
            linewidth=2,
            markersize=8,
        )

    ax4.axhline(
        y=1.0,
        color="red",
        linestyle="--",
        alpha=0.5,
        label="1.0x (no overhead)",
    )
    ax4.set_xlabel("k (neighbors)")
    ax4.set_ylabel("UDF / Built-in Ratio")
    ax4.set_title("UDF Performance Ratio\n(closer to 1.0 = better)")
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_xscale("log", base=2)
    ax4.set_ylim(0.9, max(df["udf_builtin_ratio"].max() * 1.1, 1.2))

    plt.tight_layout()

    # Save figure
    output_file = csv_file.replace(".csv", ".png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")

    # Also show
    plt.show()


def print_summary(csv_file: str):
    """Print a summary table of results."""
    df = pd.read_csv(csv_file)

    print("\n" + "=" * 80)
    print("UDF Benchmark Summary")
    print("=" * 80)
    print(
        f"\n{'dtype':<10} {'k':<6} {'First Builtin':<15} {'First UDF':<15} {'JIT Overhead':<15} {'Median Builtin':<15} {'Median UDF':<15} {'Ratio':<10}"
    )
    print("-" * 100)

    for _, row in df.iterrows():
        print(
            f"{row['dtype']:<10} {row['k']:<6} {row['first_builtin_ms']:<15.2f} {row['first_udf_ms']:<15.2f} {row['jit_overhead_ms']:<15.2f} {row['median_builtin_ms']:<15.2f} {row['median_udf_ms']:<15.2f} {row['udf_builtin_ratio']:<10.3f}"
        )

    print("\n" + "=" * 80)
    print("Key Observations:")
    print(f"  - Average JIT overhead: {df['jit_overhead_ms'].mean():.2f} ms")
    print(
        f"  - Average UDF/Built-in ratio: {df['udf_builtin_ratio'].mean():.3f}x"
    )
    print(f"  - Max UDF/Built-in ratio: {df['udf_builtin_ratio'].max():.3f}x")
    print("=" * 80 + "\n")


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

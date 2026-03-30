#!/usr/bin/env python3
"""Plot memory consumption over time from an allocation_report CSV."""

import argparse
import csv
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if any(field.strip() for field in row)]

    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    t0 = df["timestamp_us"].iloc[0]
    df["time_s"] = (df["timestamp_us"] - t0) / 1e6
    return df


def _fmt_bytes(val, _pos):
    if abs(val) >= 1e9:
        return f"{val / 1e9:.1f} GB"
    if abs(val) >= 1e6:
        return f"{val / 1e6:.0f} MB"
    if abs(val) >= 1e3:
        return f"{val / 1e3:.0f} KB"
    return f"{int(val)} B"


_COLLATE_RE_BRACKETS = re.compile(r"\[\d+\]")
_COLLATE_RE_PARENS = re.compile(r"\([^()]*\d[^()]*\)$")


def _collate_name(name):
    """Strip varying numeric arguments so similar ranges group together.

    ``foo[123]`` → ``foo[*]``; ``bar(5000, 16)`` → ``bar(*)``.
    Template args in angle brackets (e.g. ``<IVF-PQ>``) are preserved.
    """
    name = _COLLATE_RE_BRACKETS.sub("[*]", name)
    name = _COLLATE_RE_PARENS.sub("(*)", name)
    return name


def _collect_spans(df):
    """Return ``[(name, depth, t_start, t_end), ...]`` for contiguous NVTX ranges.

    Uses *nvtx_depth* (if present) to reconstruct nested ranges via a
    stack-based algorithm.  Names are kept verbatim; call :func:`_collate_spans`
    afterwards to merge small ranges that differ only in numeric arguments.
    """
    has_depth = "nvtx_depth" in df.columns
    active = {}  # depth -> (name, t_start)
    spans = []

    for _, row in df.iterrows():
        d = (
            int(row["nvtx_depth"])
            if has_depth
            else (1 if pd.notna(row.get("nvtx_range")) and row["nvtx_range"] else 0)
        )
        raw = (
            str(row["nvtx_range"])
            if pd.notna(row.get("nvtx_range")) and row["nvtx_range"]
            else ""
        )
        t = row["time_s"]

        # Close all depths strictly deeper than current.
        for dd in [k for k in active if k > d]:
            aname, at = active.pop(dd)
            if aname:
                spans.append((aname, dd, at, t))

        if d == 0:
            for dd in list(active):
                aname, at = active.pop(dd)
                if aname:
                    spans.append((aname, dd, at, t))
            continue

        # Update the innermost active depth.
        if d in active:
            aname, at = active[d]
            if aname != raw:
                if aname:
                    spans.append((aname, d, at, t))
                active[d] = (raw, t)
        else:
            active[d] = (raw, t)

    t_end = df["time_s"].iloc[-1] if not df.empty else 0
    for d, (aname, at) in active.items():
        if aname:
            spans.append((aname, d, at, t_end))

    return spans


def _collate_spans(spans, t_range, threshold=0.1):
    """Merge adjacent small spans at the same depth that share a collated name.

    Spans wider than *threshold* × *t_range* keep their original (uncollated)
    name, so that significant phases remain individually labeled.
    """
    from collections import defaultdict

    by_depth = defaultdict(list)
    for name, depth, t0, t1 in spans:
        by_depth[depth].append((name, t0, t1))

    result = []
    for depth in sorted(by_depth):
        entries = sorted(by_depth[depth], key=lambda x: x[1])
        merged = []
        for name, t0, t1 in entries:
            is_big = t_range > 0 and (t1 - t0) / t_range >= threshold
            label = name if is_big else _collate_name(name)

            if merged and merged[-1][0] == label:
                merged[-1] = (label, merged[-1][1], t1)
            else:
                merged.append((label, t0, t1))

        for name, t0, t1 in merged:
            result.append((name, depth, t0, t1))

    return result


def _pick_color(name, colors):
    palette = plt.cm.tab20.colors
    return colors.setdefault(name, palette[len(colors) % len(palette)])


def _shade_nvtx(ax, spans, colors, t_range):
    """Shade data axes, auto-selecting the shallowest non-trivial depth.

    A depth level is "trivial" when it contains a single range name that
    covers nearly the entire timeline (>95%).  In that case we step down to
    the next depth until we find one that provides useful visual contrast.
    """
    if not spans or t_range <= 0:
        return

    depths = sorted(set(d for _, d, _, _ in spans))
    for shade_depth in depths:
        depth_spans = [(n, d, t0, t1) for n, d, t0, t1 in spans if d == shade_depth]
        unique_names = set(n for n, _, _, _ in depth_spans)
        if len(unique_names) == 1:
            coverage = sum(t1 - t0 for _, _, t0, t1 in depth_spans) / t_range
            if coverage > 0.95:
                continue
        for name, _, t0, t1 in depth_spans:
            c = _pick_color(name, colors)
            ax.axvspan(t0, t1, alpha=0.08, color=c)
            ax.axvline(t0, color=c, linewidth=0.5, linestyle="--", alpha=0.35)
        return


def _draw_nvtx_strip(ax, spans, t_min, t_max, colors, max_depth=None):
    """Draw labeled NVTX ranges on a multi-lane strip (one lane per depth)."""
    if not spans:
        return

    if max_depth is not None:
        spans = [(n, d, t0, t1) for n, d, t0, t1 in spans if d <= max_depth]

    depths = sorted(set(d for _, d, _, _ in spans))
    if not depths:
        return

    n_lanes = len(depths)
    depth_to_lane = {d: i for i, d in enumerate(depths)}

    for name, depth, t0, t1 in spans:
        lane = depth_to_lane[depth]
        y_bottom = n_lanes - lane - 1

        c = _pick_color(name, colors)
        ax.broken_barh(
            [(t0, t1 - t0)],
            (y_bottom, 0.9),
            facecolor=(*c[:3], 0.3),
            edgecolor=(*c[:3], 0.6),
            linewidth=0.5,
        )

        t_range = t_max - t_min
        width_frac = (t1 - t0) / t_range if t_range > 0 else 1
        if width_frac > 0.02:
            fs = 6.5 if width_frac > 0.1 else 5.5
            rot = 0 if width_frac > 0.15 else 90
            ax.text(
                (t0 + t1) / 2,
                y_bottom + 0.45,
                name,
                ha="center",
                va="center",
                fontsize=fs,
                rotation=rot,
                clip_on=True,
            )

    ax.set_yticks([n_lanes - i - 0.55 for i in range(n_lanes)])
    ax.set_yticklabels([str(d) for d in depths], fontsize=6)
    ax.set_ylim(0, n_lanes)
    ax.set_ylabel("NVTX\ndepth", fontsize=7, rotation=0, labelpad=22, va="center")
    ax.set_xlim(t_min, t_max)


def _has(df, prefix):
    return f"{prefix}_current" in df.columns


def plot(
    df: pd.DataFrame,
    output: str | None,
    t_start: float | None = None,
    t_end: float | None = None,
    max_depth: int | None = None,
):
    if t_start is not None:
        df = df[df["time_s"] >= t_start].copy()
    if t_end is not None:
        df = df[df["time_s"] <= t_end].copy()
    if df.empty:
        print("No data in the selected time range", file=sys.stderr)
        return
    t_min = df["time_s"].iloc[0]
    t_max = df["time_s"].iloc[-1]
    t_range = t_max - t_min
    spans = _collate_spans(_collect_spans(df), t_range)

    visible_depths = sorted(set(d for _, d, _, _ in spans))
    if max_depth is not None:
        visible_depths = [d for d in visible_depths if d <= max_depth]
    n_lanes = max(len(visible_depths), 1)
    strip_height = n_lanes * 0.4

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [4, strip_height, 4]},
    )
    ax_dev, ax_nvtx, ax_host = axes

    colors = {}
    t = df["time_s"]

    # ── Device memory ────────────────────────────────────────────────────
    dev_stack = []
    dev_labels = []
    if _has(df, "workspace"):
        dev_stack.append(df["workspace_current"])
        dev_labels.append("workspace")
    if _has(df, "large_workspace"):
        dev_stack.append(df["large_workspace_current"])
        dev_labels.append("large workspace")
    if _has(df, "device"):
        dev_stack.append(df["device_current"])
        dev_labels.append("device (other)")

    if dev_stack:
        ax_dev.stackplot(t, *dev_stack, labels=dev_labels, alpha=0.7)
    ax_dev.set_ylabel("Bytes")
    ax_dev.set_title("Device memory consumption")
    ax_dev.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_bytes))
    ax_dev.legend(loc="upper left", fontsize=7)
    ax_dev.grid(axis="y", linestyle=":", alpha=0.4)
    _shade_nvtx(ax_dev, spans, colors, t_range)

    # ── NVTX range strip ─────────────────────────────────────────────────
    _draw_nvtx_strip(ax_nvtx, spans, t_min, t_max, colors, max_depth=max_depth)

    # ── Host memory ──────────────────────────────────────────────────────
    host_stack = []
    host_labels = []
    if _has(df, "host"):
        host_stack.append(df["host_current"])
        host_labels.append("host")
    if _has(df, "pinned"):
        host_stack.append(df["pinned_current"])
        host_labels.append("pinned")
    if _has(df, "managed"):
        host_stack.append(df["managed_current"])
        host_labels.append("managed")

    if host_stack:
        ax_host.stackplot(t, *host_stack, labels=host_labels, alpha=0.7)
    ax_host.set_ylabel("Bytes")
    ax_host.set_xlabel("Time (s)")
    ax_host.set_title("Host memory consumption (pinned & managed also reside on host)")
    ax_host.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_bytes))
    ax_host.legend(loc="upper left", fontsize=7)
    ax_host.grid(axis="y", linestyle=":", alpha=0.4)
    _shade_nvtx(ax_host, spans, colors, t_range)

    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=150)
        print(f"Saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Path to allocations.csv")
    parser.add_argument("-o", "--output", help="Save figure to file (e.g. plot.png)")
    parser.add_argument(
        "-s",
        "--start",
        type=float,
        default=None,
        help="Start time in seconds",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=float,
        default=None,
        help="End time in seconds",
    )
    parser.add_argument(
        "-t",
        "--duration",
        type=float,
        default=None,
        help="Duration in seconds (from --start, or from the beginning)",
    )
    parser.add_argument(
        "-d",
        "--max-depth",
        type=int,
        default=None,
        help="Maximum NVTX depth level to display (default: all)",
    )
    args = parser.parse_args()

    df = load_csv(args.csv)
    if df.empty:
        print("CSV is empty", file=sys.stderr)
        sys.exit(1)

    t_start = args.start
    if args.end is not None and args.duration is not None:
        parser.error("--end and --duration are mutually exclusive")
    if args.duration is not None:
        t_end = (t_start if t_start is not None else 0) + args.duration
    else:
        t_end = args.end

    plot(df, args.output, t_start, t_end, args.max_depth)


if __name__ == "__main__":
    main()

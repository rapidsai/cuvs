# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize RESULT lines of ci/bench/bench.cu: per configuration and variant, the median and
min-max of the timings, the change of the median vs. base, and the quality metrics."""

import statistics
import sys
from collections import OrderedDict, defaultdict

VARIANTS = ["base", "pr2801", "pr2804"]


def parse(path):
    data = OrderedDict()
    for line in open(path, errors="replace"):
        if not line.startswith("RESULT "):
            continue
        left, _, right = line[len("RESULT ") :].strip().partition(" | ")
        m = dict(kv.split("=", 1) for kv in right.split() if "=" in kv)
        v = m.pop("variant")
        data.setdefault(left, defaultdict(list))[v].append(m)
    return data


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def fmt_t(xs):
    if not xs:
        return "-"
    return f"{med(xs):.4g} ({min(xs):.4g}-{max(xs):.4g})"


def main(path):
    data = parse(path)
    print("| config | metric | " + " | ".join(VARIANTS) + " | pr2801 vs base | pr2804 vs base |")
    print("|---|---|" + "---|" * len(VARIANTS) + "---|---|")
    for cfg, per_v in data.items():
        key = "per_iter_ms" if cfg.startswith("kmeans_fit") else "ms"
        if cfg.startswith("search"):
            key = "qps"
        t = {v: [float(m[key]) for m in per_v.get(v, []) if key in m] for v in VARIANTS}
        base = med(t["base"])
        deltas = []
        for v in ("pr2801", "pr2804"):
            if t[v] and t["base"]:
                deltas.append(f"{100 * (med(t[v]) / base - 1):+.1f}%")
            else:
                deltas.append("-")
        cells = [f"{fmt_t(t[v])} n={len(t[v])}" for v in VARIANTS]
        print(f"| {cfg} | {key} | " + " | ".join(cells) + " | " + " | ".join(deltas) + " |")
        extra = []
        for metric in (
            "ms",
            "iters",
            "inertia",
            "recall",
            "inertia64",
            "max_size",
            "center_maxrel_vs64",
            "center_maxabs_vs64",
            "iters64",
        ):
            if metric == key:
                continue
            vals = {v: [m[metric] for m in per_v.get(v, []) if metric in m] for v in VARIANTS}
            if not any(vals.values()):
                continue
            row = []
            for v in VARIANTS:
                xs = vals[v]
                if not xs:
                    row.append("-")
                    continue
                try:
                    f = [float(x) for x in xs]
                    row.append(
                        f"{med(f):.6g}" if min(f) == max(f) else f"{med(f):.6g} ({min(f):.6g}-{max(f):.6g})"
                    )
                except ValueError:
                    row.append(",".join(sorted(set(xs))))
            extra.append(f"| {cfg} | {metric} | " + " | ".join(row) + " | | |")
        hashes = {v: len({m["hash"] for m in per_v.get(v, []) if "hash" in m}) for v in VARIANTS}
        if any(hashes.values()):
            extra.append(
                f"| {cfg} | distinct results | "
                + " | ".join(f"{hashes[v]} of {len(per_v.get(v, []))}" for v in VARIANTS)
                + " | | |"
            )
        for e in extra:
            print(e)


if __name__ == "__main__":
    main(sys.argv[1])

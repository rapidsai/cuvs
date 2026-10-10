# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# DO NOT MERGE: summarize an nsys sqlite export around the end of the capture
# (the stall) for the H100 hang reproducer.

import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
db.row_factory = sqlite3.Row


def q(sql, *args):
    try:
        return db.execute(sql, args).fetchall()
    except sqlite3.Error as e:
        print(f"  (query failed: {e})")
        return []


def tables():
    return [
        r[0] for r in q("SELECT name FROM sqlite_master WHERE type='table'")
    ]


def cols(t):
    return [r[1] for r in q(f"PRAGMA table_info({t})")]


def s(sid):
    r = q("SELECT value FROM StringIds WHERE id=?", sid)
    return r[0][0] if r else str(sid)


T = tables()
print("== tables with rows")
t_end = 0
for t in T:
    n = q(f"SELECT count(*) FROM {t}")[0][0]
    c = cols(t)
    extra = ""
    if n and "start" in c:
        mn, mx = q(f"SELECT min(start), max(start) FROM {t}")[0]
        extra = f" start {mn / 1e9:.3f}..{mx / 1e9:.3f} s"
        if t.startswith("CUPTI_ACTIVITY_KIND_KERNEL") or t.startswith(
            "CUPTI_ACTIVITY_KIND_RUNTIME"
        ):
            t_end = max(t_end, mx)
    if n:
        print(f"  {t}: {n}{extra}")

print(f"== last CUDA activity at {t_end / 1e9:.3f} s")

if "CUPTI_ACTIVITY_KIND_KERNEL" in T:
    print("== last 25 kernels (start s, dur ms, stream, name)")
    for r in q(
        "SELECT start, end, streamId, demangledName, gridX, gridY, gridZ FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start DESC LIMIT 25"
    )[::-1]:
        print(
            f"  {r['start'] / 1e9:.4f} {(r['end'] - r['start']) / 1e6:9.3f} s{r['streamId']} grid=({r['gridX']},{r['gridY']},{r['gridZ']}) {s(r['demangledName'])[:150]}"
        )
    print("== longest 10 kernels")
    for r in q(
        "SELECT start, end, streamId, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY end-start DESC LIMIT 10"
    ):
        print(
            f"  {r['start'] / 1e9:.4f} {(r['end'] - r['start']) / 1e6:9.3f} ms s{r['streamId']} {s(r['demangledName'])[:150]}"
        )

for t in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
    if t in T:
        print(f"== last 15 {t} (start s, dur ms, bytes, kinds)")
        c = cols(t)
        kinds = [
            k for k in ("copyKind", "srcKind", "dstKind", "memKind") if k in c
        ]
        for r in q(f"SELECT * FROM {t} ORDER BY start DESC LIMIT 15")[::-1]:
            print(
                f"  {r['start'] / 1e9:.4f} {(r['end'] - r['start']) / 1e6:9.3f} {r['bytes']} s{r['streamId']} "
                + " ".join(f"{k}={r[k]}" for k in kinds)
            )

if "CUPTI_ACTIVITY_KIND_RUNTIME" in T:
    print(
        "== last 40 CUDA runtime/driver API calls (start s, dur ms, tid, name)"
    )
    for r in q(
        "SELECT start, end, globalTid, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start DESC LIMIT 40"
    )[::-1]:
        print(
            f"  {r['start'] / 1e9:.4f} {(r['end'] - r['start']) / 1e6:9.3f} {r['globalTid'] % 0x1000000} {s(r['nameId'])}"
        )
    print("== slowest 10 API calls")
    for r in q(
        "SELECT start, end, globalTid, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY end-start DESC LIMIT 10"
    ):
        print(
            f"  {r['start'] / 1e9:.4f} {(r['end'] - r['start']) / 1e6:9.3f} ms {s(r['nameId'])}"
        )

for t in T:
    if not t.startswith("CUDA_UM_") and "UNIFIED_MEMORY" not in t:
        continue
    c = cols(t)
    print(f"== {t} columns: {c}")
    if "start" not in c:
        continue
    cnt = (
        "sum(numberOfPageFaults)" if "numberOfPageFaults" in c else "count(*)"
    )
    print(f"  per 10 s bucket ({cnt}):")
    for r in q(
        f"SELECT CAST(start/10e9 AS INT) AS b, count(*) AS n, {cnt} AS f FROM {t} GROUP BY b ORDER BY b"
    ):
        print(f"    t={r['b'] * 10:5d}s events={r['n']} faults={r['f']}")
    if "address" in c:
        print("  top 15 2MiB regions by events:")
        for r in q(
            f"SELECT (address >> 21) AS reg, count(*) AS n, min(start) AS t0, max(start) AS t1 FROM {t} GROUP BY reg ORDER BY n DESC LIMIT 15"
        ):
            print(
                f"    0x{r['reg'] << 21:x} n={r['n']} {r['t0'] / 1e9:.3f}..{r['t1'] / 1e9:.3f}s"
            )
    print("  last 15 events:")
    for r in q(f"SELECT * FROM {t} ORDER BY start DESC LIMIT 15")[::-1]:
        print(
            "    "
            + " ".join(f"{k}={r[k]}" for k in c if k not in ("globalPid",))
        )

if "OSRT_API" in T:
    print("== OS runtime calls in the last 120 s of the capture, by name")
    mx = q("SELECT max(start) FROM OSRT_API")[0][0] or 0
    for r in q(
        "SELECT nameId, count(*) n, sum(end-start) d, max(end) e FROM OSRT_API WHERE start > ? GROUP BY nameId ORDER BY d DESC LIMIT 15",
        mx - 120e9,
    ):
        print(
            f"  {s(r['nameId'])}: n={r['n']} total={r['d'] / 1e6:.1f} ms last_end={r['e'] / 1e9:.3f}s"
        )
    print("== last 20 OS runtime calls")
    for r in q(
        "SELECT start, end, globalTid, nameId FROM OSRT_API ORDER BY start DESC LIMIT 20"
    )[::-1]:
        print(
            f"  {r['start'] / 1e9:.4f} {(r['end'] - r['start']) / 1e6:9.3f} {r['globalTid'] % 0x1000000} {s(r['nameId'])}"
        )

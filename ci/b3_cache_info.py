# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# DO NOT MERGE: summarize the CUDA driver JIT cache and the cupy kernel cache.

import collections
import os
import re
import sys

for root in sys.argv[1:]:
    root = os.path.expanduser(root)
    files = []
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            try:
                files.append((p, os.path.getsize(p)))
            except OSError:
                pass
    print(
        f"== {root}: {len(files)} files, {sum(s for _, s in files) / 1e6:.1f} MB"
    )
    names = collections.Counter()
    for p, _ in files:
        try:
            data = open(p, "rb").read()
        except OSError:
            continue
        found = set(
            m.decode(errors="replace")
            for m in re.findall(rb"_Z[0-9A-Za-z_]{12,}", data)
        )
        for n in found:
            names[n[:160]] += 1
    print(f"   distinct mangled symbols: {len(names)}")
    # Group by a short prefix (namespace/function) to see which libraries the modules come from.
    groups = collections.Counter()
    for n in names:
        m = re.findall(r"(?:N|_Z)\d+([A-Za-z_]\w*)", n)
        groups["::".join(m[:3]) if m else n[:40]] += 1
    for g, c in groups.most_common(40):
        print(f"   {c:5d} {g}")
    for n in sorted(names)[:15]:
        print(f"   e.g. {n}")

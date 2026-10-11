# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# DO NOT MERGE: match register values from a gdb dump of a stalled thread
# against /proc/<pid>/maps, to find what memory the stuck CUDA call targets
# (managed memory shows up as /dev/nvidia-uvm mappings).
#
# usage: python b3_regs.py <gdb output> <maps file>

import re
import sys

gdb_out = open(sys.argv[1], errors="replace").read()
maps = []
for line in open(sys.argv[2], errors="replace"):
    parts = line.split()
    lo, hi = (int(x, 16) for x in parts[0].split("-"))
    path = parts[5] if len(parts) > 5 else "[anon]"
    maps.append((lo, hi, parts[1], path))

frame = "?"
seen = {}
for line in gdb_out.splitlines():
    m = re.match(r"#(\d+)\s+(?:0x[0-9a-f]+ in )?(\S+)", line)
    if m:
        frame = f"#{m.group(1)} {m.group(2)[:60]}"
        continue
    m = re.match(r"(r\w\w|r\d+)\s+(0x[0-9a-f]+)", line)
    if not m:
        continue
    reg, val = m.group(1), int(m.group(2), 16)
    if val < 0x10000:
        continue
    for lo, hi, perm, path in maps:
        if lo <= val < hi:
            if (
                path.startswith("/opt/")
                or path.startswith("/usr/")
                or path == "[stack]"
            ):
                break
            key = (val, path)
            seen.setdefault(key, []).append(f"{frame}:{reg}")
            break
    else:
        seen.setdefault((val, "UNMAPPED (device-only VA?)"), []).append(
            f"{frame}:{reg}"
        )

print("== register values that point into non-library mappings")
for (val, path), where in sorted(seen.items(), key=lambda kv: kv[0][0]):
    flag = ""
    if path in ("[heap]", "[anon]"):
        flag = "  <-- PAGEABLE HOST (heap/anon)"
    elif "nvidia-uvm" in path:
        flag = "  <-- MANAGED (nvidia-uvm)"
    print(f"  0x{val:x} -> {path}{flag}   ({', '.join(where[:4])})")
print("== /dev/nvidia-uvm and /dev/nvidia* mappings (count by path)")
counts = {}
for lo, hi, perm, path in maps:
    if path.startswith("/dev/nvidia"):
        counts[path] = counts.get(path, 0) + (hi - lo)
for p, sz in counts.items():
    print(f"  {p}: {sz / 2**20:.1f} MiB")

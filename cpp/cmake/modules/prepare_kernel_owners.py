# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare unity-safe copies of existing host launch/instantiation sources.

Only TU-private type aliases are renamed. Kernel bodies and public names remain
unchanged. Quoted includes retain their original source-directory resolution.
"""

from pathlib import Path
import json
import re
import sys

source, build, groups_file = (Path(p).resolve() for p in sys.argv[1:])
groups = json.loads(groups_file.read_text())
out = build / "dedup_owners"
out.mkdir(exist_ok=True)


def mask(s):
    return re.sub(
        r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
        lambda m: "".join("\n" if c == "\n" else " " for c in m[0]),
        s,
        flags=re.S,
    )


manifest = []
cmake = ["set(_dedup_originals)\nset(_dedup_copies)\n"]
for i, group in enumerate(groups):
    for j, x in enumerate(group):
        kind, rel = x.split(":", 1)
        p = (source if kind == "sources" else build) / rel
        s = p.read_text()
        masked = mask(s)
        aliases = set()
        scopes = []
        for m in re.finditer(
            r"\bnamespace\b[^{};=]*\{|[{}]|\busing\s+(\w+)\s*=", masked
        ):
            token = m[0]
            if token.startswith("namespace"):
                scopes.append(True)
            elif token == "{":
                scopes.append(False)
            elif token == "}":
                scopes.pop()
            elif all(scopes):
                aliases.add(m[1])
        # CUDA's generated stubs can look up anonymous namespaces ambiguously
        # after formerly separate headers share a TU. Lift alias-only root scopes;
        # the per-file alias suffix still keeps every spelling private and unique.
        braces: list[tuple[int, int, bool]] = []
        alias_scopes = []
        for m in re.finditer(r"\bnamespace\s*\{|[{}]", masked):
            if m[0].endswith("{"):
                braces.append(
                    (
                        m.start(),
                        m.end(),
                        m[0].startswith("namespace") and not braces,
                    )
                )
            elif m[0] == "}":
                start, end, is_root_alias = braces.pop()
                if is_root_alias:
                    body = re.sub(
                        r"\busing\s+\w+\s*=[^;]+;", "", masked[end : m.start()]
                    )
                    if not body.strip():
                        alias_scopes.append((start, end, m.start()))
        for start, end, close in reversed(alias_scopes):
            s = s[:close] + " " + s[close + 1 :]
            s = (
                s[:start]
                + "".join("\n" if c == "\n" else " " for c in s[start:end])
                + s[end:]
            )
        masked = mask(s)
        suffix = f"_dedup_{i}_{j}"
        # Rename identifiers in code only, preserving file names, strings and comments.
        for a in sorted(aliases):
            hits = [
                m
                for m in re.finditer(r"\b" + re.escape(a) + r"\b", masked)
                if not re.search(r"::\s*$", masked[: m.start()])
            ]
            for m in reversed(hits):
                s = s[: m.start()] + a + suffix + s[m.end() :]
            masked = mask(s)

        def include(m):
            q = p.parent / m[1]
            return (
                '#include "' + str(q.resolve()) + '"' if q.exists() else m[0]
            )

        s = re.sub(
            r'^[ \t]*#[ \t]*include[ \t]*"([^"\n]+)"', include, s, flags=re.M
        )
        dest = out / f"{i}_{j}_{p.name}"
        text = (
            '#define CUVS_UNITY_KERNEL_OWNER 1\n#line 1 "' + str(p) + '"\n' + s
        )
        if not dest.exists() or dest.read_text() != text:
            dest.write_text(text)
        manifest.append(
            dict(
                original=str(p),
                copy=str(dest),
                group=i,
                aliases=sorted(aliases),
            )
        )
        cmake.append('list(APPEND _dedup_originals "' + str(p) + '")\n')
        cmake.append('list(APPEND _dedup_copies "' + str(dest) + '")\n')
        cmake.append(
            'set_property(SOURCE "'
            + str(dest)
            + '" PROPERTY UNITY_GROUP duplicate_owners_'
            + str(i)
            + ")\n"
        )
(out / "owners.cmake").write_text("".join(cmake))
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(
    "Prepared",
    len(manifest),
    "host source copies in",
    len(groups),
    "owning CUDA TUs",
)

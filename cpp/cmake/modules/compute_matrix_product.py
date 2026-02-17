# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This algorithm takes a JSON dictionary and computes a matrix product of all
# of its arrays. We use this to compute all matrix combinations for kernel
# generation. We *could* write this in CMake with `string(JSON)`, but writing
# it in Python is much easier. Once we have a version of CMake that has
# https://gitlab.kitware.com/cmake/cmake/-/merge_requests/11516, we may be able
# to port the algorithm to CMake script and use it in other RAPIDS projects.

import argparse
import json
import re
import sys
import warnings
from itertools import chain
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

    MatrixValue = None | bool | int | float | str
    Matrix = MatrixValue | list["Matrix"] | dict[str, "Matrix"]


class NoKeyError(ValueError):
    pass


class UnusedKeyWarning(UserWarning):
    pass


class UsedKeyWarning(UserWarning):
    pass


IDENTIFIER_RE: re.Pattern = re.compile(r"^(?P<underscores>_*)(?P<rest>.*)$")


def iterate_matrix_product(
    matrix: "Matrix",
    warn_unused=True,
    warn_used=True,
) -> "Generator[dict[str, MatrixValue]]":
    def iterate_next_dimension(
        queue: "Iterator[tuple[tuple[str | int, ...], str | None, Matrix]]",
        entry: "dict[str, MatrixValue]",
    ) -> "Generator[tuple[dict[str, MatrixValue], bool]]":
        try:
            path, key, matrix = next(queue)
        except StopIteration:
            yield (entry, False)
        else:
            used = False
            for e, u in iterate_impl(path, key, matrix, queue, entry):
                if u:
                    used = True
                yield e, u

            try:
                last = path[-1]
            except IndexError:
                pass
            else:
                if isinstance(last, str):
                    match = IDENTIFIER_RE.search(last)
                    assert match
                    underscores = match.group("underscores")
                    rest = match.group("rest")
                    path_repr = "".join(
                        f"[{json.dumps(i)}]" for i in path[:-1]
                    )

                    if warn_used and used and underscores:
                        warnings.warn(
                            f"Key {json.dumps(last)} at root{path_repr} "
                            f"is used in a matrix product entry even though it "
                            f"begins with {json.dumps(underscores)}. Consider "
                            f"renaming it to {json.dumps(rest)} to indicate this.",
                            category=UsedKeyWarning,
                        )
                    elif warn_unused and not used and not underscores:
                        warnings.warn(
                            f"Key {json.dumps(last)} at root{path_repr} "
                            f"is never used in a matrix product entry and is used "
                            f"only for grouping. Consider renaming it to "
                            f"{json.dumps(f'_{last}')} to indicate this.",
                            category=UnusedKeyWarning,
                        )

    def iterate_impl(
        path: tuple[str | int, ...],
        key: str | None,
        matrix: "Matrix",
        queue: "Iterator[tuple[tuple[str | int, ...], str | None, Matrix]]",
        entry: "dict[str, MatrixValue]",
    ) -> "Generator[tuple[dict[str, MatrixValue], bool]]":
        if isinstance(matrix, dict):
            yield from (
                (e, False)
                for e, _ in iterate_next_dimension(
                    chain(
                        (
                            ((*path, k), k, v)
                            for (k, v) in sorted(matrix.items())
                        ),
                        queue,
                    ),
                    entry,
                )
            )
        elif isinstance(matrix, list):
            queue_list = list(queue)
            for i, v in enumerate(matrix):
                yield from iterate_next_dimension(
                    chain([((*path, i), key, v)], queue_list), entry
                )
        else:
            if key is None:
                raise NoKeyError
            yield from (
                (e, True)
                for e, _ in iterate_next_dimension(
                    queue, {**entry, key: matrix}
                )
            )

    yield from (
        entry for entry, _ in iterate_impl((), None, matrix, chain(), {})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--warn-unused", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--warn-used", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("filename", nargs="?", default="-")
    namespace = parser.parse_args()
    with (
        sys.stdin
        if namespace.filename == "-"
        else open(namespace.filename) as f
    ):
        matrix: "Matrix" = json.load(f)

    json.dump(
        list(
            iterate_matrix_product(
                matrix,
                warn_unused=namespace.warn_unused,
                warn_used=namespace.warn_used,
            )
        ),
        sys.stdout,
        indent=2,
        sort_keys=True,
    )

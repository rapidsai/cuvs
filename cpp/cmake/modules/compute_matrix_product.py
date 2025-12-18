# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This algorithm takes a JSON dictionary and computes a matrix product of all
# of its arrays. We use this to compute all matrix combinations for kernel
# generation. We *could* write this in CMake with `string(JSON)`, but writing
# it in Python is much easier. Once we have a version of CMake that has
# https://gitlab.kitware.com/cmake/cmake/-/merge_requests/11516, we may be able
# to port the algorithm to CMake script and use it in other RAPIDS projects.

import json
import sys
from itertools import chain
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

    MatrixValue = None | bool | int | float | str
    Matrix = MatrixValue | list["Matrix"] | dict[str, "Matrix"]


def iterate_matrix_product(
    matrix: "Matrix",
) -> "Generator[dict[str, MatrixValue]]":
    def iterate_next_dimension(
        queue: "Iterator[tuple[str | None, Matrix]]",
        entry: "dict[str, MatrixValue]",
    ) -> "Generator[dict[str, MatrixValue]]":
        try:
            key, matrix = next(queue)
        except StopIteration:
            yield entry
        else:
            yield from iterate_impl(matrix, key, queue, entry)

    def iterate_impl(
        matrix: "Matrix",
        key: str | None,
        queue: "Iterator[tuple[str | None, Matrix]]",
        entry: "dict[str, MatrixValue]",
    ) -> "Generator[dict[str, MatrixValue]]":
        if isinstance(matrix, dict):
            yield from iterate_next_dimension(
                chain(sorted(matrix.items()), queue),
                entry,
            )
        elif isinstance(matrix, list):
            queue_list = list(queue)
            for v in matrix:
                yield from iterate_next_dimension(
                    chain([(key, v)], queue_list), entry
                )
        else:
            assert key is not None
            yield from iterate_next_dimension(queue, {**entry, key: matrix})

    return iterate_impl(matrix, None, chain(), {})


try:
    filename = sys.argv[1]
except IndexError:
    filename = "-"
with sys.stdin if filename == "-" else open(filename) as f:
    matrix: "Matrix" = json.load(f)

json.dump(
    list(iterate_matrix_product(matrix)),
    sys.stdout,
    indent=2,
    sort_keys=True,
)

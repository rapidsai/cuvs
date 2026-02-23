# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import pathlib
import runpy
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    MatrixValue = None | bool | int | float | str
    Matrix = MatrixValue | list["Matrix"] | dict[str, "Matrix"]

compute_matrix_product_script = runpy.run_path(
    str(
        pathlib.Path(__file__).parent
        / "../../cmake/modules/compute_matrix_product.py"
    )
)

iterate_matrix_product: "Callable[[Matrix], Generator[dict[str, MatrixValue]]]" = compute_matrix_product_script[
    "iterate_matrix_product"
]
NoKeyError: type[ValueError] = compute_matrix_product_script["NoKeyError"]
UnusedKeyWarning: type[UserWarning] = compute_matrix_product_script[
    "UnusedKeyWarning"
]
UsedKeyWarning: type[UserWarning] = compute_matrix_product_script[
    "UsedKeyWarning"
]


@pytest.mark.parametrize(
    ["matrix", "warn_unused", "warn_used", "expected_product", "contexts"],
    [
        pytest.param(
            {
                "a": [1, 2],
                "b": [3, 4],
            },
            True,
            True,
            [
                {"a": 1, "b": 3},
                {"a": 1, "b": 4},
                {"a": 2, "b": 3},
                {"a": 2, "b": 4},
            ],
            [],
            id="basic",
        ),
        pytest.param(
            {
                "b": [3, 4],
                "a": [1, 2],
            },
            True,
            True,
            [
                {"a": 1, "b": 3},
                {"a": 1, "b": 4},
                {"a": 2, "b": 3},
                {"a": 2, "b": 4},
            ],
            [],
            id="sort",
        ),
        pytest.param(
            {
                "_a": [
                    {"b": 1, "c": 2},
                    {"b": 3, "c": 4},
                ],
                "_d": [
                    {"e": 5, "f": 6},
                    {"e": 7, "f": 8},
                ],
            },
            True,
            True,
            [
                {"b": 1, "c": 2, "e": 5, "f": 6},
                {"b": 1, "c": 2, "e": 7, "f": 8},
                {"b": 3, "c": 4, "e": 5, "f": 6},
                {"b": 3, "c": 4, "e": 7, "f": 8},
            ],
            [],
            id="subgroup",
        ),
        pytest.param(
            {
                "a": [[1, 2], [3, 4]],
            },
            True,
            True,
            [
                {"a": 1},
                {"a": 2},
                {"a": 3},
                {"a": 4},
            ],
            [],
            id="nested-lists",
        ),
        pytest.param(
            [
                {"a": 1},
                {"a": 2},
            ],
            True,
            True,
            [
                {"a": 1},
                {"a": 2},
            ],
            [],
            id="list-root",
        ),
        pytest.param(
            [1],
            True,
            True,
            None,
            [pytest.raises(NoKeyError)],
            id="list-root-no-dict",
        ),
        pytest.param(
            [[1]],
            True,
            True,
            None,
            [pytest.raises(NoKeyError)],
            id="nested-list-root-no-dict",
        ),
        pytest.param(
            1,
            True,
            True,
            None,
            [pytest.raises(NoKeyError)],
            id="item-root",
        ),
        pytest.param(
            {
                "a": [
                    {
                        "b": {"c": 1},
                    },
                ],
            },
            True,
            True,
            [
                {"c": 1},
            ],
            [
                pytest.warns(
                    UnusedKeyWarning,
                    match=r'^Key "a" at root is never used in a matrix product '
                    r"entry and is used only for grouping\. Consider renaming "
                    r'it to "_a" to indicate this\.$',
                ),
                pytest.warns(
                    UnusedKeyWarning,
                    match=r'^Key "b" at root\["a"\]\[0\] is never used in a '
                    r"matrix product entry and is used only for grouping\. "
                    r'Consider renaming it to "_b" to indicate this\.$',
                ),
            ],
            id="unused",
        ),
        pytest.param(
            {
                "a": [
                    {
                        "b": {"c": 1},
                    },
                ],
            },
            False,
            True,
            [
                {"c": 1},
            ],
            [],
            id="unused-no-warning",
        ),
        pytest.param(
            {
                "_a": [
                    {
                        "_b": {"_c": 1},
                    },
                ],
                "__b": 1,
            },
            True,
            True,
            [
                {"__b": 1, "_c": 1},
            ],
            [
                pytest.warns(
                    UsedKeyWarning,
                    match=r'^Key "_c" at root\["_a"\]\[0\]\["_b"\] is used in '
                    r'a matrix product entry even though it begins with "_"\. '
                    r'Consider renaming it to "c" to indicate this\.$',
                ),
                pytest.warns(
                    UsedKeyWarning,
                    match=r'^Key "__b" at root is used in a matrix product '
                    r'entry even though it begins with "__"\. Consider '
                    r'renaming it to "b" to indicate this\.$',
                ),
            ],
            id="used",
        ),
        pytest.param(
            {
                "_a": [
                    {
                        "_b": {"_c": 1},
                    },
                ],
                "__b": 1,
            },
            True,
            False,
            [
                {"__b": 1, "_c": 1},
            ],
            [],
            id="used-no-warning",
        ),
    ],
)
def test_iterate_matrix_product(
    matrix, warn_unused, warn_used, expected_product, contexts
):
    with contextlib.ExitStack() as stack:
        for context in contexts:
            stack.enter_context(context)
        assert (
            list(
                iterate_matrix_product(
                    matrix=matrix, warn_unused=warn_unused, warn_used=warn_used
                )
            )
            == expected_product
        )

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.preprocessing.quantize import binary


@pytest.mark.parametrize("n_rows", [50, 100])
@pytest.mark.parametrize("n_cols", [10, 50])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("device_memory", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
def test_binary_quantizer(n_rows, n_cols, inplace, device_memory, dtype):
    input1 = np.random.random_sample((n_rows, n_cols)).astype(dtype)

    output_cols = int(np.ceil(n_cols / 8))
    output = (
        np.zeros((n_rows, output_cols), dtype="uint8") if inplace else None
    )

    input1_device = device_ndarray(input1)
    output_device = device_ndarray(output) if inplace else None

    transformed = binary.transform(
        input1_device if device_memory else input1,
        output=(output_device if device_memory else output)
        if inplace
        else None,
    )
    if device_memory:
        actual = transformed if not inplace else output_device
        actual = actual.copy_to_host()
    else:
        actual = transformed if not inplace else output

    expected = np.packbits(input1 > 0, axis=-1, bitorder="little")
    assert np.all(actual == expected)

#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


def _check_input_array(
    cai, exp_dt, exp_rows=None, exp_cols=None, exp_row_major=True
):
    if cai.dtype not in exp_dt:
        raise TypeError("dtype %s not supported" % cai.dtype)

    if exp_row_major and not cai.c_contiguous:
        raise ValueError("Row major input is expected")

    if exp_cols is not None and cai.shape[1] != exp_cols:
        raise ValueError(
            "Incorrect number of columns, expected {} got {}".format(
                exp_cols, cai.shape[1]
            )
        )

    if exp_rows is not None and cai.shape[0] != exp_rows:
        raise ValueError(
            "Incorrect number of rows, expected {} , got {}".format(
                exp_rows, cai.shape[0]
            )
        )


def _check_memory_location(array_like, expected_host=True, name="array"):
    """
    Check if array is in expected memory location for multi-GPU operations.

    Parameters
    ----------
    array_like : array-like
        Array to check memory location of
    expected_host : bool, default=True
        If True, expects host memory. If False, expects device memory.
    name : str
        Name of the array for error messages

    Raises
    ------
    ValueError
        If array is not in expected memory location
    """
    # Check if array has __cuda_array_interface__ (device memory indicator)
    has_cuda_interface = hasattr(array_like, "__cuda_array_interface__")

    # Check if array is NumPy array (host memory indicator)
    is_numpy = isinstance(array_like, np.ndarray)

    if expected_host:
        if has_cuda_interface and not is_numpy:
            raise ValueError(
                f"Multi-GPU IVF-PQ requires {name} to be in host memory "
                f"(CPU), but received device memory (GPU). Please use "
                f"array.get() or cp.asnumpy(array) to transfer to host memory."
            )
    else:
        if is_numpy and not has_cuda_interface:
            raise ValueError(
                f"Expected {name} to be in device memory (GPU), but received "
                f"host memory (CPU). Please use cp.asarray(array) to transfer "
                f"to device memory."
            )

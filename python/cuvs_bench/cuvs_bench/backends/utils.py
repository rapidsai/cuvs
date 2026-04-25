#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""
Shared utilities for cuvs-bench backends.

Provides common functions used by Python-native backends (e.g., OpenSearch,
Elasticsearch) that need to load dataset vectors from binary files. The C++
backend does not use these since it passes file paths directly to the
subprocess.
"""

import os
from typing import Optional

import numpy as np

_DTYPE_FOR_EXT = {
    ".fbin": np.float32,
    ".f16bin": np.float16,
    ".u8bin": np.uint8,
    ".i8bin": np.int8,
    ".ibin": np.int32,
}


def load_vectors(path: str, subset_size: Optional[int] = None) -> np.ndarray:
    """
    Read a binary vector file into a numpy array.

    Supports the standard big-ann-bench binary format used by cuvs-bench
    datasets: a 4-byte uint32 ``n_rows``, a 4-byte uint32 ``n_cols``,
    followed by ``n_rows * n_cols`` elements of the dtype inferred from
    the file extension.

    Parameters
    ----------
    path : str
        Path to the binary file. The dtype is inferred from the extension:
        ``.fbin`` (float32), ``.f16bin`` (float16), ``.u8bin`` (uint8),
        ``.i8bin`` (int8), ``.ibin`` (int32).
    subset_size : Optional[int]
        If provided, only the first ``subset_size`` rows are loaded.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_rows, n_cols)`` with the inferred dtype.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    ext = os.path.splitext(path)[1].lower()
    dtype = _DTYPE_FOR_EXT.get(ext, np.float32)
    with open(path, "rb") as f:
        n_rows = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])
        n_cols = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])
        if subset_size is not None:
            n_rows = min(n_rows, subset_size)
        raw = f.read(n_rows * n_cols * np.dtype(dtype).itemsize)
        data = np.frombuffer(raw, dtype=dtype)
    return data.reshape(n_rows, n_cols)

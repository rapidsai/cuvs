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
    if ext not in _DTYPE_FOR_EXT:
        supported = ", ".join(_DTYPE_FOR_EXT.keys())
        raise ValueError(
            f"Unsupported vector file extension '{ext}' for path: {path}. "
            f"Supported extensions: {supported}"
        )
    if subset_size is not None and subset_size < 1:
        raise ValueError(
            f"subset_size must be a positive integer, got {subset_size}"
        )
    dtype = _DTYPE_FOR_EXT[ext]
    with open(path, "rb") as f:
        header = f.read(8)
        if len(header) < 8:
            raise ValueError(
                f"File too small to contain a valid header (expected 8 bytes, "
                f"got {len(header)}): {path}"
            )
        n_rows = int(np.frombuffer(header[:4], dtype=np.uint32)[0])
        n_cols = int(np.frombuffer(header[4:], dtype=np.uint32)[0])
        if subset_size is not None:
            n_rows = min(n_rows, subset_size)
        expected_bytes = n_rows * n_cols * np.dtype(dtype).itemsize
        raw = f.read(expected_bytes)
        if len(raw) < expected_bytes:
            raise ValueError(
                f"File is truncated: expected {expected_bytes} bytes of data "
                f"({n_rows} rows x {n_cols} cols x {np.dtype(dtype).itemsize} bytes), "
                f"got {len(raw)}: {path}"
            )
        data = np.frombuffer(raw, dtype=dtype)
    return data.reshape(n_rows, n_cols)

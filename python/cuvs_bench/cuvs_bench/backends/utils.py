#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""
Shared utilities for cuvs-bench backends.

Provides common functions for reading binary vector files. Used internally
by the Dataset class for transparent vector loading. Python-native backends
(e.g., OpenSearch, Elasticsearch) access vectors via Dataset properties
and do not need to call these functions directly. The C++ backend does not
use these since it passes file paths directly to the subprocess.

The dtype_from_filename function originates from generate_groundtruth/utils.py.
Note: generate_groundtruth/utils.py uses .hbin for float16 while the rest of
the codebase (get_dataset/fbin_to_f16bin.py, OpenSearch backend) uses .f16bin.
We standardize on .f16bin here to match the naming convention of the other
formats (.fbin, .i8bin, .u8bin).
"""

import os
from typing import Optional

import numpy as np


def dtype_from_filename(filename):
    """Map file extension to numpy dtype.

    Parameters
    ----------
    filename : str
        Path or filename with a supported extension.

    Returns
    -------
    numpy.dtype
        The corresponding numpy dtype.

    Raises
    ------
    RuntimeError
        If the file extension is not supported.
    """
    ext = os.path.splitext(filename)[1]
    if ext == ".fbin":
        return np.float32
    if ext == ".f16bin":
        return np.float16
    elif ext == ".ibin":
        return np.int32
    elif ext == ".u8bin":
        return np.ubyte
    elif ext == ".i8bin":
        return np.byte
    else:
        raise RuntimeError(f"Unsupported file extension: {ext}")


def load_vectors(path: str, subset_size: Optional[int] = None) -> np.ndarray:
    """
    Read a binary vector file into a numpy array.

    Supports the standard big-ann-bench binary format used by cuvs-bench
    datasets: a 4-byte uint32 ``n_rows``, a 4-byte uint32 ``n_cols``,
    followed by ``n_rows * n_cols`` elements of the dtype inferred from
    the file extension via ``dtype_from_filename``.

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
    ValueError
        If the file extension is unsupported, ``subset_size`` is not positive,
        or the file is truncated.
    """
    dtype = dtype_from_filename(path)
    if subset_size is not None and subset_size < 1:
        raise ValueError(
            f"subset_size must be a positive integer, got {subset_size}"
        )
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

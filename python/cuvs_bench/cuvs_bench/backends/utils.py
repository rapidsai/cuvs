#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""
Shared utilities for cuvs-bench backends.

Provides common functions for Python-native backends (e.g., OpenSearch,
Elasticsearch):
- Vector file I/O: used internally by the Dataset class for transparent
  vector loading. The C++ backend does not use these since it passes file
  paths directly to the subprocess.
- Parameter expansion: converts YAML param specs into lists of param dicts.
- Recall computation: computes recall@k from neighbors and ground truth.

The dtype_from_filename function originates from generate_groundtruth/utils.py.
Note: generate_groundtruth/utils.py uses .hbin for float16 while the rest of
the codebase (get_dataset/fbin_to_f16bin.py, OpenSearch backend) uses .f16bin.
We standardize on .f16bin here to match the naming convention of the other
formats (.fbin, .i8bin, .u8bin).
"""

import itertools
import os
from typing import Any, Dict, List, Optional

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


def expand_param_grid(param_spec: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Expand a parameter specification into all combinations via Cartesian product.

    Takes a dict where each key maps to a list of values (as defined in
    algorithm YAML configs) and produces a list of dicts, one per combination.

    Parameters
    ----------
    param_spec : Dict[str, List[Any]]
        Parameter specification, e.g., {"m": [16, 32], "ef_construction": [100, 200]}

    Returns
    -------
    List[Dict[str, Any]]
        List of parameter dicts, e.g.,
        [{"m": 16, "ef_construction": 100}, {"m": 16, "ef_construction": 200},
         {"m": 32, "ef_construction": 100}, {"m": 32, "ef_construction": 200}]
        Returns [{}] if param_spec is empty.
    """
    if not param_spec:
        return [{}]
    keys = list(param_spec.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*param_spec.values())]


def compute_recall(
    neighbors: np.ndarray, groundtruth: np.ndarray, k: int
) -> float:
    """
    Compute recall@k by comparing returned neighbors against ground truth.

    For each query, counts how many of the k returned neighbors appear
    in the ground truth set, then averages across all queries.

    Parameters
    ----------
    neighbors : np.ndarray
        Returned neighbor IDs, shape (n_queries, k)
    groundtruth : np.ndarray
        Ground truth neighbor IDs, shape (n_queries, gt_k)
    k : int
        Number of neighbors to evaluate

    Returns
    -------
    float
        Recall@k in range [0.0, 1.0]
    """
    n_queries = neighbors.shape[0]
    gt_k = min(k, groundtruth.shape[1])
    if gt_k == 0 or n_queries == 0:
        return 0.0
    n_correct = sum(
        len(set(neighbors[i, :k].tolist()) & set(groundtruth[i, :gt_k].tolist()))
        for i in range(n_queries)
    )
    return n_correct / (n_queries * gt_k)

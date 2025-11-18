# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.preprocessing.quantize import pq


@pytest.mark.parametrize("n_rows", [450, 700])
@pytest.mark.parametrize("n_cols", [64, 128])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("pq_kmeans_type", ["kmeans", "kmeans_balanced"])
@pytest.mark.parametrize("use_vq", [True, False])
@pytest.mark.parametrize("use_subspaces", [True, False])
def test_product_quantizer(
    n_rows, n_cols, inplace, dtype, pq_kmeans_type, use_vq, use_subspaces
):
    pq_dim = 32
    pq_bits = 8
    vq_n_centers = 0
    input1 = np.random.random_sample((n_rows, n_cols)).astype(dtype)
    input1_device = device_ndarray(input1)

    params = pq.QuantizerParams(
        pq_bits=pq_bits,
        pq_dim=pq_dim,
        vq_n_centers=vq_n_centers,
        pq_kmeans_trainset_fraction=1,
        pq_kmeans_type=pq_kmeans_type,
        use_vq=use_vq,
        use_subspaces=use_subspaces,
    )
    quantizer = pq.train(params, input1_device)

    output = (
        np.zeros((n_rows, quantizer.encoded_dim), dtype="uint8")
        if inplace
        else None
    )
    output_device = device_ndarray(output) if inplace else None
    transformed = pq.transform(quantizer, input1_device, output=output_device)
    actual = transformed if not inplace else output_device
    actual = actual.copy_to_host()

    # naive tests
    assert actual.any()  # Check that the output is not all zeros
    assert not np.isnan(actual).any()  # Check that the output has no NaNs
    assert cp.array(quantizer.pq_codebook).any()

    reconstructed = cp.empty((n_rows, n_cols), dtype=dtype)
    pq.inverse_transform(quantizer, transformed, reconstructed)
    reconstructed = cp.array(reconstructed)
    assert reconstructed.shape == input1.shape
    reconstruction_error = cp.linalg.norm(
        cp.array(input1_device) - reconstructed, axis=1
    )
    assert reconstruction_error.mean() < 1

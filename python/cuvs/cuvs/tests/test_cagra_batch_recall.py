# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Test for CAGRA search recall consistency across different query batch sizes.
# Regression test for https://github.com/rapidsai/cuvs/issues/1187

import cupy as cp
import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.neighbors import cagra


def calc_recall(found_indices, ground_truth, k):
    """Calculate recall@k between found indices and ground truth."""
    found_indices = cp.asnumpy(cp.asarray(found_indices))
    ground_truth = cp.asnumpy(cp.asarray(ground_truth))
    n_queries = found_indices.shape[0]
    correct = 0
    total = n_queries * k
    for i in range(n_queries):
        correct += len(
            set(found_indices[i, :k].tolist())
            & set(ground_truth[i, :k].tolist())
        )
    return correct / total


@pytest.mark.parametrize("search_width", [1, 4, 8])
def test_cagra_batch_recall_consistency(search_width):
    """
    Test that CAGRA search recall does not depend on query batch size.

    When the AUTO algorithm selector switches from MULTI_CTA to SINGLE_CTA
    at larger batch sizes, the recall should remain consistent. Previously,
    SINGLE_CTA with search_width > 1 computed far fewer max_iterations than
    MULTI_CTA, causing a significant recall drop at large batch sizes.
    """
    np.random.seed(42)
    cp.random.seed(42)

    n_samples = 50000
    n_queries = 2048
    dim = 64
    k = 10

    dataset = np.random.randn(n_samples, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)

    dataset_device = device_ndarray(dataset)
    queries_device = device_ndarray(queries)

    # Build CAGRA index
    build_params = cagra.IndexParams(
        graph_degree=32, intermediate_graph_degree=64
    )
    index = cagra.build(build_params, dataset_device)

    search_params = cagra.SearchParams(
        itopk_size=64, search_width=search_width
    )

    # Compute ground truth with batch_size=1 (always uses MULTI_CTA)
    gt_neighbors = np.zeros((n_queries, k), dtype=np.uint32)
    for i in range(n_queries):
        q = queries_device[i : i + 1]
        n_out = device_ndarray(np.zeros((1, k), dtype=np.uint32))
        d_out = device_ndarray(np.zeros((1, k), dtype=np.float32))
        cagra.search(search_params, index, q, k, neighbors=n_out, distances=d_out)
        gt_neighbors[i] = cp.asnumpy(cp.asarray(n_out))

    # Test different batch sizes including ones that trigger SINGLE_CTA
    batch_sizes = [64, 256, 512, 1024]
    recalls = []

    for batch_size in batch_sizes:
        all_neighbors = np.zeros((n_queries, k), dtype=np.uint32)
        for start in range(0, n_queries, batch_size):
            end = min(start + batch_size, n_queries)
            q = queries_device[start:end]
            actual_batch = end - start
            n_out = device_ndarray(
                np.zeros((actual_batch, k), dtype=np.uint32)
            )
            d_out = device_ndarray(
                np.zeros((actual_batch, k), dtype=np.float32)
            )
            cagra.search(
                search_params, index, q, k, neighbors=n_out, distances=d_out
            )
            all_neighbors[start:end] = cp.asnumpy(cp.asarray(n_out))

        recall = calc_recall(all_neighbors, gt_neighbors, k)
        recalls.append(recall)

    # Recall should be consistent across batch sizes (within small tolerance)
    recall_std = np.std(recalls)
    min_recall = min(recalls)
    max_recall = max(recalls)

    assert recall_std < 0.02, (
        f"Recall varies too much with batch size (std={recall_std:.4f}). "
        f"Recalls by batch size: "
        + ", ".join(
            f"{bs}={r:.4f}" for bs, r in zip(batch_sizes, recalls)
        )
    )
    assert (max_recall - min_recall) < 0.05, (
        f"Recall range too wide ({max_recall - min_recall:.4f}). "
        f"Recalls by batch size: "
        + ", ".join(
            f"{bs}={r:.4f}" for bs, r in zip(batch_sizes, recalls)
        )
    )

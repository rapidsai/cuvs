# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for cuVS resources and pylibraft handle compatibility."""

import numpy as np
import pytest
from pylibraft.common import DeviceResources, device_ndarray

from cuvs.common.resources import _PylibraftHandleWrapper
from cuvs.neighbors import brute_force


# --- Pylibraft Handle Wrapper Tests ---


def test_wrapper_with_device_resources():
    """Test that wrapper correctly wraps DeviceResources."""
    handle = DeviceResources()
    wrapper = _PylibraftHandleWrapper(handle)

    # Should have get_c_obj method
    assert hasattr(wrapper, "get_c_obj")
    assert hasattr(wrapper, "sync")

    # get_c_obj should return the same pointer as getHandle
    assert wrapper.get_c_obj() == handle.getHandle()


def test_wrapper_rejects_invalid_object():
    """Test that wrapper raises TypeError for invalid objects."""
    with pytest.raises(TypeError, match="cuVS Resources or pylibraft"):
        _PylibraftHandleWrapper("invalid")

    with pytest.raises(TypeError, match="cuVS Resources or pylibraft"):
        _PylibraftHandleWrapper(123)

    with pytest.raises(TypeError, match="cuVS Resources or pylibraft"):
        _PylibraftHandleWrapper(None)


# --- Pylibraft DeviceResources Compatibility Tests ---


def test_brute_force_build_with_pylibraft_handle():
    """Test brute_force.build with pylibraft DeviceResources."""
    n_samples, n_features = 100, 10

    dataset = np.random.random((n_samples, n_features)).astype(np.float32)
    dataset_device = device_ndarray(dataset)

    # Use pylibraft DeviceResources instead of cuVS Resources
    handle = DeviceResources()

    index = brute_force.build(dataset_device, resources=handle)

    # Explicit sync since we passed our own handle
    handle.sync()

    assert index.trained


# --- Multi-GPU Tests ---


def test_snmg_wrapper_with_device_resources_snmg():
    """Test that SNMG wrapper correctly wraps DeviceResourcesSNMG."""
    try:
        from pylibraft.common import DeviceResourcesSNMG
    except ImportError:
        pytest.skip("DeviceResourcesSNMG not available")

    from cuvs.common.mg_resources import _PylibraftSNMGWrapper

    # This test can run on single GPU - just testing the wrapper
    handle = DeviceResourcesSNMG()
    wrapper = _PylibraftSNMGWrapper(handle)

    assert hasattr(wrapper, "get_c_obj")
    assert hasattr(wrapper, "sync")
    assert wrapper.get_c_obj() == handle.getHandle()


def test_mg_algorithm_with_pylibraft_snmg_handle():
    """Test multi-GPU algorithm with pylibraft DeviceResourcesSNMG."""
    from pylibraft.common import DeviceResourcesSNMG

    from cuvs.neighbors.mg import cagra as mg_cagra

    n_rows, n_cols = 1000, 8

    dataset = np.random.random((n_rows, n_cols)).astype(np.float32)

    # Use pylibraft DeviceResourcesSNMG
    handle = DeviceResourcesSNMG()

    build_params = mg_cagra.IndexParams(
        graph_degree=16, intermediate_graph_degree=32
    )
    index = mg_cagra.build(build_params, dataset, resources=handle)

    # Explicit sync since we passed our own handle
    handle.sync()

    assert index.trained


def test_snmg_wrapper_rejects_invalid_object():
    """Test that SNMG wrapper raises TypeError for invalid objects."""
    from cuvs.common.mg_resources import _PylibraftSNMGWrapper

    with pytest.raises(TypeError, match="MultiGpuResources or pylibraft"):
        _PylibraftSNMGWrapper("invalid")

    with pytest.raises(TypeError, match="MultiGpuResources or pylibraft"):
        _PylibraftSNMGWrapper(123)

# Copyright (c) 2025, NVIDIA CORPORATION.
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
import pytest
from pylibraft.common import device_ndarray

from cuvs.neighbors import vamana


def _gen_data(shape, dtype):
    rng = np.random.default_rng(12345)
    if dtype == np.float32:
        return rng.random(shape, dtype=np.float32)
    if dtype == np.int8:
        # keep small magnitude to avoid overflow if used elsewhere
        return rng.integers(low=-10, high=10, size=shape, dtype=np.int8)
    if dtype == np.uint8:
        return rng.integers(low=0, high=20, size=shape, dtype=np.uint8)
    raise AssertionError("unexpected dtype in test helper")


@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.uint8])
def test_vamana_build_basic(dtype):
    n_rows, n_cols = 1000, 16
    data = _gen_data((n_rows, n_cols), dtype)
    data_dev = device_ndarray(data)

    params = vamana.IndexParams(metric="sqeuclidean")
    idx = vamana.build(params, data_dev)

    # Basic sanity: object type and flags
    assert isinstance(idx, vamana.Index)
    assert idx.trained is True
    # repr should include trained flag
    assert "Index(type=Vamana" in repr(idx)


def test_vamana_index_params_defaults_and_properties():
    params = vamana.IndexParams()

    # Defaults
    assert params.metric == "sqeuclidean"
    assert params.graph_degree == 32
    assert params.visited_size == 64
    assert params.vamana_iters == 1
    assert np.isclose(params.alpha, 1.2)
    assert np.isclose(params.max_fraction, 0.06)
    assert np.isclose(params.batch_base, 2.0)
    assert params.queue_size == 127
    assert params.reverse_batchsize == 1_000_000

    params.metric = "euclidean"
    assert params.metric == "euclidean"


def test_vamana_index_params_invalid_metric():
    with pytest.raises(ValueError):
        _ = vamana.IndexParams(metric="not_a_metric")


@pytest.mark.parametrize("include_dataset", [True, False])
def test_vamana_serialize(tmp_path, include_dataset):
    n_rows, n_cols = 256, 8
    data = _gen_data((n_rows, n_cols), np.float32)
    data_dev = device_ndarray(data)

    params = vamana.IndexParams()
    idx = vamana.build(params, data_dev)

    out_path = tmp_path / "vamana_index.bin"
    vamana.save(str(out_path), idx, include_dataset=include_dataset)

    assert out_path.exists() and out_path.stat().st_size > 0


def test_vamana_build_rejects_unsupported_dtype():
    # float16 is not in the accepted list in the build wrapper; expect failure
    n_rows, n_cols = 64, 8
    data = _gen_data((n_rows, n_cols), np.float32).astype(np.float16)
    from pylibraft.common import device_ndarray

    data_dev = device_ndarray(data)
    params = vamana.IndexParams()

    with pytest.raises(Exception):
        _ = vamana.build(params, data_dev)

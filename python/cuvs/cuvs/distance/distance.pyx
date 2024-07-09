#
# Copyright (c) 2024, NVIDIA CORPORATION.
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
#
# cython: language_level=3

import numpy as np

from cuvs.common.exceptions import check_cuvs
from cuvs.common.resources import auto_sync_resources

from cuvs.common cimport cydlpack

from pylibraft.common import auto_convert_output, device_ndarray
from pylibraft.common.cai_wrapper import wrap_array

DISTANCE_TYPES = {
    "l2": cuvsDistanceType.L2SqrtExpanded,
    "sqeuclidean": cuvsDistanceType.L2Expanded,
    "euclidean": cuvsDistanceType.L2SqrtExpanded,
    "l1": cuvsDistanceType.L1,
    "cityblock": cuvsDistanceType.L1,
    "inner_product": cuvsDistanceType.InnerProduct,
    "chebyshev": cuvsDistanceType.Linf,
    "canberra": cuvsDistanceType.Canberra,
    "cosine": cuvsDistanceType.CosineExpanded,
    "lp": cuvsDistanceType.LpUnexpanded,
    "correlation": cuvsDistanceType.CorrelationExpanded,
    "jaccard": cuvsDistanceType.JaccardExpanded,
    "hellinger": cuvsDistanceType.HellingerExpanded,
    "braycurtis": cuvsDistanceType.BrayCurtis,
    "jensenshannon": cuvsDistanceType.JensenShannon,
    "hamming": cuvsDistanceType.HammingUnexpanded,
    "kl_divergence": cuvsDistanceType.KLDivergence,
    "minkowski": cuvsDistanceType.LpUnexpanded,
    "russellrao": cuvsDistanceType.RusselRaoExpanded,
    "dice": cuvsDistanceType.DiceExpanded,
}

SUPPORTED_DISTANCES = ["euclidean", "l1", "cityblock", "l2", "inner_product",
                       "chebyshev", "minkowski", "canberra", "kl_divergence",
                       "correlation", "russellrao", "hellinger", "lp",
                       "hamming", "jensenshannon", "cosine", "sqeuclidean"]


@auto_sync_resources
@auto_convert_output
def pairwise_distance(X, Y, out=None, metric="euclidean", metric_arg=2.0,
                      resources=None):
    """
    Compute pairwise distances between X and Y

    Valid values for metric:
        ["euclidean", "l2", "l1", "cityblock", "inner_product",
         "chebyshev", "canberra", "lp", "hellinger", "jensenshannon",
         "kl_divergence", "russellrao", "minkowski", "correlation",
         "cosine"]

    Parameters
    ----------

    X : CUDA array interface compliant matrix shape (m, k)
    Y : CUDA array interface compliant matrix shape (n, k)
    out : Optional writable CUDA array interface matrix shape (m, n)
    metric : string denoting the metric type (default="euclidean")
    metric_arg : metric parameter (currently used only for "minkowski")
    {resources_docstring}

    Examples
    --------

    >>> import cupy as cp
    >>> from cuvs.distance import pairwise_distance
    >>> n_samples = 5000
    >>> n_features = 50
    >>> in1 = cp.random.random_sample((n_samples, n_features),
    ...                               dtype=cp.float32)
    >>> in2 = cp.random.random_sample((n_samples, n_features),
    ...                               dtype=cp.float32)
    >>> output = pairwise_distance(in1, in2, metric="euclidean")
    """

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    x_cai = wrap_array(X)
    y_cai = wrap_array(Y)

    m = x_cai.shape[0]
    n = y_cai.shape[0]

    if out is None:
        out = device_ndarray.empty((m, n), dtype=y_cai.dtype)
    out_cai = wrap_array(out)

    x_k = x_cai.shape[1]
    y_k = y_cai.shape[1]

    if x_k != y_k:
        raise ValueError("Inputs must have same number of columns. "
                         "a=%s, b=%s" % (x_k, y_k))

    if metric not in SUPPORTED_DISTANCES:
        raise ValueError("metric %s is not supported" % metric)

    cdef cuvsDistanceType distance_type = DISTANCE_TYPES[metric]

    x_dt = x_cai.dtype
    y_dt = y_cai.dtype
    d_dt = out_cai.dtype

    if x_dt != y_dt or x_dt != d_dt:
        raise ValueError("Inputs must have the same dtypes")

    cdef cydlpack.DLManagedTensor* x_dlpack = \
        cydlpack.dlpack_c(x_cai)
    cdef cydlpack.DLManagedTensor* y_dlpack = \
        cydlpack.dlpack_c(y_cai)
    cdef cydlpack.DLManagedTensor* out_dlpack = \
        cydlpack.dlpack_c(out_cai)

    check_cuvs(cuvsPairwiseDistance(res,
                                    x_dlpack,
                                    y_dlpack,
                                    out_dlpack,
                                    distance_type,
                                    metric_arg))

    return out

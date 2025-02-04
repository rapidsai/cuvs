#
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
#
# cython: language_level=3

import numpy as np

cimport cuvs.common.cydlpack

from cuvs.common.resources import auto_sync_resources

from cython.operator cimport dereference as deref
from libcpp cimport bool, cast
from libcpp.string cimport string

from cuvs.common cimport cydlpack
from cuvs.distance_type cimport cuvsDistanceType

from pylibraft.common import auto_convert_output, cai_wrapper, device_ndarray
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible

from cuvs.distance import DISTANCE_TYPES
from cuvs.neighbors.common import _check_input_array

from libc.stdint cimport (
    int8_t,
    int64_t,
    uint8_t,
    uint32_t,
    uint64_t,
    uintptr_t,
)

from cuvs.common.exceptions import check_cuvs


cdef class IndexParams:
    """
    Parameters to build NN-Descent Index

    Parameters
    ----------
    metric : str, default = "sqeuclidean"
        String denoting the metric type.
        distribution of the newly added data.
    graph_degree :  int
        For an input dataset of dimensions (N, D), determines the final
        dimensions of the all-neighbors knn graph which turns out to be of
        dimensions (N, graph_degree)
    intermediate_graph_degree : int
        Internally, nn-descent builds an all-neighbors knn graph of dimensions
        (N, intermediate_graph_degree) before selecting the final
        `graph_degree` neighbors. It's recommended that
        `intermediate_graph_degree` >= 1.5 * graph_degree
    max_iterations : int
        The number of iterations that nn-descent will refine the graph for.
        More iterations produce a better quality graph at cost of performance
    termination_threshold : float
        The delta at which nn-descent will terminate its iterations
    return_distances : bool
        Whether or not to include distances in the output.
    """

    cdef cuvsNNDescentIndexParams* params
    cdef object _metric

    def __cinit__(self):
        cuvsNNDescentIndexParamsCreate(&self.params)

    def __dealloc__(self):
        check_cuvs(cuvsNNDescentIndexParamsDestroy(self.params))

    def __init__(self, *,
                 n_lists=1024,
                 metric="sqeuclidean",
                 metric_arg=2.0,
                 graph_degree=64,
                 intermediate_graph_degree=128,
                 max_iterations=20,
                 termination_threshold=0.0001,
                 return_distances=False,
                 n_clusters=1
                 ):
        self._metric = metric
        self.params.metric = <cuvsDistanceType>DISTANCE_TYPES[metric]
        self.params.graph_degree = graph_degree
        self.params.intermediate_graph_degree = intermediate_graph_degree
        self.params.max_iterations = max_iterations
        self.params.termination_threshold = termination_threshold
        self.params.return_distances = return_distances
        self.params.n_clusters = n_clusters

    @property
    def metric(self):
        return self._metric

    @property
    def metric_arg(self):
        return self.params.metric_arg

    @property
    def graph_degree(self):
        return self.params.graph_degree

    @property
    def intermediate_graph_degree(self):
        return self.params.intermediate_graph_degree

    @property
    def max_iterations(self):
        return self.params.max_iterations

    @property
    def termination_threshold(self):
        return self.params.termination_threshold

    @property
    def return_distances(self):
        return self.params.return_distances

    @property
    def n_clusters(self):
        return self.params.n_clusters

cdef class Index:
    """
    NN-Descent index object. This object stores the trained NN-Descent index,
    which can be used to get the NN-Descent graph and distances after
    building
    """

    cdef cuvsNNDescentIndex_t index
    cdef bool trained
    cdef int64_t num_rows
    cdef size_t graph_degree

    def __cinit__(self):
        self.trained = False
        self.num_rows = 0
        self.graph_degree = 0
        check_cuvs(cuvsNNDescentIndexCreate(&self.index))

    def __dealloc__(self):
        check_cuvs(cuvsNNDescentIndexDestroy(self.index))

    @property
    def trained(self):
        return self.trained

    @property
    def graph(self):
        if not self.trained:
            raise ValueError("Index needs to be built before getting graph")

        output = np.empty((self.num_rows, self.graph_degree), dtype='uint32')
        ai = wrap_array(output)
        cdef cydlpack.DLManagedTensor* output_dlpack = cydlpack.dlpack_c(ai)
        check_cuvs(cuvsNNDescentIndexGetGraph(self.index, output_dlpack))
        return output

    def __repr__(self):
        return "Index(type=NNDescent)"


@auto_sync_resources
def build(IndexParams index_params, dataset, graph=None, resources=None):
    """
    Build KNN graph from the dataset

    Parameters
    ----------
    index_params : :py:class:`cuvs.neighbors.nn_descent.IndexParams`
    dataset : Array interface compliant matrix, on either host or device memory
        Supported dtype [float, int8, uint8]
    graph : Optional host matrix for storing output graph
    {resources_docstring}

    Returns
    -------
    index: py:class:`cuvs.neighbors.nn_descent.Index`

    Examples
    --------

    >>> import cupy as cp
    >>> from cuvs.neighbors import nn_descent
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> k = 10
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> build_params = nn_descent.IndexParams(metric="sqeuclidean")
    >>> index = nn_descent.build(build_params, dataset)
    >>> graph = index.graph
    """
    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai, [np.dtype('float32'), np.dtype('byte'),
                                    np.dtype('ubyte')])

    cdef Index idx = Index()
    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)
    cdef cuvsNNDescentIndexParams* params = index_params.params

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef cydlpack.DLManagedTensor* graph_dlpack = NULL
    if graph is not None:
        graph_ai = wrap_array(graph)
        graph_dlpack = cydlpack.dlpack_c(graph_ai)

    with cuda_interruptible():
        check_cuvs(cuvsNNDescentBuild(
            res,
            params,
            dataset_dlpack,
            graph_dlpack,
            idx.index
        ))
        idx.trained = True
        idx.num_rows = dataset_ai.shape[0]
        idx.graph_degree = params.graph_degree

    return idx

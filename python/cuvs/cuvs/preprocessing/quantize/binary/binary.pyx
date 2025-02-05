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

from cuvs.common cimport cydlpack

from pylibraft.common import auto_convert_output, device_ndarray
from pylibraft.common.cai_wrapper import wrap_array

from cuvs.common.exceptions import check_cuvs
from cuvs.common.resources import auto_sync_resources
from cuvs.neighbors.common import _check_input_array


@auto_sync_resources
@auto_convert_output
def transform(dataset, output=None, resources=None):
    """
    Applies binary quantization transform to given dataset

    Parameters
    ----------
    dataset : row major host or device dataset to transform
    output : optional preallocated output memory, on host or device memory
    {resources_docstring}

    Returns
    -------
    output : transformed dataset quantized into a uint8

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.preprocessing.quantize import binary
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.standard_normal((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> transformed = binary.transform(dataset)
    """

    dataset_ai = wrap_array(dataset)

    _check_input_array(dataset_ai,
                       [np.dtype("float32"),
                        np.dtype("float64"),
                        np.dtype("float16")])

    if output is None:
        on_device = hasattr(dataset, "__cuda_array_interface__")
        ndarray = device_ndarray if on_device else np
        cols = int(np.ceil(dataset_ai.shape[1] / 8))
        output = ndarray.empty((dataset_ai.shape[0], cols), dtype="uint8")

    output_ai = wrap_array(output)
    _check_input_array(output_ai, [np.dtype("uint8")])

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)
    cdef cydlpack.DLManagedTensor* output_dlpack = \
        cydlpack.dlpack_c(output_ai)

    check_cuvs(cuvsBinaryQuantizerTransform(res,
                                            dataset_dlpack,
                                            output_dlpack))

    return output

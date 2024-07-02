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

from libc.stdint cimport uintptr_t

from cuvs.common cimport cydlpack

from .filters cimport BITMAP, NO_FILTER, cuvsFilter

from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.neighbors.common import _check_input_array


cdef class Prefilter:
    cdef public cuvsFilter prefilter
    cdef public object parent

    def __init__(self, cuvsFilter prefilter, parent=None):
        if parent is not None:
            self.parent = parent
        self.prefilter = prefilter


def no_filter():
    """
    Create a default pre-filter which filters nothing.
    """
    cdef cuvsFilter filter
    filter.type = NO_FILTER
    filter.addr = <uintptr_t> NULL
    return Prefilter(filter)


def from_bitmap(bitmap):
    """
    Create a pre-filter from an array with type of uint32.

    Parameters
    ----------
    bitmap : numpy.ndarray
        An array with type of `uint32` where each bit in the array corresponds
        to if a sample and query pair is greenlit (not filtered) or filtered.
        The array is row-major, meaning the bits are ordered by rows first.
        Each bit in a `uint32` element represents a different sample-query
        pair.

        - Bit value of 1: The sample-query pair is greenlit (allowed).
        - Bit value of 0: The sample-query pair is filtered.

    Returns
    -------
    filter : cuvs.neighbors.filters.Prefilter
        An instance of `Prefilter` that can be used to filter neighbors
        based on the given bitmap.
    {resources_docstring}

    Examples
    --------

    >>> import cupy as cp
    >>> import numpy as np
    >>> from cuvs.neighbors import filters
    >>>
    >>> n_samples = 50000
    >>> n_queries = 1000
    >>>
    >>> n_bitmap = np.ceil(n_samples * n_queries / 32).astype(int)
    >>> bitmap = cp.random.randint(1, 100, size=(n_bitmap,), dtype=cp.uint32)
    >>> prefilter = filters.from_bitmap(bitmap)
    """
    bitmap_cai = wrap_array(bitmap)
    _check_input_array(bitmap_cai, [np.dtype('uint32')])

    cdef cydlpack.DLManagedTensor* bitmap_dlpack = \
        cydlpack.dlpack_c(bitmap_cai)

    cdef cuvsFilter filter
    filter.type = BITMAP
    filter.addr = <uintptr_t> bitmap_dlpack

    return Prefilter(filter, parent=bitmap)

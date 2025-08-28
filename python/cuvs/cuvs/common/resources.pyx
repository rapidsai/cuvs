#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

import functools

from cuda.bindings.cyruntime cimport cudaStream_t
from libc.stdlib cimport free, malloc

from cuvs.common.c_api cimport (
    cuvsResources_t,
    cuvsResourcesCreate,
    cuvsResourcesDestroy,
    cuvsSNMGResourcesCreate,
    cuvsSNMGResourcesCreateWithDevices,
    cuvsStreamSet,
    cuvsStreamSync,
)

from cuvs.common.exceptions import check_cuvs


cdef class Resources:
    """
    Resources  is a lightweight python wrapper around the corresponding
    C++ class of resources exposed by RAFT's C++ interface. Refer to
    the header file raft/core/resources.hpp for interface level
    details of this struct.

    Parameters
    ----------
    stream : Optional stream to use for ordering CUDA instructions

    Examples
    --------

    Basic usage:

    >>> from cuvs.common import Resources
    >>> handle = Resources()
    >>>
    >>> # call algos here
    >>>
    >>> # final sync of all work launched in the stream of this handle
    >>> handle.sync()

    Using a cuPy stream with cuVS Resources:

    >>> import cupy
    >>> from cuvs.common import Resources
    >>>
    >>> cupy_stream = cupy.cuda.Stream()
    >>> handle = Resources(stream=cupy_stream.ptr)
    """

    def __cinit__(self, stream=None):
        check_cuvs(cuvsResourcesCreate(&self.c_obj))
        if stream:
            check_cuvs(cuvsStreamSet(self.c_obj, <cudaStream_t>stream))

    def sync(self):
        check_cuvs(cuvsStreamSync(self.c_obj))

    def get_c_obj(self):
        """
        Return the pointer to the underlying c_obj as a size_t
        """
        return <size_t> self.c_obj

    def __dealloc__(self):
        check_cuvs(cuvsResourcesDestroy(self.c_obj))

cdef class SNMGResources:
    """
    SNMGResources is a lightweight python wrapper around the corresponding
    C++ SNMG device resources (`raft::device_resources_snmg`) exposed via
    cuVS's C API.

    Parameters
    ----------
    device_ids : Optional sequence[int]
        If provided, an SNMG world will be constructed with only the
        specified device IDs. If omitted, the SNMG world will be constructed
        with all available GPUs.
    """

    cdef cuvsResources_t c_obj

    def __cinit__(self, device_ids=None):
        """
        Construct SNMGResources. If device_ids is provided (any Python
        sequence of ints), use the C API `cuvsSNMGResourcesCreateWithDevices`.
        Otherwise use `cuvsSNMGResourcesCreate`.
        """
        cdef Py_ssize_t num_ids
        cdef int* ids = NULL
        cdef Py_ssize_t i

        if device_ids is None:
            check_cuvs(cuvsSNMGResourcesCreate(&self.c_obj))
        else:
            # Convert Python sequence -> C int array
            seq = device_ids
            # Allow any sequence (list/tuple/np array-like)
            num_ids = len(seq)
            if num_ids == 0:
                # fallback to create default world with all GPUs
                check_cuvs(cuvsSNMGResourcesCreate(&self.c_obj))
                return

            ids = <int*> malloc(num_ids * sizeof(int))
            if ids == NULL:
                raise MemoryError(
                    "unable to allocate temporary device id array"
                )

            try:
                for i in range(num_ids):
                    # use int() to coerce python objects to int; will raise
                    # on bad values
                    ids[i] = <int> int(seq[i])
                check_cuvs(cuvsSNMGResourcesCreateWithDevices(
                    &self.c_obj, ids, <int>num_ids
                ))
            finally:
                free(ids)

    def sync(self):
        """
        Synchronize all device streams for the SNMG resources object.
        """
        check_cuvs(cuvsStreamSync(self.c_obj))

    def get_c_obj(self):
        """
        Return the pointer to the underlying c_obj as a size_t
        """
        return <size_t> self.c_obj

    def __dealloc__(self):
        # Destroy underlying SNMG resources handle
        # If c_obj is 0 / NULL this call should be harmless (but depends on
        # C API)
        try:
            check_cuvs(cuvsResourcesDestroy(self.c_obj))
        except Exception:
            # Suppress exceptions in dealloc to avoid raising during
            # GC/finalize
            pass

_resources_param_string = """
     resources : Optional cuVS Resource handle for reusing CUDA resources.
        If Resources aren't supplied, CUDA resources will be
        allocated inside this function and synchronized before the
        function exits. If resources are supplied, you will need to
        explicitly synchronize yourself by calling `resources.sync()`
        before accessing the output.
""".strip()

_snmg_resources_param_string = """
     resources : Optional SNMG Resource handle (SNMGResources) for multi-GPU.
        If resources aren't supplied, an instance of SNMGResources will be
        constructed for all available GPUs and synchronized before the
        function exits. If resources are supplied, explicitly call
        `resources.sync()` before accessing the output.
""".strip()


def auto_sync_resources(f):
    """Decorator to automatically call sync on a cuVS Resources object when
    it isn't passed to a function.

    When a resources=None is passed to the wrapped function, this decorator
    will automatically create a default resources for the function, and
    call sync on that resources when the function exits.

    This will also insert the appropriate docstring for the resources parameter
    """

    @functools.wraps(f)
    def wrapper(*args, resources=None, **kwargs):
        sync_resources = resources is None
        resources = resources if resources is not None else Resources()

        ret_value = f(*args, resources=resources, **kwargs)

        if sync_resources:
            resources.sync()

        return ret_value

    wrapper.__doc__ = wrapper.__doc__.format(
        resources_docstring=_resources_param_string
    )
    return wrapper


def auto_sync_snmg_resources(f):
    """Decorator to automatically call sync on an SNMGResources object when
    it isn't passed to a function.

    When resources=None is passed to the wrapped function, this decorator
    will automatically create a default SNMGResources for the function, and
    call sync on that resources when the function exits.

    This will also insert the appropriate docstring for the resources parameter
    """
    @functools.wraps(f)
    def wrapper(*args, resources=None, **kwargs):
        sync_resources = resources is None
        resources = resources if resources is not None else SNMGResources()

        ret_value = f(*args, resources=resources, **kwargs)

        if sync_resources:
            resources.sync()

        return ret_value

    wrapper.__doc__ = wrapper.__doc__.format(
        resources_docstring=_snmg_resources_param_string
    )
    return wrapper

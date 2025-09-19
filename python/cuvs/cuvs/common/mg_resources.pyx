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

from cuda.bindings.cyruntime cimport cudaStream_t

from cuvs.common.c_api cimport (
    cuvsMultiGpuResourcesCreate,
    cuvsMultiGpuResourcesCreateWithDeviceIds,
    cuvsMultiGpuResourcesDestroy,
    cuvsResources_t,
    cuvsStreamSet,
    cuvsStreamSync,
)
from cuvs.common.cydlpack cimport DLManagedTensor, dlpack_c

import numpy as np
from pylibraft.common.cai_wrapper import wrap_array

from cuvs.common.exceptions import check_cuvs


cdef class MultiGpuResources:
    """
    Multi-GPU Resources is a lightweight python wrapper around the
    corresponding C++ class of multi-GPU resources exposed by RAFT's C++
    interface. This class provides a handle for multi-GPU operations across
    all available GPUs.

    Parameters
    ----------
    stream : int, optional
        A CUDA stream pointer to use for this resource handle. If None, a
        default stream will be used.
    device_ids : list of int, optional
        A list of device IDs to use for multi-GPU operations. If None, all
        available GPUs will be used.

    Examples
    --------

    Basic usage:

    >>> from cuvs.common import MultiGpuResources
    >>> handle = MultiGpuResources()
    >>>
    >>> # call multi-GPU algos here
    >>>
    >>> # final sync of all work launched in the stream of this handle
    >>> handle.sync()

    Using a cuPy stream with cuVS Multi-GPU Resources:

    >>> import cupy
    >>> from cuvs.common import MultiGpuResources
    >>>
    >>> cupy_stream = cupy.cuda.Stream()
    >>> handle = MultiGpuResources(stream=cupy_stream.ptr)

    Using specific device IDs:

    >>> from cuvs.common import MultiGpuResources
<<<<<<< HEAD
    >>> # Use only GPUs 0 and 2
=======
>>>>>>> branch-25.10
    >>> handle = MultiGpuResources(device_ids=[0])
    >>>
    >>> # call multi-GPU algos here
    >>>
    >>> handle.sync()
    """

    def __cinit__(self, stream=None, device_ids=None):
        cdef DLManagedTensor* device_ids_dlpack = NULL

        if device_ids is not None:
            # Convert device_ids list to DLManagedTensor (keep on host/CPU)
            # NumPy arrays are naturally on the host, which is what we need
            device_ids_array = np.asarray(device_ids, dtype=np.int32)
            ai = wrap_array(device_ids_array)
            device_ids_dlpack = dlpack_c(ai)
            check_cuvs(cuvsMultiGpuResourcesCreateWithDeviceIds(
                &self.c_obj, device_ids_dlpack))
        else:
            check_cuvs(cuvsMultiGpuResourcesCreate(&self.c_obj))

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
        check_cuvs(cuvsMultiGpuResourcesDestroy(self.c_obj))


_multi_gpu_resources_param_string = """
     resources : Optional cuVS Multi-GPU Resource handle for reusing CUDA \
resources.
        If Multi-GPU Resources aren't supplied, CUDA resources will be
        allocated inside this function and synchronized before the
        function exits. If resources are supplied, you will need to
        explicitly synchronize yourself by calling `resources.sync()`
        before accessing the output.
""".strip()


def auto_sync_multi_gpu_resources(f):
    """Decorator to automatically call sync on a cuVS Multi-GPU Resources
    object when it isn't passed to a function.

    When a resources=None is passed to the wrapped function, this decorator
    will automatically create a default multi-GPU resources for the function,
    and call sync on that resources when the function exits.

    This will also insert the appropriate docstring for the resources
    parameter
    """
    import functools

    @functools.wraps(f)
    def wrapper(*args, resources=None, **kwargs):
        sync_resources = resources is None
        resources = (resources if resources is not None
                     else MultiGpuResources())

        ret_value = f(*args, resources=resources, **kwargs)

        if sync_resources:
            resources.sync()

        return ret_value

    wrapper.__doc__ = wrapper.__doc__.format(
        resources_docstring=_multi_gpu_resources_param_string
    )
    return wrapper

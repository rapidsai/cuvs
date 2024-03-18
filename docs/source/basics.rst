cuVS API Basics
===============

- `Memory management`_
- `Resource management`_

Memory management
-----------------

Centralized memory management allows flexible configuration of allocation strategies, such as sharing the same CUDA memory pool across library boundaries. cuVS uses the [RMM](https://github.com/rapidsai/rmm) library, which eases the burden of configuring different allocation strategies globally across GPU-accelerated libraries.

RMM currently has APIs for C++ and Python.

C++
^^^

Here's an example of configuring RMM to use a pool allocator in C++ (derived from the RMM example `here <https://github.com/rapidsai/rmm?tab=readme-ov-file#example>`_):

.. code-block:: c++

    rmm::mr::cuda_memory_resource cuda_mr;
    // Construct a resource that uses a coalescing best-fit pool allocator
    // With the pool initially half of available device memory
    auto initial_size = rmm::percent_of_free_device_memory(50);
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr, initial_size};
    rmm::mr::set_current_device_resource(&pool_mr); // Updates the current device resource pointer to `pool_mr`
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(); // Points to `pool_mr`

Python
^^^^^^

And the corresponding code in Python (derived from the RMM example `here <https://github.com/rapidsai/rmm?tab=readme-ov-file#memoryresource-objects>`_):

.. code-block:: python

    import rmm
    pool = rmm.mr.PoolMemoryResource(
      rmm.mr.CudaMemoryResource(),
      initial_pool_size=2**30,
      maximum_pool_size=2**32)
    rmm.mr.set_current_device_resource(pool)


Resource management
-------------------

cuVS uses an API from the `RAFT <https://github.com/rapidsai/raft>`_ library of ML and data mining primitives to centralize and reuse expensive resources, such as memory management. The below code examples demonstrate how to create these resources for use throughout this guide.

See RAFT's `resource API documentation <https://docs.rapids.ai/api/raft/nightly/cpp_api/core_resources/>`_ for more information.

C
^

.. code-block:: c

    #include <cuda_runtime.h>
    #include <cuvs/core/c_api.h>

    cuvsResources_t res;
    cuvsResourcesCreate(&res);

    // ... do some processing ...

    cuvsResourcesDestroy(res);

C++
^^^

.. code-block:: c++

    #include <raft/core/device_resources.hpp>

    raft::device_resources res;

Python
^^^^^^

.. code-block:: python

    import pylibraft

    res = pylibraft.common.DeviceResources()


Rust
^^^^

.. code-block:: rust

    let res = cuvs::Resources::new()?;
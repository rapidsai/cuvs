# Quick Start

This guide is meant to provide a quick-start primer for using the various different APIs and tools used in the cuVS software development kit. 

## Table of Contents

- [Memory management](#memory-management)
- [Resource management](#resource-management)
- [Interoperability](#multi-dimensional-array-formats-and-interoperability)
  - [DLPack (C)](#dlpack-c)
  - [Multi-dimensional span (C++)](#multi-dimensional-span-c)
  - [CUDA array interface (Python)](#cuda-array-interface-python)
- [Working with ANN indexes](#working-with-ann-indexes)
  - [Building an index](#building-an-index)
  - [Searching an index](#searching-an-index)
  - [CPU/GPU Interoperability](#cpu-gpu-interoperability)
  - [Serializing an index](#serializing-an-index)

------

## Memory management

Centralized memory management allows flexible configuration of allocation strategies, such as sharing the same CUDA memory pool across library boundaries. cuVS uses the [RMM](https://github.com/rapidsai/rmm) library, which eases the burden of configuring different allocation strategies globally across GPU-accelerated libraries. 

RMM currently has APIs for C++ and Python.

### C++

Here's an example of configuring RMM to use a pool allocator in C++ (derived from the RMM example [here](https://github.com/rapidsai/rmm?tab=readme-ov-file#example)):

```c++
rmm::mr::cuda_memory_resource cuda_mr;
// Construct a resource that uses a coalescing best-fit pool allocator
// With the pool initially half of available device memory
auto initial_size = rmm::percent_of_free_device_memory(50);
rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr, initial_size};
rmm::mr::set_current_device_resource(&pool_mr); // Updates the current device resource pointer to `pool_mr`
rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(); // Points to `pool_mr`
```

### Python

And the corresponding code in Python (derived from the RMM example [here](https://github.com/rapidsai/rmm?tab=readme-ov-file#memoryresource-objects)):
```python
import rmm
pool = rmm.mr.PoolMemoryResource(
  rmm.mr.CudaMemoryResource(),
  initial_pool_size=2**30,
  maximum_pool_size=2**32)
rmm.mr.set_current_device_resource(pool)
```


## Resource management

cuVS uses an API from the [RAFT](https://github.com/rapidsai/raft) library of ML and data mining primitives to centralize and reuse expensive resources, such as memory management. The below code examples demonstrate how to create these resources for use throughout this guide.

See RAFT's [resource API documentation](https://docs.rapids.ai/api/raft/nightly/cpp_api/core_resources/) for more information. 

### C

```c
#include <cuda_runtime.h>
#include <cuvs/core/c_api.h>

cuvsResources_t res;
cuvsResourcesCreate(&res);

// ... do some processing ...

cuvsResourcesDestroy(res);
```

### C++

```c++
#include <raft/core/device_resources.hpp>

raft::device_resources res;
```

### Python

```Python
import pylibraft

res = pylibraft.common.DeviceResources()
```

### Rust




## Multi-dimensional array formats and interoperability


### DLPack (C)



### Multi-dimensional span (C++)

cuVS is built on top of the GPU-accelerated machine learning and data mining primitives in the [RAFT](https://github.com/rapidsai/raft) library. Most of the C++ APIs in cuVS accept [mdspan](https://arxiv.org/abs/2010.06474) multi-dimensional array view for representing data in higher dimensions similar to the `ndarray` in the Numpy Python library. RAFT also contains the corresponding owning `mdarray` structure, which simplifies the allocation and management of multi-dimensional data in both host and device (GPU) memory.

The `mdarray` is an owning object that forms a convenience layer over RMM and can be constructed in RAFT using a number of different helper functions:

```c++
#include <raft/core/device_mdarray.hpp>

int n_rows = 10;
int n_cols = 10;

auto scalar = raft::make_device_scalar<float>(handle, 1.0);
auto vector = raft::make_device_vector<float>(handle, n_cols);
auto matrix = raft::make_device_matrix<float>(handle, n_rows, n_cols);
```

The `mdspan` is a lightweight non-owning view that can wrap around any pointer, maintaining shape, layout, and indexing information for accessing elements.

We can construct `mdspan` instances directly from the above `mdarray` instances:

```c++
// Scalar mdspan on device
auto scalar_view = scalar.view();

// Vector mdspan on device
auto vector_view = vector.view();

// Matrix mdspan on device
auto matrix_view = matrix.view();
```
Since the `mdspan` is just a lightweight wrapper, we can also construct it from the underlying data handles in the `mdarray` instances above. We use the extent to get information about the `mdarray` or `mdspan`'s shape.

```c++
#include <raft/core/device_mdspan.hpp>

auto scalar_view = raft::make_device_scalar_view(scalar.data_handle());
auto vector_view = raft::make_device_vector_view(vector.data_handle(), vector.extent(0));
auto matrix_view = raft::make_device_matrix_view(matrix.data_handle(), matrix.extent(0), matrix.extent(1));
```

Of course, RAFT's `mdspan`/`mdarray` APIs aren't just limited to the `device`. You can also create `host` variants:

```c++
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>

int n_rows = 10;
int n_cols = 10;

auto scalar = raft::make_host_scalar<float>(handle, 1.0);
auto vector = raft::make_host_vector<float>(handle, n_cols);
auto matrix = raft::make_host_matrix<float>(handle, n_rows, n_cols);

auto scalar_view = raft::make_host_scalar_view(scalar.data_handle());
auto vector_view = raft::make_host_vector_view(vector.data_handle(), vector.extent(0));
auto matrix_view = raft::make_host_matrix_view(matrix.data_handle(), matrix.extent(0), matrix.extent(1));
```

Please refer to RAFT's `mdspan` [documentation](https://docs.rapids.ai/api/raft/stable/cpp_api/mdspan/) to learn more. 


### CUDA array interface (Python)



## Working with ANN indexes



### Building an index

### Searching an index

### CPU/GPU interoperability

### Serializing an index
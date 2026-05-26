# Dense Arrays

Most NVIDIA cuVS APIs operate on dense vectors and matrices: datasets, queries, labels, distances, neighbors, centroids, and intermediate buffers. Each language binding exposes those arrays in the form that is most natural for that language, while the lower-level NVIDIA cuVS C API uses DLPack-compatible tensor metadata to describe the same shape, dtype, device, and layout. For APIs that explicitly accept sparse inputs, see [Sparse Arrays](/user-guide/api-guides/core-types/array-types/sparse-arrays).

Most APIs expect row-major matrices. A dataset is usually shaped `n_rows x n_features`, a query matrix is shaped `n_queries x n_features`, and top-k outputs are shaped `n_queries x k`.

NVIDIA cuVS uses RMM for arrays allocated in device-accessible memory, including device memory, managed memory, and pinned host memory. See [Memory Management](/user-guide/api-guides/core-types/memory-management) for allocator configuration and memory-resource guidance.

## Common types and interoperability

| Language | Common array type | What to watch |
| --- | --- | --- |
| C | `DLManagedTensor*` | The tensor must describe shape, dtype, device, and layout. The caller owns the memory and lifetime. |
| C++ | RAFT `mdspan` views and `mdarray` containers | C++ APIs usually accept non-owning views. Owning arrays are convenient when C++ should allocate storage. |
| Python | CuPy, NumPy, PyTorch, TensorFlow, DLPack-compatible, or other array-interface objects | GPU APIs commonly expect CUDA array interface inputs; some build or extend paths can accept host arrays. |
| Java | `CuVSMatrix` | Prefer `CuVSMatrix` for matrix-shaped inputs so large datasets can live outside the Java heap. |
| Rust | `ndarray` arrays and `ManagedTensor` | Host arrays can be wrapped and copied to device memory before device-only calls. |
| Go | `cuvs.Tensor[T]` | Tensors wrap DLPack metadata and can be copied between host and device with resource-aware helpers. |

<Note>Use the specific API page to confirm whether a given argument must be on the GPU or can be in host memory. Search inputs and outputs usually need device memory. Build inputs vary by algorithm and language binding.</Note>

### Allocating dense device arrays

The examples below allocate a row-major `n_rows x n_features` `float32` matrix that can be used as a dense NVIDIA cuVS input. Python has several interoperability paths: CuPy arrays can be passed directly, PyTorch and TensorFlow tensors can be shared through DLPack, and NumPy arrays are host arrays that need to be copied to device memory before GPU-only calls.

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>
#include <dlpack/dlpack.h>

#include <stddef.h>
#include <stdint.h>

void allocate_device_matrix(int64_t n_rows, int64_t n_features)
{
  cuvsResources_t res;
  void* data = NULL;
  int64_t shape[2] = {n_rows, n_features};
  size_t bytes = (size_t)n_rows * (size_t)n_features * sizeof(float);

  cuvsResourcesCreate(&res);
  cuvsRMMAlloc(res, &data, bytes);

  DLManagedTensor dataset = {0};
  dataset.dl_tensor.data = data;
  dataset.dl_tensor.device = (DLDevice){kDLCUDA, 0};
  dataset.dl_tensor.ndim = 2;
  dataset.dl_tensor.dtype = (DLDataType){kDLFloat, 32, 1};
  dataset.dl_tensor.shape = shape;
  dataset.dl_tensor.strides = NULL;
  dataset.dl_tensor.byte_offset = 0;

  // Pass &dataset to NVIDIA cuVS C APIs while shape and data remain alive.

  cuvsRMMFree(res, data, bytes);
  cuvsResourcesDestroy(res);
}
```

</Tab>
<Tab title="C++">

```cpp
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>

raft::device_resources res;
int64_t n_rows = 100000;
int64_t n_features = 128;

auto dataset_storage = raft::make_device_matrix<float, int64_t>(
    res, n_rows, n_features);

auto dataset = dataset_storage.view();
```

</Tab>
<Tab title="Python: CuPy">

```python
import cupy as cp

n_rows = 100_000
n_features = 128

dataset = cp.empty((n_rows, n_features), dtype=cp.float32)
```

</Tab>
<Tab title="Python: PyTorch">

```python
import cupy as cp
import torch

n_rows = 100_000
n_features = 128

torch_dataset = torch.empty(
    (n_rows, n_features),
    device="cuda",
    dtype=torch.float32,
)

# Keep torch_dataset alive while the CuPy view is used.
dataset = cp.from_dlpack(torch_dataset)
```

</Tab>
<Tab title="Python: TensorFlow">

```python
import cupy as cp
import tensorflow as tf

n_rows = 100_000
n_features = 128

with tf.device("/GPU:0"):
    tf_dataset = tf.zeros((n_rows, n_features), dtype=tf.float32)

# Keep tf_dataset alive while the CuPy view is used.
dataset = cp.from_dlpack(tf.experimental.dlpack.to_dlpack(tf_dataset))
```

</Tab>
<Tab title="Python: DLPack-compatible">

```python
import cupy as cp

external_dataset = make_dlpack_compatible_cuda_array()

# external_dataset can be an object with __dlpack__ or a one-time
# DLPack capsule produced by another library.
dataset = cp.from_dlpack(external_dataset)
```

</Tab>
<Tab title="Python: NumPy">

```python
import cupy as cp
import numpy as np

n_rows = 100_000
n_features = 128

host_dataset = np.empty((n_rows, n_features), dtype=np.float32)
dataset = cp.asarray(host_dataset)
```

</Tab>
<Tab title="Java">

```java
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.CuVSDeviceMatrix;

long nRows = 100_000;
long nFeatures = 128;

try (CuVSResources resources = CuVSResources.create()) {
  CuVSMatrix.Builder<CuVSDeviceMatrix> builder =
      CuVSMatrix.deviceBuilder(
          resources, nRows, nFeatures, CuVSMatrix.DataType.FLOAT);

  for (long row = 0; row < nRows; row++) {
    builder.addVector(loadVector(row));
  }

  try (CuVSMatrix dataset = builder.build()) {
    // Pass dataset to NVIDIA cuVS Java APIs.
  }
}
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::{ManagedTensor, Resources, Result};

fn allocate_device_matrix(n_rows: usize, n_features: usize) -> Result<ManagedTensor> {
    let res = Resources::new()?;
    let host = ndarray::Array2::<f32>::zeros((n_rows, n_features));

    ManagedTensor::from(&host).to_device(&res)
}
```

</Tab>
<Tab title="Go">

```go
package main

import cuvs "github.com/rapidsai/cuvs/go"

func allocateDeviceMatrix(nRows int64, nFeatures int64) error {
	resource, err := cuvs.NewResource(nil)
	if err != nil {
		return err
	}
	defer resource.Close()

	dataset, err := cuvs.NewTensorOnDevice[float32](
		&resource,
		[]int64{nRows, nFeatures},
	)
	if err != nil {
		return err
	}
	defer dataset.Close()

	// Pass dataset to NVIDIA cuVS Go APIs.
	return nil
}
```

</Tab>
</Tabs>

## Using dense arrays in cuVS APIs

The examples below all pass a two-dimensional dataset into the same kind of NVIDIA cuVS operation. The array container changes by language, but the logical shape is the same: rows are vectors and columns are features.

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/brute_force.h>
#include <dlpack/dlpack.h>

cuvsResources_t res;
cuvsBruteForceIndex_t index;
DLManagedTensor* dataset;

// dataset describes a dense matrix with shape [n_rows, n_features].
load_dataset(dataset);

cuvsResourcesCreate(&res);
cuvsBruteForceIndexCreate(&index);

cuvsBruteForceBuild(res, dataset, L2Expanded, 0.0f, index);

cuvsBruteForceIndexDestroy(index);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>

namespace brute_force = cuvs::neighbors::brute_force;

raft::device_resources res;
float const* dataset_ptr = load_device_dataset();
int64_t n_rows = get_dataset_rows();
int64_t n_features = get_dataset_cols();

auto dataset = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    dataset_ptr, n_rows, n_features);

auto index = brute_force::build(res, brute_force::index_params{}, dataset);
```

</Tab>
<Tab title="Python">

```python
import cupy as cp
from cuvs.neighbors import brute_force

dataset = cp.asarray(load_dataset(), dtype=cp.float32)

index = brute_force.build(dataset, metric="sqeuclidean")
```

</Tab>
<Tab title="Java">

```java
try (CuVSResources resources = CuVSResources.create();
    CuVSMatrix dataset = loadDatasetMatrix()) {
  BruteForceIndexParams indexParams =
      new BruteForceIndexParams.Builder().build();

  try (BruteForceIndex index =
      BruteForceIndex.newBuilder(resources)
          .withDataset(dataset)
          .withIndexParams(indexParams)
          .build()) {
    // Use index for search or serialization.
  }
}
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::brute_force::Index;
use cuvs::distance_type::DistanceType;
use cuvs::{Resources, Result};

fn build_index(dataset: &ndarray::Array2<f32>) -> Result<Index> {
    let res = Resources::new()?;

    Index::build(&res, DistanceType::L2Expanded, None, dataset)
}
```

</Tab>
<Tab title="Go">

```go
package main

import (
	cuvs "github.com/rapidsai/cuvs/go"
	"github.com/rapidsai/cuvs/go/brute_force"
)

func buildIndex(data [][]float32) (*brute_force.BruteForceIndex, error) {
	resource, err := cuvs.NewResource(nil)
	if err != nil {
		return nil, err
	}

	dataset, err := cuvs.NewTensor(data)
	if err != nil {
		return nil, err
	}

	_, err = dataset.ToDevice(&resource)
	if err != nil {
		return nil, err
	}

	index, err := brute_force.CreateIndex()
	if err != nil {
		return nil, err
	}

	err = brute_force.BuildIndex(resource, &dataset, cuvs.DistanceL2, 2.0, index)
	return index, err
}
```

</Tab>
</Tabs>

## Passing outputs

Many NVIDIA cuVS APIs allocate outputs for the caller in higher-level bindings and require explicit output arrays in lower-level bindings. Output arrays should have the expected shape before the API call.

<Tabs>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

namespace brute_force = cuvs::neighbors::brute_force;

raft::device_resources res;
auto queries = load_queries();
auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, n_queries, k);
auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, k);

brute_force::search(
    res,
    brute_force::search_params{},
    index,
    queries,
    neighbors.view(),
    distances.view());
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import brute_force

queries = load_queries()
k = 10

distances, neighbors = brute_force.search(index, queries, k)
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::{ManagedTensor, Resources, Result};

fn search(
    res: &Resources,
    index: &cuvs::brute_force::Index,
    queries: &ndarray::ArrayView2<f32>,
    k: usize,
) -> Result<()> {
    let n_queries = queries.shape()[0];
    let queries = ManagedTensor::from(queries).to_device(res)?;

    let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
    let neighbors = ManagedTensor::from(&neighbors_host).to_device(res)?;

    let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
    let distances = ManagedTensor::from(&distances_host).to_device(res)?;

    index.search(res, &queries, &neighbors, &distances)?;

    neighbors.to_host(res, &mut neighbors_host)?;
    distances.to_host(res, &mut distances_host)?;
    Ok(())
}
```

</Tab>
</Tabs>

## Dense arrays in C++

If you use NVIDIA cuVS from C++, dense inputs and outputs are usually described with RAFT multi-dimensional array types. These types make it clear whether the memory is on the host or device, what shape it has, and whether the NVIDIA cuVS call can write to it.

There are two main families:

- Non-owning views, such as `raft::device_matrix_view`, `raft::device_vector_view`, `raft::host_matrix_view`, `raft::pinned_matrix_view`, `raft::managed_mdspan`, and `raft::span`. These wrap memory owned somewhere else.
- Owning arrays, such as `raft::device_matrix`, `raft::device_vector`, `raft::host_matrix`, `raft::managed_matrix`, and `raft::pinned_matrix`. These allocate and free their own storage.

Use views when your data already exists in memory. Use owning arrays when you want RAFT to allocate storage for an input, output, or staging buffer.

### Creating non-owning views

Use device views when memory already exists on the GPU. Use host views for CPU-resident memory. A view stores a pointer, extents, layout, and accessor, but it does not own the allocation.

<Tabs>
<Tab title="Device">

```cpp
#include <raft/core/device_mdspan.hpp>

#include <cstdint>

void use_existing_device_buffers(float const* dataset_ptr,
                                 int64_t* labels_ptr,
                                 int64_t n_rows,
                                 int64_t n_features)
{
  auto dataset = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
      dataset_ptr, n_rows, n_features);

  auto labels = raft::make_device_vector_view<int64_t, int64_t>(
      labels_ptr, n_rows);

  // dataset is a non-owning read-only matrix view.
  // labels is a non-owning mutable vector view.
}
```

</Tab>
<Tab title="Host">

```cpp
#include <raft/core/host_mdspan.hpp>

#include <cstdint>
#include <vector>

void use_existing_host_buffers(int64_t n_rows, int64_t n_features)
{
  std::vector<float> host_dataset(n_rows * n_features);
  std::vector<int64_t> host_labels(n_rows);

  auto dataset = raft::make_host_matrix_view<float, int64_t, raft::row_major>(
      host_dataset.data(), n_rows, n_features);

  auto labels = raft::make_host_vector_view<int64_t, int64_t>(
      host_labels.data(), n_rows);

  // The RAFT views are valid only while the vectors remain alive.
}
```

</Tab>
</Tabs>

Use `const` element types for read-only inputs. This documents intent and lets C++ APIs reject accidental writes at compile time.

### Creating owning arrays

Use `device_matrix` and `device_vector` when RAFT should allocate GPU memory. Device arrays allocate through the active RMM device resource, so they work with the memory policy described in [Memory Management](/user-guide/api-guides/core-types/memory-management).

<Tabs>
<Tab title="Device">

```cpp
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>

void allocate_device_arrays(raft::device_resources const& res,
                            int64_t n_rows,
                            int64_t n_features,
                            int64_t k)
{
  auto dataset = raft::make_device_matrix<float, int64_t>(
      res, n_rows, n_features);

  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(
      res, n_rows, k);

  auto distances = raft::make_device_matrix<float, int64_t>(
      res, n_rows, k);

  auto dataset_view = dataset.view();
  auto neighbors_view = neighbors.view();
  auto distances_view = distances.view();
}
```

</Tab>
<Tab title="Host">

```cpp
#include <raft/core/host_mdarray.hpp>

#include <cstdint>

void allocate_host_arrays(int64_t n_rows, int64_t n_features)
{
  auto host_dataset = raft::make_host_matrix<float, int64_t>(
      n_rows, n_features);

  auto host_labels = raft::make_host_vector<int64_t, int64_t>(
      n_rows);

  auto dataset_view = host_dataset.view();
  auto labels_view = host_labels.view();
}
```

</Tab>
<Tab title="Pinned Host">

```cpp
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>

void allocate_pinned_arrays(raft::device_resources const& res,
                            int64_t n_rows,
                            int64_t n_features)
{
  auto staging_dataset = raft::make_pinned_matrix<float, int64_t>(
      res, n_rows, n_features);

  auto staging_labels = raft::make_pinned_vector<int64_t, int64_t>(
      res, n_rows);
}
```

</Tab>
</Tabs>

An owning `mdarray` should outlive any view created from it. Passing `dataset.view()` to an API does not transfer ownership.

### Creating one-dimensional spans

Use `raft::span` for simple one-dimensional buffers when the shape does not need matrix or vector metadata. For NVIDIA cuVS public APIs, prefer `device_vector_view` or `host_vector_view` when the memory space matters.

<Tabs>
<Tab title="C++">

```cpp
#include <raft/core/span.hpp>

#include <cstddef>
#include <cstdint>

void normalize_ids(int64_t* ids, std::size_t n_ids)
{
  raft::span<int64_t> id_span(ids, n_ids);

  for (auto& id : id_span) {
    if (id < 0) { id = 0; }
  }
}
```

</Tab>
</Tabs>

`span` is one-dimensional and does not encode row count, column count, layout, or memory space. Use it for lightweight buffer utilities, not for matrix-shaped NVIDIA cuVS inputs.

### How dense arrays work

`mdspan` and `span` are views. They do not allocate memory and do not free memory. They only describe existing memory.

`mdarray` is an owning container. It allocates memory in a specific memory space and releases that memory when the object is destroyed. The `.view()` method returns an `mdspan` that refers to the same allocation.

The most important properties are:

- Memory space: device, host, managed, or pinned host.
- Shape: vector length, matrix rows and columns, or higher-dimensional extents.
- Layout: usually `raft::row_major` for NVIDIA cuVS matrices unless an API explicitly requests another layout.
- Constness: read-only inputs should use `const` element types or `raft::make_const_mdspan()`.
- Lifetime: views must not outlive the memory they describe.

### Choosing C++ array types

| Type | Owns memory? | Typical use |
| --- | --- | --- |
| `raft::device_matrix_view`, `raft::device_vector_view`, `raft::device_scalar_view` | No | GPU inputs and outputs already allocated by RAFT, RMM, CuPy, or another CUDA-aware library. |
| `raft::host_matrix_view`, `raft::host_vector_view`, `raft::host_scalar_view` | No | CPU-resident buffers that are passed into host-side APIs or copied to device arrays. |
| `raft::pinned_matrix_view`, `raft::pinned_vector_view`, `raft::pinned_scalar_view` | No | Existing page-locked host buffers used for transfers or host-device coordination. |
| `raft::managed_mdspan` | No | Existing CUDA Unified Memory allocations that need a non-owning RAFT view. |
| `raft::span` | No | One-dimensional utility buffers where matrix shape and memory-space metadata are unnecessary. |
| `raft::device_matrix`, `raft::device_vector`, `raft::device_scalar` | Yes | Owning GPU allocations for NVIDIA cuVS C++ inputs, outputs, temporary arrays, and indexes. |
| `raft::host_matrix`, `raft::host_vector`, `raft::host_scalar` | Yes | Owning CPU allocations for host data, small results, and CPU-side staging. |
| `raft::managed_matrix`, `raft::managed_vector`, `raft::managed_scalar` | Yes | Owning CUDA Unified Memory allocations that can be accessed from host and device when that trade-off is useful. |
| `raft::pinned_matrix`, `raft::pinned_vector`, `raft::pinned_scalar` | Yes | Owning page-locked host allocations for repeated host-device transfers or host-device coordination. |
| `raft::device_mdarray`, `raft::host_mdarray`, `raft::managed_mdarray`, `raft::pinned_mdarray` | Yes | Generic owning arrays for ranks beyond scalar, vector, and matrix aliases. |

## Using arrays safely

Check the API page for the expected shape, dtype, layout, and memory location before passing an array. Most NVIDIA cuVS matrices are row-major unless the API says otherwise.

Keep the backing allocation alive for as long as a view is used. A view does not own memory, so destroying the original allocation makes the view invalid.

Allocate output arrays with the exact shape requested by the API when the binding requires explicit outputs. For top-k search, that usually means `n_queries x k` arrays for neighbors and distances.

Synchronize appropriately before reading data on the host. Many NVIDIA cuVS operations enqueue GPU work asynchronously on the stream owned by `raft::device_resources`.

Use pinned host arrays when repeated host-device transfers are important. Ordinary host arrays are simpler and are usually the right choice for CPU-only data.

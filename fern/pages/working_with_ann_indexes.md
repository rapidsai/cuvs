# Using cuVS APIs

cuVS provides APIs for C, C++, Python, and Rust. The examples below use CAGRA to show the basic pattern for building an index and searching it from each language.

## C

### Building an index

```c
#include <cuvs/neighbors/cagra.h>

cuvsResources_t res;
cuvsCagraIndexParams_t index_params;
cuvsCagraIndex_t index;

DLManagedTensor *dataset;

// populate tensor with data
load_dataset(dataset);

cuvsResourcesCreate(&res);
cuvsCagraIndexParamsCreate(&index_params);
cuvsCagraIndexCreate(&index);

cuvsCagraBuild(res, index_params, dataset, index);

cuvsCagraIndexDestroy(index);
cuvsCagraIndexParamsDestroy(index_params);
cuvsResourcesDestroy(res);
```

### Searching an index

```c
#include <cuvs/neighbors/cagra.h>

cuvsResources_t res;
cuvsCagraSearchParams_t search_params;
cuvsCagraIndex_t index;

// ... build index ...

DLManagedTensor *queries;

DLManagedTensor *neighbors;
DLManagedTensor *distances;

// populate tensor with data
load_queries(queries);

cuvsResourcesCreate(&res);
cuvsCagraSearchParamsCreate(&search_params);

cuvsCagraSearch(res, search_params, index, queries, neighbors, distances);

cuvsCagraSearchParamsDestroy(search_params);
cuvsCagraIndexDestroy(index);
cuvsResourcesDestroy(res);
```

## C++

### Building an index

```c++
#include <cuvs/neighbors/cagra.hpp>

using namespace cuvs::neighbors;

raft::device_matrix_view<float> dataset = load_dataset();
raft::device_resources res;

cagra::index_params index_params;

auto index = cagra::build(res, index_params, dataset);
```

### Searching an index

```c++
#include <cuvs/neighbors/cagra.hpp>

using namespace cuvs::neighbors;
cagra::index index;

// ... build index ...

raft::device_matrix_view<float> queries = load_queries();
raft::device_matrix_view<uint32_t> neighbors = make_device_matrix_view<uint32_t>(n_queries, k);
raft::device_matrix_view<float> distances = make_device_matrix_view<float>(n_queries, k);
raft::device_resources res;

cagra::search_params search_params;

cagra::search(res, search_params, index, queries, neighbors, distances);
```

## Python

### Building an index

```python
from cuvs.neighbors import cagra

dataset = load_data()
index_params = cagra.IndexParams()

index = cagra.build(index_params, dataset)
```

### Searching an index

```python
from cuvs.neighbors import cagra

queries = load_queries()

search_params = cagra.SearchParams()

# ... build index ...

neighbors, distances = cagra.search(search_params, index, queries, k)
```

## Rust

### Building and searching an index

```rust
use cuvs::cagra::{Index, IndexParams, SearchParams};
use cuvs::core::ManagedTensor;
use cuvs::{Resources, Result};
use ndarray::s;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

/// Example showing how to index and search data with CAGRA
fn cagra_example() -> Result<()> {
    let res = Resources::new()?;

    // Create a new random dataset to index
    let n_datapoints = 65536;
    let n_features = 512;
    let dataset =
        ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

    // Build the CAGRA index
    let build_params = IndexParams::new()?;
    let index = Index::build(&res, &build_params, &dataset)?;

    // Use the first 4 points from the dataset as queries. This checks that
    // each query returns itself as its own nearest neighbor.
    let n_queries = 4;
    let queries = dataset.slice(s![0..n_queries, ..]);

    let k = 10;

    // CAGRA search requires queries and outputs to be in device memory.
    let queries = ManagedTensor::from(&queries).to_device(&res)?;
    let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
    let neighbors = ManagedTensor::from(&neighbors_host).to_device(&res)?;

    let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
    let distances = ManagedTensor::from(&distances_host).to_device(&res)?;

    let search_params = SearchParams::new()?;

    index.search(&res, &search_params, &queries, &neighbors, &distances)?;

    // Copy results back to host memory.
    distances.to_host(&res, &mut distances_host)?;
    neighbors.to_host(&res, &mut neighbors_host)?;

    println!("Neighbors {:?}", neighbors_host);
    println!("Distances {:?}", distances_host);

    Ok(())
}
```

## Filtering vector indexes

cuVS supports different types of filtering depending on the vector index being used. The main method used by vector indexes is pre-filtering, which accounts for filtered vectors before computing closest neighbors and avoids unnecessary distance calculations.

### Bitset

A bitset is an array of bits where each bit can have two possible values: `0` and `1`. In the context of filtering, `0` means the corresponding vector is filtered out and will not be present in the search results.

Bitsets are optimized to use as little memory as possible and are available through RAFT. See the RAFT [bitset API documentation](https://docs.rapids.ai/api/raft/stable/cpp_api/core_bitset/) for more information. When calling a search function of an ANN index, the bitset length should match the number of vectors present in the database.

### Bitmap

A bitmap is based on the same principle as a bitset, but in two dimensions. This allows users to provide a different bitset for each query being searched. See the RAFT [bitmap API documentation](https://docs.rapids.ai/api/raft/stable/cpp_api/core_bitmap/) for more information.

### Using a Bitset filter on a CAGRA index

```c++
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/core/bitset.hpp>

using namespace cuvs::neighbors;
cagra::index index;

// ... build index ...

cagra::search_params search_params;
raft::device_resources res;
raft::device_matrix_view<float> queries = load_queries();
raft::device_matrix_view<uint32_t> neighbors = make_device_matrix_view<uint32_t>(n_queries, k);
raft::device_matrix_view<float> distances = make_device_matrix_view<float>(n_queries, k);

// Load a list of all the samples that will get filtered.
std::vector<uint32_t> removed_indices_host = get_invalid_indices();
auto removed_indices_device =
      raft::make_device_vector<uint32_t, uint32_t>(res, removed_indices_host.size());

// Copy this list to device.
raft::copy(removed_indices_device.data_handle(), removed_indices_host.data(),
           removed_indices_host.size(), raft::resource::get_cuda_stream(res));

// Create a bitset with the list of samples to filter.
cuvs::core::bitset<uint32_t, uint32_t> removed_indices_bitset(
    res, removed_indices_device.view(), index.size());

// Use a `bitset_filter` in the `cagra::search` function call.
auto bitset_filter =
      cuvs::neighbors::filtering::bitset_filter(removed_indices_bitset.view());
cagra::search(res,
              search_params,
              index,
              queries,
              neighbors,
              distances,
              bitset_filter);
```

### Using a Bitmap filter on a Brute-force index

```c++
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/core/bitmap.hpp>

using namespace cuvs::neighbors;
using indexing_dtype = int64_t;

// ... build index ...
brute_force::index_params index_params;
brute_force::search_params search_params;
raft::device_resources res;
raft::device_matrix_view<float, indexing_dtype> dataset = load_dataset(n_vectors, dim);
raft::device_matrix_view<float, indexing_dtype> queries = load_queries(n_queries, dim);
auto index = brute_force::build(res, index_params, raft::make_const_mdspan(dataset.view()));

// Load a list of all the samples that will get filtered.
std::vector<uint32_t> removed_indices_host = get_invalid_indices();
auto removed_indices_device =
      raft::make_device_vector<uint32_t, uint32_t>(res, removed_indices_host.size());

// Copy this list to device.
raft::copy(removed_indices_device.data_handle(), removed_indices_host.data(),
           removed_indices_host.size(), raft::resource::get_cuda_stream(res));

// Create a bitmap with the list of samples to filter.
cuvs::core::bitset<uint32_t, indexing_dtype> removed_indices_bitset(
  res, removed_indices_device.view(), n_queries * n_vectors);
cuvs::core::bitmap_view<const uint32_t, indexing_dtype> removed_indices_bitmap(
    removed_indices_bitset.data(), n_queries, n_vectors);

// Use a `bitmap_filter` in the `brute_force::search` function call.
auto bitmap_filter =
      cuvs::neighbors::filtering::bitmap_filter(removed_indices_bitmap);

auto neighbors = raft::make_device_matrix_view<uint32_t, indexing_dtype>(n_queries, k);
auto distances = raft::make_device_matrix_view<float, indexing_dtype>(n_queries, k);
brute_force::search(res,
                    search_params,
                    index,
                    raft::make_const_mdspan(queries.view()),
                    neighbors.view(),
                    distances.view(),
                    bitmap_filter);
```

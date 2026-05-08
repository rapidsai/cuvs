# Brute-force

Brute-force, or a flat index, is the simplest nearest-neighbor index. It stores the dataset and checks every vector during search.

Brute-force works well when exact results matter, the dataset is small enough to scan, or filters remove most of the dataset before search.

## Example API Usage

[C API](/api-reference/c-api-neighbors-brute-force) | [C++ API](/api-reference/cpp-api-neighbors-brute-force) | [Python API](/api-reference/python-api-neighbors-brute-force) | [Java API](/api-reference/java-api-com-nvidia-cuvs-bruteforceindex) | [Rust API](/api-reference/rust-api-cuvs-brute-force) | [Go API](/api-reference/go-api-brute-force)

### Building an index

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/brute_force.h>

cuvsResources_t res;
cuvsBruteForceIndex_t index;
DLManagedTensor *dataset;

load_dataset(dataset);

cuvsResourcesCreate(&res);
cuvsBruteForceIndexCreate(&index);

cuvsBruteForceBuild(res, dataset, L2Expanded, 0.0f, index);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/brute_force.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
auto dataset = load_dataset();
brute_force::index_params index_params;

index_params.metric = cuvs::distance::DistanceType::L2Expanded;

auto index = brute_force::build(res, index_params, dataset);
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import brute_force

dataset = load_data()

index = brute_force.build(dataset, metric="sqeuclidean")
```

</Tab>
<Tab title="Java">

```java
try (CuVSResources resources = CuVSResources.create()) {
  BruteForceIndexParams indexParams =
      new BruteForceIndexParams.Builder()
          .withNumWriterThreads(32)
          .build();

  BruteForceIndex index =
      BruteForceIndex.newBuilder(resources)
          .withDataset(vectors)
          .withIndexParams(indexParams)
          .build();
}
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::brute_force::Index;
use cuvs::distance_type::DistanceType;
use cuvs::{Resources, Result};

fn build_brute_force_index(dataset: &ndarray::Array2<f32>) -> Result<Index> {
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

func buildBruteForceIndex(dataset cuvs.Tensor[float32]) (*brute_force.BruteForceIndex, error) {
	resource, err := cuvs.NewResource(nil)
	if err != nil {
		return nil, err
	}

	index, err := brute_force.CreateIndex()
	if err != nil {
		return nil, err
	}

	_, err = dataset.ToDevice(&resource)
	if err != nil {
		return nil, err
	}

	err = brute_force.BuildIndex(resource, &dataset, cuvs.DistanceL2, 2.0, index)
	return index, err
}
```

</Tab>
</Tabs>

### Searching an index

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/brute_force.h>

cuvsResources_t res;
cuvsBruteForceIndex_t index;
DLManagedTensor *queries;
DLManagedTensor *neighbors;
DLManagedTensor *distances;

load_queries(queries);
allocate_outputs(neighbors, distances);

cuvsResourcesCreate(&res);

// ... build or load index ...
cuvsFilter prefilter = {0, NO_FILTER};
cuvsBruteForceSearch(res, index, queries, neighbors, distances, prefilter);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/brute_force.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
brute_force::index<float, float> index(res);
auto queries = load_queries();
auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, n_queries, k);
auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, k);
brute_force::search_params search_params;

// ... build or load index ...
brute_force::search(
    res,
    search_params,
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

# ... build or load index ...
distances, neighbors = brute_force.search(index, queries, k)
```

</Tab>
<Tab title="Java">

```java
BruteForceQuery query =
    new BruteForceQuery.Builder(resources)
        .withTopK(10)
        .withQueryVectors(queries)
        .build();

// ... build or load index ...
SearchResults results = index.search(query);
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::{ManagedTensor, Resources, Result};

fn search_brute_force_index(
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

    distances.to_host(res, &mut distances_host)?;
    neighbors.to_host(res, &mut neighbors_host)?;

    Ok(())
}
```

</Tab>
<Tab title="Go">

```go
_, err := queries.ToDevice(&resource)
if err != nil {
	return err
}

err = brute_force.SearchIndex(resource, *index, &queries, &neighbors, &distances)
if err != nil {
	return err
}

_, err = neighbors.ToHost(&resource)
if err != nil {
	return err
}

_, err = distances.ToHost(&resource)
return err
```

</Tab>
</Tabs>

## How Brute-force works

Brute-force stores the vectors and compares each query against every vector in the dataset. There is no partitioning step, graph traversal, or approximation step.

The build step is mostly a data setup step. The search step does the real work: for each query, cuVS computes distances to every vector, keeps the best `k` candidates, and returns exact nearest neighbors.

## When to use Brute-force

Use brute-force when exact nearest neighbors are required.

Use brute-force when the dataset fits comfortably in device memory and search latency is acceptable.

Use brute-force as a ground-truth baseline for tuning approximate indexes.

Use brute-force for heavily filtered queries. If a filter excludes most vectors, brute-force can search only the remaining candidates while IVF and graph indexes may skip useful vectors before the filter is applied.

Use an approximate index instead when the dataset is large enough that scanning every vector is too slow.

## Using Filters

Brute-force supports filtered search. A filter tells the search which vectors are allowed before distances are computed.

This is useful when queries should only search a small subset of the dataset. For example, if a filter removes 90%-99% of the vectors, brute-force may do much less work while still returning exact results within the allowed subset.

Unlike partitioned or graph-based indexes, brute-force does not need to guess which part of the index might contain the answer. If a vector passes the filter, it can be considered.

## Configuration parameters

### Build parameters

| Name | Default | Description |
| --- | --- | --- |
| `metric` | `L2Expanded` / `sqeuclidean` | Distance metric used for search. |
| `metric_arg` | `2.0` | Extra argument for metrics that need one, such as Minkowski distance. |
| `numWriterThreads` | `2` | Java build option for the number of writer threads. |

### Search parameters

| Name | Default | Description |
| --- | --- | --- |
| `prefilter` | None | Optional bitmap or bitset filter that restricts which vectors can be returned. |

## Tuning Considerations

Brute-force has very few tuning knobs. The main decisions are distance metric, query batch size, output `k`, and whether a filter can reduce the amount of work.

Brute-force is exact, but ties can make result order look different across runs. If many vectors have the same distance near the cutoff, increasing `k` can make comparisons against ground truth more stable.

For large batches, memory allocation can affect performance. Reuse output buffers and memory resources when the API supports it.

## Memory footprint

`precision` is the number of bytes in each vector element, such as 4 bytes for `float32`.

### Index footprint

Raw vectors: $n_{\text{vectors}} \times n_{\text{dimensions}} \times \text{precision}$

Vector norms, for distance metrics that use them: $n_{\text{vectors}} \times \text{distance precision}$

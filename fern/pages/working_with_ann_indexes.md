# Using cuVS APIs

cuVS provides APIs for C, C++, Python, Java, Rust, and Go. For CAGRA build and search examples in each language, see the [CAGRA indexing guide](neighbors/cagra.md).

## Filtering vector indexes

cuVS supports different types of filtering depending on the vector index being used. The main method used by vector indexes is pre-filtering, which accounts for filtered vectors before computing closest neighbors and avoids unnecessary distance calculations.

### Bitset

A bitset is an array of bits where each bit can have two possible values: `0` and `1`. In the context of filtering, `0` means the corresponding vector is filtered out and will not be present in the search results.

Bitsets are optimized to use as little memory as possible and are available through RAFT. See the RAFT [bitset API documentation](https://docs.rapids.ai/api/raft/stable/cpp_api/core_bitset/) for more information. When calling a search function of an ANN index, the bitset length should match the number of vectors present in the database.

For CAGRA bitset examples, see [Using Filters](neighbors/cagra.md#using-filters).

### Bitmap

A bitmap is based on the same principle as a bitset, but in two dimensions. This allows users to provide a different bitset for each query being searched. See the RAFT [bitmap API documentation](https://docs.rapids.ai/api/raft/stable/cpp_api/core_bitmap/) for more information.

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

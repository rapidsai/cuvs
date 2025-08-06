/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuvs/neighbors/index_base.hpp>
#include <memory>

namespace cuvs::neighbors {

/**
 * @brief Intermediate wrapper layer for handling transition between new and old index designs.
 *
 * This class provides a common interface for wrapping various index implementations
 * and handles compatibility issues between different algorithm implementations.
 * It serves as an adapter layer that:
 * - Provides default implementations for common functionality
 * - Handles parameter conversion between different index types
 * - Manages post-processing of search results
 * - Facilitates gradual migration from old to new index designs
 *
 * The primary purpose of this wrapper is to help existing index implementations
 * transition from their original designs to the new object-oriented polymorphic
 * design based on IndexBase. This allows:
 * - Legacy index implementations to work with the new polymorphic interface
 * - Gradual refactoring of existing code without breaking changes
 * - Unified access patterns across different index types
 * - Smooth migration path for users adopting the new architecture
 *
 * By using this wrapper, existing index types (like CAGRA, IVF-PQ, IVF-Flat, HNSW)
 * can be adapted to work with the IndexBase interface without requiring immediate
 * complete rewrites of their internal implementations.
 *
 * @tparam T Data element type (e.g., float, int8, uint8)
 * @tparam IdxT Index type for vector indices
 * @tparam OutputIdxT Output index type, defaults to IdxT
 */
template <typename T, typename IdxT, typename OutputIdxT = IdxT>
class IndexWrapper : public IndexBase<T, IdxT, OutputIdxT> {
 public:
  /** Type definitions inherited from base class */
  using value_type        = typename IndexBase<T, IdxT, OutputIdxT>::value_type;
  using index_type        = typename IndexBase<T, IdxT, OutputIdxT>::index_type;
  using out_index_type    = typename IndexBase<T, IdxT, OutputIdxT>::out_index_type;
  using matrix_index_type = typename IndexBase<T, IdxT, OutputIdxT>::matrix_index_type;

  /** Virtual destructor to enable proper cleanup of derived classes */
  virtual ~IndexWrapper() = default;

  /**
   * @brief Merge this index with other indices (optional functionality).
   *
   * This interface provides polymorphic merge capability for index types that support merging.
   * The merge strategy and parameters are determined by the specific merge_params implementation.
   * Not all index types need to support merging, so this has a default implementation that
   * throws an error.
   *
   * @param[in] handle RAFT resources for executing operations
   * @param[in] params Merge parameters containing strategy and algorithm-specific settings
   * @param[in] other_indices Vector of other indices to merge with this one
   * @return Shared pointer to merged index
   */
  virtual std::shared_ptr<IndexBase<value_type, index_type, out_index_type>> merge(
    const raft::resources& handle,
    const cuvs::neighbors::merge_params& params,
    const std::vector<std::shared_ptr<IndexBase<value_type, index_type, out_index_type>>>&
      other_indices) const
  {
    // Default implementation: not supported
    RAFT_FAIL("Merge operation not supported for this index type");
  }

 protected:
  /**
   * @brief Helper method for derived classes to handle parameter conversion.
   *
   * Derived classes can override this to provide custom parameter handling
   * for their specific index types. The default implementation returns
   * the parameters unchanged.
   *
   * @param[in] params Search parameters to convert
   * @return Converted search parameters
   */
  virtual const search_params& convert_search_params(const search_params& params) const
  {
    return params;
  }
};

/**
 * @brief Migrating Existing Algorithms to New Polymorphic Index Architecture
 *
 * To migrate an existing index algorithm (e.g., IVF-PQ, IVF-Flat, HNSW) to the new polymorphic
 * IndexBase architecture, follow these steps:
 *
 * 1. **Create algorithm-specific wrapper header**:
 *    - Create `cpp/include/cuvs/neighbors/<algorithm>_index_wrapper.hpp`
 *    - Define `<algorithm>::IndexWrapper` class inheriting from `cuvs::neighbors::IndexWrapper`
 *    - Place it in the `cuvs::neighbors::<algorithm>` namespace
 *    - Provide a `make_index_wrapper()` factory function
 *
 * 2. **Create algorithm-specific wrapper implementation**:
 *    - Create `cpp/src/neighbors/<algorithm>_index_wrapper.cu`
 *    - Implement the wrapper methods (search, size, metric, merge if supported)
 *    - Bridge existing algorithm API to new IndexBase interface
 *    - Add explicit template instantiations for supported data types
 *
 * 3. **Include wrapper in main algorithm header**:
 *    - Add `#include <cuvs/neighbors/<algorithm>_index_wrapper.hpp>` at the end of
 *      `cpp/include/cuvs/neighbors/<algorithm>.hpp`
 *
 * 4. **Update composite merge support** (if merge is supported):
 *    - Update `cpp/src/neighbors/composite/merge.cpp` to handle the algorithm
 *
 * Example structure for algorithm "ivf_pq":
 * ```
 * cpp/include/cuvs/neighbors/ivf_pq_index_wrapper.hpp:
 *   namespace cuvs::neighbors::ivf_pq {
 *     template<typename T, typename IdxT, typename OutputIdxT = IdxT>
 *     class IndexWrapper : public cuvs::neighbors::IndexWrapper<T, IdxT, OutputIdxT> {
 *       // Bridge existing ivf_pq::index to new interface
 *     };
 *
 *     template<typename T, typename IdxT, typename OutputIdxT = IdxT>
 *     auto make_index_wrapper(ivf_pq::index<T, IdxT>* index)
 *       -> std::shared_ptr<cuvs::neighbors::IndexBase<T, IdxT, OutputIdxT>>;
 *   }
 *
 * cpp/src/neighbors/ivf_pq_index_wrapper.cu:
 *   // Implementation bridging old API to new interface + explicit instantiations
 *
 * cpp/include/cuvs/neighbors/ivf_pq.hpp:
 *   // ... existing ivf_pq algorithm code ...
 *   }  // namespace cuvs::neighbors::ivf_pq
 *
 *   #include <cuvs/neighbors/ivf_pq_index_wrapper.hpp>
 * ```
 *
 * Usage example (following CAGRA pattern for same-algorithm composite):
 * ```cpp
 * // 1. Build multiple CAGRA indices on different data partitions
 * auto dataset1 = raft::make_device_matrix<float, int64_t>(res, size1, dim);
 * auto dataset2 = raft::make_device_matrix<float, int64_t>(res, size2, dim);
 * auto dataset3 = raft::make_device_matrix<float, int64_t>(res, size3, dim);
 *
 * auto cagra_index1 = cuvs::neighbors::cagra::build(res, params, dataset1.view());
 * auto cagra_index2 = cuvs::neighbors::cagra::build(res, params, dataset2.view());
 * auto cagra_index3 = cuvs::neighbors::cagra::build(res, params, dataset3.view());
 *
 * // 2. Wrap each index for polymorphic usage
 * auto wrapped_index1 = cuvs::neighbors::cagra::make_index_wrapper(&cagra_index1);
 * auto wrapped_index2 = cuvs::neighbors::cagra::make_index_wrapper(&cagra_index2);
 * auto wrapped_index3 = cuvs::neighbors::cagra::make_index_wrapper(&cagra_index3);
 *
 * // 3. Merge indices using the composite merge function
 * std::vector<std::shared_ptr<cuvs::neighbors::IndexWrapper<float, uint32_t>>> indices;
 * indices.push_back(wrapped_index1);
 * indices.push_back(wrapped_index2);
 * indices.push_back(wrapped_index3);
 *
 * cuvs::neighbors::cagra::merge_params merge_params;
 * auto merged_index = cuvs::neighbors::composite::merge(res, merge_params, indices);
 *
 * // 4. Search using the merged index
 * auto queries = raft::make_device_matrix<float, int64_t>(res, n_queries, dim);
 * auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, k);
 * auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, k);
 *
 * cuvs::neighbors::cagra::search_params search_params;
 * merged_index->search(res, search_params, queries.view(), neighbors.view(), distances.view());
 *
 * // The merge function automatically:
 * // - Merges all 3 CAGRA indices based on the merge strategy
 * // - Returns a unified index that can search across all partitions
 * // - Handles global addressing and result merging internally
 * ```
 *
 * Migration benefits:
 * - Enables distributed/partitioned indexing with same design
 * - Maintains full backward compatibility with existing APIs
 * - Allows gradual transition to new architecture
 * - Enables composite index functionality for data partitioning
 * - Provides unified search interface across multiple index instances
 */

}  // namespace cuvs::neighbors

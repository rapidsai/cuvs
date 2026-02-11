/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <limits>

namespace cuvs::neighbors::nn_descent::detail {

using dist_data_t = float;
constexpr int kDegreeOnDevice{32};
constexpr int kSegmentSize{32};
constexpr int kCounterInterval{100};
template <typename index_t>
struct InternalID_t;

inline auto round_up_32(size_t num) -> size_t { return (num + 31) / 32 * 32; }

// InternalID_t uses 1 bit for marking (new or old).
template <>
struct InternalID_t<int> {
 private:
  using index_t = int;
  index_t id_{std::numeric_limits<index_t>::max()};

 public:
  [[nodiscard]] inline _RAFT_HOST_DEVICE auto is_new() const -> bool { return id_ >= 0; }
  inline _RAFT_HOST_DEVICE auto id_with_flag() -> index_t& { return id_; }
  [[nodiscard]] inline _RAFT_HOST_DEVICE auto id() const -> index_t
  {
    if (is_new()) return id_;
    return -id_ - 1;
  }
  inline _RAFT_HOST_DEVICE void mark_old()
  {
    if (id_ >= 0) id_ = -id_ - 1;
  }
  inline _RAFT_HOST_DEVICE auto operator==(const InternalID_t<int>& other) const -> bool
  {
    return id() == other.id();
  }
};

struct build_config {
  size_t max_dataset_size;
  size_t dataset_dim;
  size_t node_degree{64};
  size_t internal_node_degree{0};
  // If internal_node_degree == 0, the value of node_degree will be assigned to it
  size_t max_iterations{50};
  float termination_threshold{0.0001};
  size_t output_graph_degree{32};
  cuvs::distance::DistanceType metric{cuvs::distance::DistanceType::L2Expanded};
  cuvs::neighbors::nn_descent::DIST_COMP_DTYPE dist_comp_dtype{
    cuvs::neighbors::nn_descent::DIST_COMP_DTYPE::AUTO};
};

template <typename index_t>
class bloom_filter {
 public:
  bloom_filter(size_t nrow, size_t num_sets_per_list, size_t num_hashs)
    : nrow_(nrow),
      num_sets_per_list_(num_sets_per_list),
      num_hashs_(num_hashs),
      bitsets_(nrow * kNumBitsPerSet * num_sets_per_list)
  {
  }

  void add(size_t list_id, index_t key)
  {
    if (is_cleared_) { is_cleared_ = false; }
    uint32_t hash = hash_0(key);
    size_t global_set_idx =
      list_id * kNumBitsPerSet * num_sets_per_list_ + key % num_sets_per_list_ * kNumBitsPerSet;
    bitsets_[global_set_idx + hash % kNumBitsPerSet] = 1;
    for (size_t i = 1; i < num_hashs_; i++) {
      hash                                             = hash + hash_1(key);
      bitsets_[global_set_idx + hash % kNumBitsPerSet] = 1;
    }
  }

  void set_nrow(size_t nrow) { nrow_ = nrow; }

  auto check(size_t list_id, index_t key) -> bool
  {
    bool is_present = true;
    uint32_t hash   = hash_0(key);
    size_t global_set_idx =
      list_id * kNumBitsPerSet * num_sets_per_list_ + key % num_sets_per_list_ * kNumBitsPerSet;
    is_present &= bitsets_[global_set_idx + hash % kNumBitsPerSet];

    if (!is_present) return false;
    for (size_t i = 1; i < num_hashs_; i++) {
      hash = hash + hash_1(key);
      is_present &= bitsets_[global_set_idx + hash % kNumBitsPerSet];
      if (!is_present) return false;
    }
    return true;
  }

  void clear()
  {
    if (is_cleared_) return;
#pragma omp parallel for
    for (size_t i = 0; i < nrow_ * kNumBitsPerSet * num_sets_per_list_; i++) {
      bitsets_[i] = 0;
    }
    is_cleared_ = true;
  }

 private:
  auto hash_0(uint32_t value) -> uint32_t
  {
    value *= 1103515245;
    value += 12345;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
  }

  auto hash_1(uint32_t value) -> uint32_t
  {
    value *= 1664525;
    value += 1013904223;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
  }

  static constexpr int kNumBitsPerSet = 512;
  bool is_cleared_{true};
  std::vector<bool> bitsets_;
  size_t nrow_;
  size_t num_sets_per_list_;
  size_t num_hashs_;
};

template <typename index_t>
struct gnnd_graph {
  raft::resources const& res;
  static constexpr int kSegmentSize = 32;
  InternalID_t<index_t>* h_graph;

  size_t nrow;
  size_t node_degree;
  int num_samples;
  int num_segments;

  raft::host_matrix<dist_data_t, size_t, raft::row_major> h_dists;

  raft::pinned_matrix<index_t, size_t> h_graph_new;
  raft::pinned_vector<int2, size_t> h_list_sizes_new;

  raft::pinned_matrix<index_t, size_t> h_graph_old;
  raft::pinned_vector<int2, size_t> h_list_sizes_old;
  bloom_filter<index_t> bloom_filter;

  gnnd_graph(const gnnd_graph&)                    = delete;
  auto operator=(const gnnd_graph&) -> gnnd_graph& = delete;
  gnnd_graph(raft::resources const& res,
             const size_t nrow,
             const size_t node_degree,
             const size_t internal_node_degree,
             const size_t num_samples);
  void init_random_graph();
  // TODO(snanditale): Create a generic bloom filter utility
  // https://github.com/rapidsai/raft/issues/1827 Use Bloom filter to sample "new" neighbors for
  // local joining
  void sample_graph_new(InternalID_t<index_t>* new_neighbors, const size_t width);
  void sample_graph(bool sample_new);
  void update_graph(const InternalID_t<index_t>* new_neighbors,
                    const dist_data_t* new_dists,
                    const size_t width,
                    std::atomic<int64_t>& update_counter);
  void sort_lists();
  void clear();
  ~gnnd_graph();
};

template <typename DataT = float, typename index_t = int>
class gnnd {
 public:
  gnnd(raft::resources const& res, const build_config& build_config);
  gnnd(const gnnd&)                    = delete;
  auto operator=(const gnnd&) -> gnnd& = delete;

  template <typename DistEpilogueT = raft::identity_op>
  void build(DataT* data,
             const index_t nrow,
             index_t* output_graph,
             bool return_distances,
             dist_data_t* output_distances,
             DistEpilogueT dist_epilogue = DistEpilogueT{});
  ~gnnd()    = default;
  using id_t = InternalID_t<index_t>;
  void reset(raft::resources const& res);

 private:
  void add_reverse_edges(index_t* graph_ptr,
                         index_t* h_rev_graph_ptr,
                         index_t* d_rev_graph_ptr,
                         int2* list_sizes,
                         cudaStream_t stream = nullptr);

  template <typename DistEpilogueT = raft::identity_op>
  void local_join(cudaStream_t stream = nullptr, DistEpilogueT dist_epilogue = DistEpilogueT{});

  raft::resources const& res_;

  build_config build_config_;
  gnnd_graph<index_t> graph_;
  std::atomic<int64_t> update_counter_;

  size_t nrow_;
  size_t ndim_;

  std::optional<raft::device_matrix<float, size_t, raft::row_major>> d_data_float_;
  std::optional<raft::device_matrix<half, size_t, raft::row_major>> d_data_half_;
  raft::device_vector<dist_data_t, size_t> l2_norms_;

  raft::device_matrix<id_t, size_t, raft::row_major> graph_buffer_;
  raft::device_matrix<dist_data_t, size_t, raft::row_major> dists_buffer_;

  raft::pinned_matrix<id_t, size_t> graph_host_buffer_;
  raft::pinned_matrix<dist_data_t, size_t> dists_host_buffer_;

  raft::device_vector<int, size_t> d_locks_;

  raft::pinned_matrix<index_t, size_t> h_rev_graph_new_;
  raft::pinned_matrix<index_t, size_t> h_graph_old_;
  raft::pinned_matrix<index_t, size_t> h_rev_graph_old_;
  // int2.x is the number of forward edges, int2.y is the number of reverse edges

  raft::device_vector<int2, size_t> d_list_sizes_new_;
  raft::device_vector<int2, size_t> d_list_sizes_old_;
};

inline auto get_build_config(raft::resources const& res,
                             const index_params& params,
                             size_t num_rows,
                             size_t num_cols,
                             const cuvs::distance::DistanceType metric,
                             size_t& extended_graph_degree,
                             size_t& graph_degree) -> build_config
{
  RAFT_EXPECTS(num_rows < std::numeric_limits<int>::max() - 1,
               "The dataset size for gnnd should be less than %d",
               std::numeric_limits<int>::max() - 1);
  auto allowed_metrics = params.metric == cuvs::distance::DistanceType::L2Expanded ||
                         params.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                         params.metric == cuvs::distance::DistanceType::CosineExpanded ||
                         params.metric == cuvs::distance::DistanceType::InnerProduct ||
                         params.metric == cuvs::distance::DistanceType::BitwiseHamming;
  RAFT_EXPECTS(allowed_metrics,
               "The metric for NN Descent should be L2Expanded, L2SqrtExpanded, CosineExpanded, "
               "InnerProduct or BitwiseHamming");
  RAFT_EXPECTS(
    metric == params.metric,
    "The metrics set in nn_descent::index_params and nn_descent::index are inconsistent");
  size_t intermediate_degree = params.intermediate_graph_degree;
  graph_degree               = params.graph_degree;

  if (intermediate_degree >= num_rows) {
    RAFT_LOG_WARN(
      "Intermediate graph degree cannot be larger than number of rows in dataset, reducing it to "
      "%lu",
      num_rows - 1);
    intermediate_degree = num_rows - 1;
  }
  if (intermediate_degree < graph_degree) {
    RAFT_LOG_WARN(
      "Graph degree (%lu) cannot be larger than intermediate graph degree (%lu), reducing "
      "graph_degree.",
      graph_degree,
      intermediate_degree);
    graph_degree = intermediate_degree;
  }

  // The elements in each knn-list are partitioned into different buckets, and we need more buckets
  // to mitigate bucket collisions. `intermediate_degree` is OK to larger than
  // extended_graph_degree.
  extended_graph_degree =
    round_up_32(static_cast<size_t>(graph_degree * (graph_degree <= 32 ? 1.0 : 1.3)));
  size_t extended_intermediate_degree =
    round_up_32(static_cast<size_t>(intermediate_degree * (intermediate_degree <= 32 ? 1.0 : 1.3)));

  build_config build_config{.max_dataset_size      = num_rows,
                            .dataset_dim           = num_cols,
                            .node_degree           = extended_graph_degree,
                            .internal_node_degree  = extended_intermediate_degree,
                            .max_iterations        = params.max_iterations,
                            .termination_threshold = params.termination_threshold,
                            .output_graph_degree   = params.graph_degree,
                            .metric                = params.metric,
                            .dist_comp_dtype       = params.dist_comp_dtype};
  return build_config;
}

}  // namespace cuvs::neighbors::nn_descent::detail

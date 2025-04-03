#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/nn_descent.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/pow2_utils.cuh>

#include <cuda_runtime.h>

#include <mma.h>

#include <limits>

namespace cuvs::neighbors::nn_descent::detail {

using DistData_t = float;
using align32    = raft::Pow2<32>;

template <typename Index_t>
struct InternalID_t;

template <>
class InternalID_t<int> {
 private:
  using Index_t = int;
  Index_t id_{std::numeric_limits<Index_t>::max()};

 public:
  __host__ __device__ bool is_new() const { return id_ >= 0; }
  __host__ __device__ Index_t& id_with_flag() { return id_; }
  __host__ __device__ Index_t id() const { return is_new() ? id_ : -id_ - 1; }
  __host__ __device__ void mark_old()
  {
    if (id_ >= 0) id_ = -id_ - 1;
  }
  __host__ __device__ bool operator==(const InternalID_t<int>& other) const
  {
    return id() == other.id();
  }
};

struct BuildConfig {
  size_t max_dataset_size;
  size_t dataset_dim;
  size_t node_degree{64};
  size_t internal_node_degree{0};
  size_t max_iterations{50};
  float termination_threshold{0.0001};
  size_t output_graph_degree{32};
  cuvs::distance::DistanceType metric{cuvs::distance::DistanceType::L2Expanded};
};

template <typename Index_t>
class BloomFilter {
 public:
  BloomFilter(size_t nrow, size_t num_sets_per_list, size_t num_hashs);
  void add(size_t list_id, Index_t key);
  bool check(size_t list_id, Index_t key);
  void clear();
  void set_nrow(size_t nrow) { nrow_ = nrow; }

 private:
  uint32_t hash_0(uint32_t value);
  uint32_t hash_1(uint32_t value);

  static constexpr int num_bits_per_set_ = 512;
  bool is_cleared{true};
  std::vector<bool> bitsets_;
  size_t nrow_;
  size_t num_sets_per_list_;
  size_t num_hashs_;
};

template <typename Index_t>
struct GnndGraph {
  raft::resources const& res;
  static constexpr int segment_size = 32;
  InternalID_t<Index_t>* h_graph;
  size_t nrow;
  size_t node_degree;
  int num_samples;
  int num_segments;

  raft::host_matrix<DistData_t, size_t, raft::row_major> h_dists;
  raft::pinned_matrix<Index_t, size_t> h_graph_new;
  raft::pinned_vector<int2, size_t> h_list_sizes_new;
  raft::pinned_matrix<Index_t, size_t> h_graph_old;
  raft::pinned_vector<int2, size_t> h_list_sizes_old;
  BloomFilter<Index_t> bloom_filter;

  GnndGraph(const GnndGraph&)            = delete;
  GnndGraph& operator=(const GnndGraph&) = delete;
  GnndGraph(raft::resources const& res,
            const size_t nrow,
            const size_t node_degree,
            const size_t internal_node_degree,
            const size_t num_samples);
  void init_random_graph();
  void sample_graph_new(InternalID_t<Index_t>* new_neighbors, const size_t width);
  void sample_graph(bool sample_new);
  void update_graph(const InternalID_t<Index_t>* new_neighbors,
                    const DistData_t* new_dists,
                    const size_t width,
                    std::atomic<int64_t>& update_counter);
  void sort_lists();
  void clear();
  ~GnndGraph();
};

template <typename Data_t = float, typename Index_t = int>
class GNND {
 public:
  GNND(raft::resources const& res, const BuildConfig& build_config);
  GNND(const GNND&)            = delete;
  GNND& operator=(const GNND&) = delete;
  void build(Data_t* data,
             const Index_t nrow,
             Index_t* output_graph,
             bool return_distances,
             DistData_t* output_distances);
  ~GNND()    = default;
  using ID_t = InternalID_t<Index_t>;
  void reset(raft::resources const& res);

 private:
  void add_reverse_edges(Index_t* graph_ptr,
                         Index_t* h_rev_graph_ptr,
                         Index_t* d_rev_graph_ptr,
                         int2* list_sizes,
                         cudaStream_t stream = 0);
  void local_join(cudaStream_t stream = 0);

  raft::resources const& res;
  BuildConfig build_config_;
  GnndGraph<Index_t> graph_;
  std::atomic<int64_t> update_counter_;
  size_t nrow_;
  size_t ndim_;

  raft::device_matrix<__half, size_t, raft::row_major> d_data_;
  raft::device_vector<DistData_t, size_t> l2_norms_;
  raft::device_matrix<ID_t, size_t, raft::row_major> graph_buffer_;
  raft::device_matrix<DistData_t, size_t, raft::row_major> dists_buffer_;
  raft::pinned_matrix<ID_t, size_t> graph_host_buffer_;
  raft::pinned_matrix<DistData_t, size_t> dists_host_buffer_;
  raft::device_vector<int, size_t> d_locks_;
  raft::pinned_matrix<Index_t, size_t> h_rev_graph_new_;
  raft::pinned_matrix<Index_t, size_t> h_graph_old_;
  raft::pinned_matrix<Index_t, size_t> h_rev_graph_old_;
  raft::device_vector<int2, size_t> d_list_sizes_new_;
  raft::device_vector<int2, size_t> d_list_sizes_old_;
};

template <typename T,
          typename IdxT     = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
BuildConfig get_build_config(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  index<IdxT>& idx,
  size_t& extended_graph_degree,
  size_t& graph_degree)
{
  RAFT_EXPECTS(dataset.extent(0) < std::numeric_limits<int>::max() - 1,
               "The dataset size for GNND should be less than %d",
               std::numeric_limits<int>::max() - 1);
  auto allowed_metrics = params.metric == cuvs::distance::DistanceType::L2Expanded ||
                         params.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                         params.metric == cuvs::distance::DistanceType::CosineExpanded ||
                         params.metric == cuvs::distance::DistanceType::InnerProduct;
  RAFT_EXPECTS(allowed_metrics,
               "The metric for NN Descent should be L2Expanded, L2SqrtExpanded, CosineExpanded or "
               "InnerProduct");
  RAFT_EXPECTS(
    idx.metric() == params.metric,
    "The metrics set in nn_descent::index_params and nn_descent::index are inconsistent");
  size_t intermediate_degree = params.intermediate_graph_degree;
  graph_degree               = params.graph_degree;

  if (intermediate_degree >= static_cast<size_t>(dataset.extent(0))) {
    RAFT_LOG_WARN(
      "Intermediate graph degree cannot be larger than dataset size, reducing it to %lu",
      dataset.extent(0));
    intermediate_degree = dataset.extent(0) - 1;
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
    align32::roundUp(static_cast<size_t>(graph_degree * (graph_degree <= 32 ? 1.0 : 1.3)));
  size_t extended_intermediate_degree = align32::roundUp(
    static_cast<size_t>(intermediate_degree * (intermediate_degree <= 32 ? 1.0 : 1.3)));

  BuildConfig build_config{.max_dataset_size      = static_cast<size_t>(dataset.extent(0)),
                           .dataset_dim           = static_cast<size_t>(dataset.extent(1)),
                           .node_degree           = extended_graph_degree,
                           .internal_node_degree  = extended_intermediate_degree,
                           .max_iterations        = params.max_iterations,
                           .termination_threshold = params.termination_threshold,
                           .output_graph_degree   = params.graph_degree,
                           .metric                = params.metric};
  return build_config;
}

}  // namespace cuvs::neighbors::nn_descent::detail

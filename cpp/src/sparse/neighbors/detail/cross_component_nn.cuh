/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../../distance/masked_nn.cuh"
#include <cuvs/distance/distance.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/label/classlabels.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/matrix/scatter.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/sparse/op/reduce.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/fast_int_div.cuh>

#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cstdint>
#include <limits>

namespace cuvs::sparse::neighbors::detail {

/**
 * Base functor with reduction ops for performing masked 1-nn
 * computation.
 * @tparam ValueIdx
 * @tparam ValueT
 */
template <typename ValueIdx, typename ValueT>  // NOLINT(readability-identifier-naming)
struct fix_connectivities_red_op {
  ValueIdx m;

  // default constructor for cutlass
  DI fix_connectivities_red_op() : m(0) {}

  explicit fix_connectivities_red_op(ValueIdx m_) : m(m_) {};

  using kvp = typename raft::KeyValuePair<ValueIdx, ValueT>;
  DI void operator()(ValueIdx rit, kvp* out, const kvp& other) const
  {
    if (rit < m && other.value < out->value) {
      out->key   = other.key;
      out->value = other.value;
    }
  }

  DI auto operator()(ValueIdx rit, const kvp& a, const kvp& b) const -> kvp
  {
    if (rit < m && a.value < b.value) {
      return a;
    } else {
      return b;
    }
  }

  DI void init(ValueT* out, ValueT maxVal) const { *out = maxVal; }
  DI void init(kvp* out, ValueT maxVal) const
  {
    out->key   = -1;
    out->value = maxVal;
  }

  DI void init_key(ValueT& out, ValueIdx idx) const { return; }
  DI void init_key(kvp& out, ValueIdx idx) const { out.key = idx; }

  DI auto get_value(kvp& out) const -> ValueT { return out.value; }

  DI auto get_value(ValueT& out) const -> ValueT { return out; }

  /** The gather and scatter ensure that operator() is still consistent after rearranging the data.
   * TODO (tarang-jain): refactor cross_component_nn API to separate out the gather and scatter
   * functions from the reduction op. Reference: https://github.com/rapidsai/raft/issues/1614 */
  void gather(const raft::resources& handle, ValueIdx* map) {}

  void scatter(const raft::resources& handle, ValueIdx* map) {}
};

template <typename ValueIdx, typename ValueT>  // NOLINT(readability-identifier-naming)
struct mutual_reachability_fix_connectivities_red_op {
  ValueT* core_dists;
  ValueIdx m;

  DI mutual_reachability_fix_connectivities_red_op() : m(0), core_dists(nullptr) {}

  mutual_reachability_fix_connectivities_red_op(ValueT* core_dists, ValueIdx m)
    : core_dists(core_dists), m(m) {};

  using kvp = typename raft::KeyValuePair<ValueIdx, ValueT>;
  DI void operator()(ValueIdx rit, kvp* out, const kvp& other) const
  {
    if (rit < m && other.value < std::numeric_limits<ValueT>::max()) {
      ValueT core_dist_rit   = core_dists[rit];
      ValueT core_dist_other = max(core_dist_rit, max(core_dists[other.key], other.value));

      ValueT core_dist_out;
      if (out->key > -1) {
        core_dist_out = max(core_dist_rit, max(core_dists[out->key], out->value));
      } else {
        core_dist_out = out->value;
      }

      bool smaller = core_dist_other < core_dist_out;
      out->key     = smaller ? other.key : out->key;
      out->value   = smaller ? core_dist_other : core_dist_out;
    }
  }

  DI auto operator()(ValueIdx rit, const kvp& a, const kvp& b) const -> kvp
  {
    if (rit < m && a.key > -1) {
      ValueT core_dist_rit = core_dists[rit];
      ValueT core_dist_a   = max(core_dist_rit, max(core_dists[a.key], a.value));

      ValueT core_dist_b;
      if (b.key > -1) {
        core_dist_b = max(core_dist_rit, max(core_dists[b.key], b.value));
      } else {
        core_dist_b = b.value;
      }

      return core_dist_a < core_dist_b ? kvp(a.key, core_dist_a) : kvp(b.key, core_dist_b);
    }

    return b;
  }

  DI void init(ValueT* out, ValueT maxVal) const { *out = maxVal; }
  DI void init(kvp* out, ValueT maxVal) const
  {
    out->key   = -1;
    out->value = maxVal;
  }

  DI void init_key(ValueT& out, ValueIdx idx) const { return; }
  DI void init_key(kvp& out, ValueIdx idx) const { out.key = idx; }

  DI auto get_value(kvp& out) const -> ValueT { return out.value; }
  DI auto get_value(ValueT& out) const -> ValueT { return out; }

  void gather(const raft::resources& handle, ValueIdx* map)
  {
    auto tmp_core_dists = raft::make_device_vector<ValueT>(handle, m);
    thrust::gather(raft::resource::get_thrust_policy(handle),
                   map,
                   map + m,
                   core_dists,
                   tmp_core_dists.data_handle());
    raft::copy_async(
      core_dists, tmp_core_dists.data_handle(), m, raft::resource::get_cuda_stream(handle));
  }

  void scatter(const raft::resources& handle, ValueIdx* map)
  {
    auto tmp_core_dists = raft::make_device_vector<ValueT>(handle, m);
    thrust::scatter(raft::resource::get_thrust_policy(handle),
                    core_dists,
                    core_dists + m,
                    map,
                    tmp_core_dists.data_handle());
    raft::copy_async(
      core_dists, tmp_core_dists.data_handle(), m, raft::resource::get_cuda_stream(handle));
  }
};

/**
 * Assumes 3-iterator tuple containing COO rows, cols, and
 * a cub keyvalue pair object. Sorts the 3 arrays in
 * ascending order: row->col->keyvaluepair
 */
struct tuple_comp {
  template <typename One, typename Two>
  __host__ __device__ auto operator()(const One& t1, const Two& t2) -> bool
  {
    // sort first by each sample's color,
    if (thrust::get<0>(t1) < thrust::get<0>(t2)) return true;
    if (thrust::get<0>(t1) > thrust::get<0>(t2)) return false;

    // then by the color of each sample's closest neighbor,
    if (thrust::get<1>(t1) < thrust::get<1>(t2)) return true;
    if (thrust::get<1>(t1) > thrust::get<1>(t2)) return false;

    // then sort by value in descending order
    return thrust::get<2>(t1).value < thrust::get<2>(t2).value;
  }
};

template <typename LabelT, typename DataT>
struct cub_kvp_min_reduce {
  using kvp = raft::KeyValuePair<LabelT, DataT>;

  DI auto

  operator()(LabelT rit, const kvp& a, const kvp& b) -> kvp
  {
    return b.value < a.value ? b : a;
  }

  DI auto

  operator()(const kvp& a, const kvp& b) -> kvp
  {
    return b.value < a.value ? b : a;
  }

};  // kvp_min_reduce

/**
 * Gets the number of unique components from array of
 * colors or labels. This does not assume the components are
 * drawn from a monotonically increasing set.
 * @tparam ValueIdx
 * @param[in] colors array of components
 * @param[in] n_rows size of components array
 * @param[in] stream cuda stream for which to order cuda operations
 * @return total number of components
 */
template <typename ValueIdx>
auto get_n_components(ValueIdx* colors, size_t n_rows, cudaStream_t stream) -> ValueIdx
{
  rmm::device_uvector<ValueIdx> map_ids(0, stream);
  int num_clusters = raft::label::getUniquelabels(map_ids, colors, n_rows, stream);
  return num_clusters;
}

/**
 * Functor to look up a component for a vertex
 * @tparam ValueIdx
 * @tparam ValueT
 */
template <typename ValueIdx, typename ValueT>  // NOLINT(readability-identifier-naming)
struct lookup_color_op {
  ValueIdx* colors;

  explicit lookup_color_op(ValueIdx* colors_) : colors(colors_) {}

  DI auto

  operator()(const raft::KeyValuePair<ValueIdx, ValueT>& kvp) -> ValueIdx
  {
    return colors[kvp.key];
  }
};

/**
 * Compute the cross-component 1-nearest neighbors for each row in X using
 * the given array of components
 * @tparam ValueIdx
 * @tparam ValueT
 * @param[in] handle raft handle
 * @param[out] kvp mapping of closest neighbor vertex and distance for each vertex in the given
 * array of components
 * @param[out] nn_colors components of nearest neighbors for each vertex
 * @param[in] colors components of each vertex
 * @param[in] X original dense data
 * @param[in] n_rows number of rows in original dense data
 * @param[in] n_cols number of columns in original dense data
 * @param[in] row_batch_size row batch size for computing nearest neighbors
 * @param[in] col_batch_size column batch size for sorting and 'unsorting'
 * @param[in] reduction_op reduction operation for computing nearest neighbors
 */
template <typename ValueIdx,
          typename ValueT,
          typename RedOp>  // NOLINT(readability-identifier-naming)
void perform_1nn(raft::resources const& handle,
                 raft::KeyValuePair<ValueIdx, ValueT>* kvp,
                 ValueIdx* nn_colors,
                 ValueIdx* colors,
                 const ValueT* X,
                 size_t n_rows,
                 size_t n_cols,
                 size_t row_batch_size,
                 size_t col_batch_size,
                 RedOp reduction_op)
{
  auto stream      = raft::resource::get_cuda_stream(handle);
  auto exec_policy = raft::resource::get_thrust_policy(handle);

  auto sort_plan = raft::make_device_vector<ValueIdx>(handle, static_cast<ValueIdx>(n_rows));
  raft::linalg::map_offset(
    handle, sort_plan.view(), [] __device__(ValueIdx idx) -> ValueIdx { return idx; });

  thrust::sort_by_key(
    raft::resource::get_thrust_policy(handle), colors, colors + n_rows, sort_plan.data_handle());

  // Modify the reduction operation based on the sort plan.
  reduction_op.gather(handle, sort_plan.data_handle());

  auto x_mutable_view =
    raft::make_device_matrix_view<ValueT, ValueIdx>(const_cast<ValueT*>(X), n_rows, n_cols);
  auto sort_plan_const_view =
    raft::make_device_vector_view<const ValueIdx, ValueIdx>(sort_plan.data_handle(), n_rows);
  raft::matrix::gather(
    handle, x_mutable_view, sort_plan_const_view, static_cast<ValueIdx>(col_batch_size));

  // Get the number of unique components from the array of colors
  ValueIdx n_components = get_n_components(colors, n_rows, stream);

  // colors_group_idxs is an array containing the *end* indices of each color
  // component in colors. That is, the value of colors_group_idxs[j] indicates
  // the start of color j + 1, i.e., it is the inclusive scan of the sizes of
  // the color components.
  auto colors_group_idxs = raft::make_device_vector<ValueIdx, ValueIdx>(handle, n_components + 1);
  raft::sparse::convert::sorted_coo_to_csr(
    colors, n_rows, colors_group_idxs.data_handle(), n_components + 1, stream);

  auto group_idxs_view = raft::make_device_vector_view<const ValueIdx, ValueIdx>(
    colors_group_idxs.data_handle() + 1, n_components);

  auto x_norm = raft::make_device_vector<ValueT, ValueIdx>(handle, static_cast<ValueIdx>(n_rows));
  raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
    x_norm.data_handle(), X, n_cols, n_rows, stream);

  auto adj      = raft::make_device_matrix<bool, ValueIdx>(handle, row_batch_size, n_components);
  using out_t   = raft::KeyValuePair<ValueIdx, ValueT>;
  using param_t = cuvs::distance::masked_l2_nn_params<RedOp, RedOp>;

  bool apply_sqrt      = true;
  bool init_out_buffer = true;
  param_t params{reduction_op, reduction_op, apply_sqrt, init_out_buffer};

  auto x_full_view = raft::make_device_matrix_view<const ValueT, ValueIdx>(X, n_rows, n_cols);

  size_t n_batches = raft::ceildiv(n_rows, row_batch_size);

  for (size_t bid = 0; bid < n_batches; bid++) {
    size_t batch_offset   = bid * row_batch_size;
    size_t rows_per_batch = min(row_batch_size, n_rows - batch_offset);

    auto x_batch_view = raft::make_device_matrix_view<const ValueT, ValueIdx>(
      X + batch_offset * n_cols, rows_per_batch, n_cols);

    auto x_norm_batch_view = raft::make_device_vector_view<const ValueT, ValueIdx>(
      x_norm.data_handle() + batch_offset, rows_per_batch);

    auto mask_op = [colors,
                    n_components = raft::util::FastIntDiv(n_components),
                    batch_offset] __device__(ValueIdx idx) -> bool {
      ValueIdx row = idx / n_components;
      ValueIdx col = idx % n_components;
      return colors[batch_offset + row] != col;
    };

    auto adj_vector_view = raft::make_device_vector_view<bool, ValueIdx>(
      adj.data_handle(), rows_per_batch * n_components);

    raft::linalg::map_offset(handle, adj_vector_view, mask_op);

    auto adj_view = raft::make_device_matrix_view<const bool, ValueIdx>(
      adj.data_handle(), rows_per_batch, n_components);

    auto kvp_view = raft::make_device_vector_view<raft::KeyValuePair<ValueIdx, ValueT>, ValueIdx>(
      kvp + batch_offset, rows_per_batch);

    cuvs::distance::masked_l2_nn<ValueT, out_t, ValueIdx, RedOp, RedOp>(handle,
                                                                        params,
                                                                        x_batch_view,
                                                                        x_full_view,
                                                                        x_norm_batch_view,
                                                                        x_norm.view(),
                                                                        adj_view,
                                                                        group_idxs_view,
                                                                        kvp_view);
  }

  // Transform the keys so that they correctly point to the unpermuted indices.
  thrust::transform(exec_policy,
                    kvp,
                    kvp + n_rows,
                    kvp,
                    [sort_plan = sort_plan.data_handle()] __device__(out_t kvp) -> out_t {
                      out_t res;
                      res.value = kvp.value;
                      res.key   = sort_plan[kvp.key];
                      return res;
                    });

  // Undo permutation of the rows of X by scattering in place.
  raft::matrix::scatter(
    handle, x_mutable_view, sort_plan_const_view, static_cast<ValueIdx>(col_batch_size));

  // Undo permutation of the key-value pair and color vectors. This is not done
  // inplace, so using Two temporary vectors.
  auto tmp_colors = raft::make_device_vector<ValueIdx>(handle, n_rows);
  auto tmp_kvp    = raft::make_device_vector<out_t>(handle, n_rows);

  thrust::scatter(exec_policy, kvp, kvp + n_rows, sort_plan.data_handle(), tmp_kvp.data_handle());
  thrust::scatter(
    exec_policy, colors, colors + n_rows, sort_plan.data_handle(), tmp_colors.data_handle());
  reduction_op.scatter(handle, sort_plan.data_handle());

  raft::copy_async(colors, tmp_colors.data_handle(), n_rows, stream);
  raft::copy_async(kvp, tmp_kvp.data_handle(), n_rows, stream);

  lookup_color_op<ValueIdx, ValueT> extract_colors_op(colors);
  thrust::transform(exec_policy, kvp, kvp + n_rows, nn_colors, extract_colors_op);
}

/**
 * Sort nearest neighboring components wrt component of source vertices
 * @tparam ValueIdx
 * @tparam ValueT
 * @param[inout] colors components array of source vertices
 * @param[inout] nn_colors nearest neighbors components array
 * @param[inout] kvp nearest neighbor source vertex / distance array
 * @param[inout] src_indices array of source vertex indices which will become arg_sort
 *               indices
 * @param n_rows number of components in `colors`
 * @param stream stream for which to order CUDA operations
 */
template <typename ValueIdx, typename ValueT>  // NOLINT(readability-identifier-naming)
void sort_by_color(raft::resources const& handle,
                   ValueIdx* colors,
                   ValueIdx* nn_colors,
                   raft::KeyValuePair<ValueIdx, ValueT>* kvp,
                   ValueIdx* src_indices,
                   size_t n_rows)
{
  auto exec_policy = raft::resource::get_thrust_policy(handle);
  thrust::counting_iterator<ValueIdx> arg_sort_iter(0);
  thrust::copy(exec_policy, arg_sort_iter, arg_sort_iter + n_rows, src_indices);

  auto keys = thrust::make_zip_iterator(thrust::make_tuple(
    colors, nn_colors, reinterpret_cast<raft::KeyValuePair<ValueIdx, ValueT>*>(kvp)));
  auto vals = thrust::make_zip_iterator(thrust::make_tuple(src_indices));
  // get all the colors in contiguous locations so we can map them to warps.
  thrust::sort_by_key(exec_policy, keys, keys + n_rows, vals, tuple_comp());
}

template <typename ValueIdx, typename ValueT>  // NOLINT(readability-identifier-naming)
RAFT_KERNEL min_components_by_color_kernel(ValueIdx* out_rows,
                                           ValueIdx* out_cols,
                                           ValueT* out_vals,
                                           const ValueIdx* out_index,
                                           const ValueIdx* indices,
                                           const raft::KeyValuePair<ValueIdx, ValueT>* kvp,
                                           size_t nnz)
{
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= nnz) return;

  int idx = out_index[tid];

  if ((tid == 0 || (out_index[tid - 1] != idx))) {
    out_rows[idx] = indices[tid];
    out_cols[idx] = kvp[tid].key;
    out_vals[idx] = kvp[tid].value;
  }
}

/**
 * Computes the min set of unique components that neighbor the
 * components of each source vertex.
 * @tparam ValueIdx
 * @tparam ValueT
 * @param[out] coo output edge list
 * @param[in] out_index output indptr for ordering edge list
 * @param[in] indices indices of source vertices for each component
 * @param[in] kvp indices and distances of each destination vertex for each component
 * @param[in] n_colors number of components
 * @param[in] stream cuda stream for which to order cuda operations
 */
template <typename ValueIdx, typename ValueT>  // NOLINT(readability-identifier-naming)
void min_components_by_color(raft::sparse::COO<ValueT, ValueIdx>& coo,
                             const ValueIdx* out_index,
                             const ValueIdx* indices,
                             const raft::KeyValuePair<ValueIdx, ValueT>* kvp,
                             size_t nnz,
                             cudaStream_t stream)
{
  /**
   * Arrays should be ordered by: colors_indptr->colors_n->kvp.value
   * so the last element of each column in the input CSR should be
   * the min.
   */
  min_components_by_color_kernel<<<raft::ceildiv(nnz, (size_t)256), 256, 0, stream>>>(
    coo.rows(), coo.cols(), coo.vals(), out_index, indices, kvp, nnz);
}

/**
 * Connects the components of an otherwise unconnected knn graph
 * by computing a 1-nn to neighboring components of each data point
 * (e.g. component(nn) != component(self)) and reducing the results to
 * include the set of smallest destination components for each source
 * component. The result will not necessarily contain
 * n_components^2 - n_components number of elements because many components
 * will likely not be contained in the neighborhoods of 1-nns.
 * @tparam ValueIdx
 * @tparam ValueT
 * @param[in] handle raft handle
 * @param[out] out output edge list containing nearest cross-component
 *             edges.
 * @param[in] X original (row-major) dense matrix for which knn graph should be constructed.
 * @param[in] orig_colors array containing component number for each row of X
 * @param[in] n_rows number of rows in X
 * @param[in] n_cols number of cols in X
 * @param[in] reduction_op reduction operation for computing nearest neighbors. The reduction
 * operation must have `gather` and `scatter` functions defined
 * @param[in] row_batch_size the batch size for computing nearest neighbors. This parameter controls
 * the number of samples for which the nearest neighbors are computed at once. Therefore, it affects
 * the memory consumption mainly by reducing the size of the adjacency matrix for masked nearest
 * neighbors computation. default 0 indicates that no batching is done
 * @param[in] col_batch_size the input data is sorted and 'unsorted' based on color. An additional
 * scratch space buffer of shape (n_rows, col_batch_size) is created for this. Usually, this
 * parameter affects the memory consumption more drastically than the col_batch_size with a marginal
 * increase in compute time as the col_batch_size is reduced. default 0 indicates that no batching
 * is done
 * @param[in] metric distance metric
 */
template <typename ValueIdx,
          typename ValueT,
          typename RedOp,
          typename NnzT = size_t>  // NOLINT(readability-identifier-naming)
void cross_component_nn(
  raft::resources const& handle,
  raft::sparse::COO<ValueT, ValueIdx, NnzT>& out,
  const ValueT* X,
  const ValueIdx* orig_colors,
  ValueIdx n_rows,
  ValueIdx n_cols,
  RedOp reduction_op,
  ValueIdx row_batch_size,
  ValueIdx col_batch_size,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2SqrtExpanded)
{
  auto stream = raft::resource::get_cuda_stream(handle);

  RAFT_EXPECTS(metric == cuvs::distance::DistanceType::L2SqrtExpanded,
               "Fixing connectivities for an unconnected k-NN graph only "
               "supports L2SqrtExpanded currently.");

  if (row_batch_size == 0 || row_batch_size > n_rows) { row_batch_size = n_rows; }

  if (col_batch_size == 0 || col_batch_size > n_cols) { col_batch_size = n_cols; }

  rmm::device_uvector<ValueIdx> colors(n_rows, stream);

  // Normalize colors so they are drawn from a monotonically increasing set
  constexpr bool kZeroBased = true;
  raft::label::make_monotonic(
    colors.data(), const_cast<ValueIdx*>(orig_colors), n_rows, stream, kZeroBased);

  /**
   * First compute 1-nn for all colors where the color of each data point
   * is guaranteed to be != color of its nearest neighbor.
   */
  rmm::device_uvector<ValueIdx> nn_colors(n_rows, stream);
  rmm::device_uvector<raft::KeyValuePair<ValueIdx, ValueT>> temp_inds_dists(n_rows, stream);
  rmm::device_uvector<ValueIdx> src_indices(n_rows, stream);

  perform_1nn(handle,
              temp_inds_dists.data(),
              nn_colors.data(),
              colors.data(),
              X,
              n_rows,
              n_cols,
              row_batch_size,
              col_batch_size,
              reduction_op);

  /**
   * Sort data points by color (neighbors are not sorted)
   */
  // max_color + 1 = number of connected components
  // sort nn_colors by key w/ original colors
  sort_by_color(
    handle, colors.data(), nn_colors.data(), temp_inds_dists.data(), src_indices.data(), n_rows);

  /**
   * Take the min for any duplicate colors
   */
  // Compute mask of duplicates
  rmm::device_uvector<ValueIdx> out_index(n_rows + 1, stream);
  raft::sparse::op::compute_duplicates_mask(
    out_index.data(), colors.data(), nn_colors.data(), n_rows, stream);

  thrust::exclusive_scan(raft::resource::get_thrust_policy(handle),
                         out_index.data(),
                         out_index.data() + out_index.size(),
                         out_index.data());

  // compute final size
  ValueIdx size_int = 0;
  raft::update_host(&size_int, out_index.data() + (out_index.size() - 1), 1, stream);
  raft::resource::sync_stream(handle, stream);
  auto size = static_cast<NnzT>(size_int);

  size++;

  raft::sparse::COO<ValueT, ValueIdx, NnzT> min_edges(stream);
  min_edges.allocate(size, n_rows, n_rows, true, stream);

  min_components_by_color(
    min_edges, out_index.data(), src_indices.data(), temp_inds_dists.data(), n_rows, stream);

  /**
   * Symmetrize resulting edge list
   */
  raft::sparse::linalg::symmetrize(
    handle, min_edges.rows(), min_edges.cols(), min_edges.vals(), n_rows, n_rows, size, out);
}

};  // end namespace cuvs::sparse::neighbors::detail

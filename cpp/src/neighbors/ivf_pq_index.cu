/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/ivf_pq.hpp>

#include "detail/ann_utils.cuh"

#include <raft/core/operators.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <raft/util/cudart_utils.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace cuvs::neighbors::ivf_pq {
index_params index_params::from_dataset(raft::matrix_extent<int64_t> dataset,
                                        cuvs::distance::DistanceType metric)
{
  index_params params;
  params.n_lists =
    dataset.extent(0) < 4 * 2500 ? 4 : static_cast<uint32_t>(std::sqrt(dataset.extent(0)));
  params.n_lists = std::min<uint32_t>(params.n_lists, dataset.extent(0));
  params.pq_dim =
    raft::round_up_safe(static_cast<uint32_t>(dataset.extent(1) / 4), static_cast<uint32_t>(8));
  if (params.pq_dim == 0) params.pq_dim = 8;
  params.pq_bits                  = 8;
  params.kmeans_trainset_fraction = dataset.extent(0) < 10000 ? 1 : 0.1;
  params.metric                   = metric;
  return params;
}

template <typename IdxT>
index<IdxT>::index(raft::resources const& handle)
  // this constructor is just for a temporary index, for use in the deserialization
  // api. all the parameters here will get replaced with loaded values - that aren't
  // necessarily known ahead of time before deserialization.
  // TODO: do we even need a handle here - could just construct one?
  : index(handle,
          cuvs::distance::DistanceType::L2Expanded,
          codebook_gen::PER_SUBSPACE,
          0,
          0,
          8,
          0,
          true)
{
}

template <typename IdxT>
index<IdxT>::index(raft::resources const& handle, const index_params& params, uint32_t dim)
  : index(handle,
          params.metric,
          params.codebook_kind,
          params.n_lists,
          dim,
          params.pq_bits,
          params.pq_dim,
          params.conservative_memory_allocation)
{
}

template <typename IdxT>
index<IdxT>::index(raft::resources const& handle,
                   cuvs::distance::DistanceType metric,
                   codebook_gen codebook_kind,
                   uint32_t n_lists,
                   uint32_t dim,
                   uint32_t pq_bits,
                   uint32_t pq_dim,
                   bool conservative_memory_allocation)
  : cuvs::neighbors::index(),
    metric_(metric),
    codebook_kind_(codebook_kind),
    dim_(dim),
    pq_bits_(pq_bits),
    pq_dim_(pq_dim == 0 ? calculate_pq_dim(dim) : pq_dim),
    conservative_memory_allocation_(conservative_memory_allocation),
    lists_{n_lists},
    list_sizes_{raft::make_device_vector<uint32_t, uint32_t>(handle, n_lists)},
    pq_centers_{raft::make_device_mdarray<float>(handle, make_pq_centers_extents())},
    centers_{raft::make_device_matrix<float, uint32_t>(handle, n_lists, this->dim_ext())},
    centers_rot_{raft::make_device_matrix<float, uint32_t>(handle, n_lists, this->rot_dim())},
    rotation_matrix_{
      raft::make_device_matrix<float, uint32_t>(handle, this->rot_dim(), this->dim())},
    data_ptrs_{raft::make_device_vector<uint8_t*, uint32_t>(handle, n_lists)},
    inds_ptrs_{raft::make_device_vector<IdxT*, uint32_t>(handle, n_lists)},
    accum_sorted_sizes_{raft::make_host_vector<IdxT, uint32_t>(n_lists + 1)},
    pq_centers_view_{pq_centers_->view()},
    centers_view_{centers_->view()},
    centers_rot_view_{centers_rot_->view()},
    rotation_matrix_view_{rotation_matrix_->view()}
{
  check_consistency();
  accum_sorted_sizes_(n_lists) = 0;
}

template <typename IdxT>
index<IdxT>::index(
  raft::resources const& handle,
  cuvs::distance::DistanceType metric,
  codebook_gen codebook_kind,
  uint32_t n_lists,
  uint32_t dim,
  uint32_t pq_bits,
  uint32_t pq_dim,
  bool conservative_memory_allocation,
  raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers_view,
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_view,
  std::optional<raft::device_matrix_view<const float, uint32_t, raft::row_major>>
    centers_rot_view,
  std::optional<raft::device_matrix_view<const float, uint32_t, raft::row_major>>
    rotation_matrix_view)
  : cuvs::neighbors::index(),
    metric_(metric),
    codebook_kind_(codebook_kind),
    dim_(dim),
    pq_bits_(pq_bits),
    pq_dim_(pq_dim == 0 ? calculate_pq_dim(dim) : pq_dim),
    conservative_memory_allocation_(conservative_memory_allocation),
    lists_{n_lists},
    list_sizes_{raft::make_device_vector<uint32_t, uint32_t>(handle, n_lists)},
    data_ptrs_{raft::make_device_vector<uint8_t*, uint32_t>(handle, n_lists)},
    inds_ptrs_{raft::make_device_vector<IdxT*, uint32_t>(handle, n_lists)},
    accum_sorted_sizes_{raft::make_host_vector<IdxT, uint32_t>(n_lists + 1)},
    pq_centers_view_{pq_centers_view},
    centers_view_{centers_view},
    centers_rot_view_{centers_rot_view.value_or(
      raft::device_matrix_view<const float, uint32_t, raft::row_major>{})},
    rotation_matrix_view_{rotation_matrix_view.value_or(
      raft::device_matrix_view<const float, uint32_t, raft::row_major>{})}
{
  auto stream = raft::resource::get_cuda_stream(handle);
  
  // Check if we need to own the pq_centers (format conversion needed)
  auto expected_pq_extents = make_pq_centers_extents();
  bool pq_centers_match = (pq_centers_view.extent(0) == expected_pq_extents.extent(0)) &&
                          (pq_centers_view.extent(1) == expected_pq_extents.extent(1)) &&
                          (pq_centers_view.extent(2) == expected_pq_extents.extent(2));
  
  if (!pq_centers_match) {
    // Need to own and potentially transpose/convert the pq_centers
    pq_centers_ = raft::make_device_mdarray<float>(handle, expected_pq_extents);
    // TODO: Add conversion logic here
    pq_centers_view_ = pq_centers_->view();
  }
  
  // Check if we need to own the centers (format conversion needed)
  bool centers_match = (centers_view.extent(0) == n_lists) && 
                       (centers_view.extent(1) == this->dim_ext());
  
  if (!centers_match) {
    // Need to own and convert centers
    centers_ = raft::make_device_matrix<float, uint32_t>(handle, n_lists, this->dim_ext());
    
    // Clear the memory for the extended dimension
    RAFT_CUDA_TRY(cudaMemsetAsync(
      centers_->data_handle(), 0, centers_->size() * sizeof(float), stream));
    
    // Copy the centers, handling different dimensions
    if (centers_view.extent(1) == this->dim()) {
      // Centers provided with exact dimension, need to add padding and norms
      RAFT_CUDA_TRY(cudaMemcpy2DAsync(centers_->data_handle(),
                                      sizeof(float) * this->dim_ext(),
                                      centers_view.data_handle(),
                                      sizeof(float) * this->dim(),
                                      sizeof(float) * this->dim(),
                                      n_lists,
                                      cudaMemcpyDefault,
                                      stream));
      
      // Compute and add norms
      rmm::device_uvector<float> center_norms(n_lists, stream);
      raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
        center_norms.data(), centers_view.data_handle(), this->dim(), n_lists, stream);
      
      RAFT_CUDA_TRY(cudaMemcpy2DAsync(centers_->data_handle() + this->dim(),
                                      sizeof(float) * this->dim_ext(),
                                      center_norms.data(),
                                      sizeof(float),
                                      sizeof(float),
                                      n_lists,
                                      cudaMemcpyDefault,
                                      stream));
    } else {
      // Centers already have extended dimension
      raft::copy(centers_->data_handle(), centers_view.data_handle(), 
                 centers_view.size(), stream);
    }
    centers_view_ = centers_->view();
  }
  
  // Check if we need centers_rot
  if (centers_rot_view.has_value()) {
    bool centers_rot_match = (centers_rot_view.value().extent(0) == n_lists) &&
                             (centers_rot_view.value().extent(1) == this->rot_dim());
    if (!centers_rot_match) {
      // Need to own and convert centers_rot
      centers_rot_ = raft::make_device_matrix<float, uint32_t>(handle, n_lists, this->rot_dim());
      // TODO: Add conversion logic here if needed
      centers_rot_view_ = centers_rot_->view();
    } else {
      centers_rot_view_ = centers_rot_view.value();
    }
  } else {
    // Need to compute centers_rot if not provided
    centers_rot_ = raft::make_device_matrix<float, uint32_t>(handle, n_lists, this->rot_dim());
    centers_rot_view_ = centers_rot_->view();
  }
  
  // Check if we need rotation_matrix
  if (rotation_matrix_view.has_value()) {
    bool rotation_match = (rotation_matrix_view.value().extent(0) == this->rot_dim()) &&
                         (rotation_matrix_view.value().extent(1) == this->dim());
    if (!rotation_match) {
      // Need to own and convert rotation_matrix
      rotation_matrix_ = raft::make_device_matrix<float, uint32_t>(
        handle, this->rot_dim(), this->dim());
      // TODO: Add conversion logic here if needed
      rotation_matrix_view_ = rotation_matrix_->view();
    } else {
      rotation_matrix_view_ = rotation_matrix_view.value();
    }
  } else {
    // Need to compute rotation_matrix if not provided
    rotation_matrix_ = raft::make_device_matrix<float, uint32_t>(
      handle, this->rot_dim(), this->dim());
    rotation_matrix_view_ = rotation_matrix_->view();
  }
  
  check_consistency();
  accum_sorted_sizes_(n_lists) = 0;
}

template <typename IdxT>
IdxT index<IdxT>::size() const noexcept
{
  return accum_sorted_sizes_(n_lists());
}

template <typename IdxT>
uint32_t index<IdxT>::dim() const noexcept
{
  return dim_;
}

template <typename IdxT>
uint32_t index<IdxT>::dim_ext() const noexcept
{
  return raft::round_up_safe(dim() + 1, 8u);
}

template <typename IdxT>
uint32_t index<IdxT>::rot_dim() const noexcept
{
  return pq_len() * pq_dim();
}

template <typename IdxT>
uint32_t index<IdxT>::pq_bits() const noexcept
{
  return pq_bits_;
}

template <typename IdxT>
uint32_t index<IdxT>::pq_dim() const noexcept
{
  return pq_dim_;
}

template <typename IdxT>
uint32_t index<IdxT>::pq_len() const noexcept
{
  return raft::div_rounding_up_unsafe(dim(), pq_dim());
}

template <typename IdxT>
uint32_t index<IdxT>::pq_book_size() const noexcept
{
  return 1 << pq_bits();
}

template <typename IdxT>
cuvs::distance::DistanceType index<IdxT>::metric() const noexcept
{
  return metric_;
}

template <typename IdxT>
codebook_gen index<IdxT>::codebook_kind() const noexcept
{
  return codebook_kind_;
}

template <typename IdxT>
uint32_t index<IdxT>::n_lists() const noexcept
{
  return lists_.size();
}

template <typename IdxT>
bool index<IdxT>::conservative_memory_allocation() const noexcept
{
  return conservative_memory_allocation_;
}

template <typename IdxT>
raft::device_mdspan<float,
                    typename cuvs::neighbors::ivf_pq::index<IdxT>::pq_centers_extents,
                    raft::row_major>
index<IdxT>::pq_centers() noexcept
{
  return raft::make_device_mdspan<float, typename pq_centers_extents, raft::row_major>(
    const_cast<float*>(pq_centers_view_.data_handle()), pq_centers_view_.extents());
}

template <typename IdxT>
raft::device_mdspan<const float,
                    typename cuvs::neighbors::ivf_pq::index<IdxT>::pq_centers_extents,
                    raft::row_major>
index<IdxT>::pq_centers() const noexcept
{
  return pq_centers_view_;
}

template <typename IdxT>
std::vector<std::shared_ptr<list_data<IdxT>>>& index<IdxT>::lists() noexcept
{
  return lists_;
}

template <typename IdxT>
const std::vector<std::shared_ptr<list_data<IdxT>>>& index<IdxT>::lists() const noexcept
{
  return lists_;
}

template <typename IdxT>
raft::device_vector_view<uint8_t*, uint32_t, raft::row_major> index<IdxT>::data_ptrs() noexcept
{
  return data_ptrs_.view();
}

template <typename IdxT>
raft::device_vector_view<const uint8_t* const, uint32_t, raft::row_major> index<IdxT>::data_ptrs()
  const noexcept
{
  return raft::make_mdspan<const uint8_t* const, uint32_t, raft::row_major, false, true>(
    data_ptrs_.data_handle(), data_ptrs_.extents());
}

template <typename IdxT>
raft::device_vector_view<IdxT*, uint32_t, raft::row_major> index<IdxT>::inds_ptrs() noexcept
{
  return inds_ptrs_.view();
}

template <typename IdxT>
raft::device_vector_view<const IdxT* const, uint32_t, raft::row_major> index<IdxT>::inds_ptrs()
  const noexcept
{
  return raft::make_mdspan<const IdxT* const, uint32_t, raft::row_major, false, true>(
    inds_ptrs_.data_handle(), inds_ptrs_.extents());
}

template <typename IdxT>
raft::device_matrix_view<float, uint32_t, raft::row_major> index<IdxT>::rotation_matrix() noexcept
{
  return raft::make_device_matrix_view<float, uint32_t, raft::row_major>(
    const_cast<float*>(rotation_matrix_view_.data_handle()),
    rotation_matrix_view_.extent(0),
    rotation_matrix_view_.extent(1));
}

template <typename IdxT>
raft::device_matrix_view<const float, uint32_t, raft::row_major> index<IdxT>::rotation_matrix()
  const noexcept
{
  return rotation_matrix_view_;
}

template <typename IdxT>
raft::host_vector_view<IdxT, uint32_t, raft::row_major> index<IdxT>::accum_sorted_sizes() noexcept
{
  return accum_sorted_sizes_.view();
}

template <typename IdxT>
raft::host_vector_view<const IdxT, uint32_t, raft::row_major> index<IdxT>::accum_sorted_sizes()
  const noexcept
{
  return accum_sorted_sizes_.view();
}

template <typename IdxT>
raft::device_vector_view<uint32_t, uint32_t, raft::row_major> index<IdxT>::list_sizes() noexcept
{
  return list_sizes_.view();
}

template <typename IdxT>
raft::device_vector_view<const uint32_t, uint32_t, raft::row_major> index<IdxT>::list_sizes()
  const noexcept
{
  return list_sizes_.view();
}

template <typename IdxT>
raft::device_matrix_view<float, uint32_t, raft::row_major> index<IdxT>::centers() noexcept
{
  return raft::make_device_matrix_view<float, uint32_t, raft::row_major>(
    const_cast<float*>(centers_view_.data_handle()),
    centers_view_.extent(0),
    centers_view_.extent(1));
}

template <typename IdxT>
raft::device_matrix_view<const float, uint32_t, raft::row_major> index<IdxT>::centers()
  const noexcept
{
  return centers_view_;
}

template <typename IdxT>
raft::device_matrix_view<float, uint32_t, raft::row_major> index<IdxT>::centers_rot() noexcept
{
  return raft::make_device_matrix_view<float, uint32_t, raft::row_major>(
    const_cast<float*>(centers_rot_view_.data_handle()),
    centers_rot_view_.extent(0),
    centers_rot_view_.extent(1));
}

template <typename IdxT>
raft::device_matrix_view<const float, uint32_t, raft::row_major> index<IdxT>::centers_rot()
  const noexcept
{
  return centers_rot_view_;
}

template <typename IdxT>
uint32_t index<IdxT>::get_list_size_in_bytes(uint32_t label)
{
  RAFT_EXPECTS(label < this->n_lists(),
               "Expected label to be less than number of lists in the index");
  auto& list_data = this->lists()[label]->data;
  return list_data.size();
}

template <typename IdxT>
void index<IdxT>::check_consistency()
{
  RAFT_EXPECTS(pq_bits() >= 4 && pq_bits() <= 8,
               "`pq_bits` must be within closed range [4,8], but got %u.",
               pq_bits());
  RAFT_EXPECTS((pq_bits() * pq_dim()) % 8 == 0,
               "`pq_bits * pq_dim` must be a multiple of 8, but got %u * %u = %u.",
               pq_bits(),
               pq_dim(),
               pq_bits() * pq_dim());
}

template <typename IdxT>
typename index<IdxT>::pq_centers_extents index<IdxT>::make_pq_centers_extents()
{
  switch (codebook_kind()) {
    case codebook_gen::PER_SUBSPACE:
      return raft::make_extents<uint32_t>(pq_dim(), pq_len(), pq_book_size());
    case codebook_gen::PER_CLUSTER:
      return raft::make_extents<uint32_t>(n_lists(), pq_len(), pq_book_size());
    default: RAFT_FAIL("Unreachable code");
  }
}

template <typename IdxT>
uint32_t index<IdxT>::calculate_pq_dim(uint32_t dim)
{
  // If the dimensionality is large enough, we can reduce it to improve performance
  if (dim >= 128) { dim /= 2; }
  // Round it down to 32 to improve performance.
  auto r = raft::round_down_safe<uint32_t>(dim, 32);
  if (r > 0) return r;
  // If the dimensionality is really low, round it to the closest power-of-two
  r = 1;
  while ((r << 1) <= dim) {
    r = r << 1;
  }
  return r;
}

template <typename IdxT>
raft::device_matrix_view<const int8_t, uint32_t, raft::row_major> index<IdxT>::rotation_matrix_int8(
  const raft::resources& res) const
{
  if (!rotation_matrix_int8_.has_value()) {
    rotation_matrix_int8_.emplace(
      raft::make_device_mdarray<int8_t, uint32_t>(res, rotation_matrix().extents()));
    raft::linalg::map(res,
                      rotation_matrix_int8_->view(),
                      cuvs::spatial::knn::detail::utils::mapping<int8_t>{},
                      rotation_matrix());
  }
  return rotation_matrix_int8_->view();
}

template <typename IdxT>
raft::device_matrix_view<const int8_t, uint32_t, raft::row_major> index<IdxT>::centers_int8(
  const raft::resources& res) const
{
  if (!centers_int8_.has_value()) {
    uint32_t n_lists      = this->n_lists();
    uint32_t dim          = this->dim();
    uint32_t dim_ext      = this->dim_ext();
    uint32_t dim_ext_int8 = raft::round_up_safe(dim + 2, 16u);
    centers_int8_.emplace(raft::make_device_matrix<int8_t, uint32_t>(res, n_lists, dim_ext_int8));
    auto* inputs = centers().data_handle();
    /* NOTE: maximizing the range and the precision of int8_t GEMM

    int8_t has a very limited range [-128, 127], which is problematic when storing both vectors and
    their squared norms in one place.

    We map all dimensions by multiplying by 128. But that means we need to multiply the squared norm
    component by `128^2`, which we cannot afford, since it most likely overflows.
    So, a naive mapping would be:
    ```
      [c_1 * 128, c_2, * 128, ...., c_(dim-1) * 128,  n2 * 128 * 128, 0 ... 0]
      • [q_1 * 128, q_2 * 128, ..., q_(dim-1)*128, -0.5, 0, ... 0]
    ```

    Which is at first can be improved by moving one 128 to the query side:
    ```
      [c_1 * 128, c_2, * 128, ...., c_(dim-1) * 128,  n2 * 128, 0 ... 0]
      • [q_1 * 128, q_2 * 128, ..., q_(dim-1)*128, -64, 0, ... 0]
    ```

    Yet this still only works for vectors with L2 norms not bigger than one and has a rather awful
    granularity of 64. To improve both the range and the precision, we count the number of available
    slots `m > 2` and decompose the squared norm, such that:
    ```
      0.5 * 128 * n2 = 64 * n2 = 128 * z + (m - 1) * y
    ```
    where `y` maximizes the available range while `z` encodes the rounding error.
    Then we get following dot product during the coarse search:
    ```
      [c_1 * 128, c_2, * 128, ...., c_(dim-1) * 128,  z, y, ... y]
      • [q_1 * 128, q_2 * 128, ..., q_(dim-1)*128, 1 - m,  -128, ... -128]
    ```
    `m` is maximum 16, so we get the coefficient much lower than the naive 64 on the query side; and
    it is limited by the range we can cover (the squared norm must be within `m * 2` before
    normalization).
    */
    raft::linalg::map_offset(
      res, centers_int8_->view(), [dim, dim_ext, dim_ext_int8, inputs] __device__(uint32_t ix) {
        uint32_t col = ix % dim_ext_int8;
        uint32_t row = ix / dim_ext_int8;
        if (col < dim) {
          return static_cast<int8_t>(
            std::clamp(inputs[col + row * dim_ext] * 128.0f, -128.0f, 127.f));
        }
        auto x = inputs[row * dim_ext + dim];
        auto c = 64.0f / static_cast<float>(dim_ext_int8 - dim - 1);
        auto y = std::clamp(x * c, -128.0f, 127.f);
        auto z = std::clamp((y - std::round(y)) * 128.0f, -128.0f, 127.f);
        if (col > dim) { return static_cast<int8_t>(std::round(y)); }
        return static_cast<int8_t>(z);
      });
  }
  return centers_int8_->view();
}

template <typename IdxT>
raft::device_matrix_view<const half, uint32_t, raft::row_major> index<IdxT>::rotation_matrix_half(
  const raft::resources& res) const
{
  if (!rotation_matrix_half_.has_value()) {
    rotation_matrix_half_.emplace(
      raft::make_device_mdarray<half, uint32_t>(res, rotation_matrix().extents()));
    raft::linalg::map(res, rotation_matrix_half_->view(), raft::cast_op<half>{}, rotation_matrix());
  }
  return rotation_matrix_half_->view();
}

template <typename IdxT>
raft::device_matrix_view<const half, uint32_t, raft::row_major> index<IdxT>::centers_half(
  const raft::resources& res) const
{
  if (!centers_half_.has_value()) {
    centers_half_.emplace(raft::make_device_mdarray<half, uint32_t>(res, centers().extents()));
    raft::linalg::map(res, centers_half_->view(), raft::cast_op<half>{}, centers());
  }
  return centers_half_->view();
}

template <typename IdxT>
void index<IdxT>::update_centers_rot(
  raft::resources const& res,
  raft::device_matrix_view<const float, uint32_t, raft::row_major> new_centers_rot)
{
  RAFT_EXPECTS(new_centers_rot.extent(0) == n_lists(),
               "Number of rows in centers_rot must equal n_lists");
  RAFT_EXPECTS(new_centers_rot.extent(1) == rot_dim(),
               "Number of columns in centers_rot must equal rot_dim");
  
  if (centers_rot_.has_value()) {
    // Copy into owned storage
    raft::copy(centers_rot_->data_handle(), 
               new_centers_rot.data_handle(),
               new_centers_rot.size(),
               raft::resource::get_cuda_stream(res));
  } else {
    // Just update the view
    centers_rot_view_ = new_centers_rot;
  }
}

template <typename IdxT>
void index<IdxT>::update_centers(
  raft::resources const& res,
  raft::device_matrix_view<const float, uint32_t, raft::row_major> new_centers)
{
  RAFT_EXPECTS(new_centers.extent(0) == n_lists(),
               "Number of rows in centers must equal n_lists");
  
  auto stream = raft::resource::get_cuda_stream(res);
  
  if (new_centers.extent(1) == dim_ext()) {
    // Direct update if dimensions match
    if (centers_.has_value()) {
      raft::copy(centers_->data_handle(), 
                 new_centers.data_handle(),
                 new_centers.size(),
                 stream);
    } else {
      centers_view_ = new_centers;
    }
  } else if (new_centers.extent(1) == dim()) {
    // Need to add padding and norms
    if (!centers_.has_value()) {
      centers_ = raft::make_device_matrix<float, uint32_t>(res, n_lists(), dim_ext());
    }
    
    // Clear the memory
    RAFT_CUDA_TRY(cudaMemsetAsync(centers_->data_handle(), 0, 
                                  centers_->size() * sizeof(float), stream));
    
    // Copy centers
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(centers_->data_handle(),
                                    sizeof(float) * dim_ext(),
                                    new_centers.data_handle(),
                                    sizeof(float) * dim(),
                                    sizeof(float) * dim(),
                                    n_lists(),
                                    cudaMemcpyDefault,
                                    stream));
    
    // Compute and add norms
    rmm::device_uvector<float> center_norms(n_lists(), stream);
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
      center_norms.data(), new_centers.data_handle(), dim(), n_lists(), stream);
    
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(centers_->data_handle() + dim(),
                                    sizeof(float) * dim_ext(),
                                    center_norms.data(),
                                    sizeof(float),
                                    sizeof(float),
                                    n_lists(),
                                    cudaMemcpyDefault,
                                    stream));
    
    centers_view_ = centers_->view();
  } else {
    RAFT_FAIL("Invalid centers dimensions: expected %u or %u columns, got %u",
              dim(), dim_ext(), new_centers.extent(1));
  }
}

template <typename IdxT>
void index<IdxT>::update_pq_centers(
  raft::resources const& res,
  raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> new_pq_centers)
{
  auto expected_extents = make_pq_centers_extents();
  
  RAFT_EXPECTS(new_pq_centers.extent(0) == expected_extents.extent(0),
               "PQ centers extent 0 mismatch");
  RAFT_EXPECTS(new_pq_centers.extent(1) == expected_extents.extent(1),
               "PQ centers extent 1 mismatch");
  RAFT_EXPECTS(new_pq_centers.extent(2) == expected_extents.extent(2),
               "PQ centers extent 2 mismatch");
  
  if (pq_centers_.has_value()) {
    // Copy into owned storage
    raft::copy(pq_centers_->data_handle(),
               new_pq_centers.data_handle(),
               new_pq_centers.size(),
               raft::resource::get_cuda_stream(res));
  } else {
    // Just update the view
    pq_centers_view_ = new_pq_centers;
  }
}

template struct index<int64_t>;

}  // namespace cuvs::neighbors::ivf_pq

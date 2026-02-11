/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "base_strategy.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/util/cuda_dev_essentials.cuh>

#include <cuco/static_map.cuh>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

#include <cooperative_groups.h>
#include <rmm/device_uvector.hpp>

// this is needed by cuco as key, value must be bitwise comparable.
// compilers don't declare float/double as bitwise comparable
// but that is too strict
// for example, the following is true (or 0):
// float a = 5;
// float b = 5;
// memcmp(&a, &b, sizeof(float));
CUCO_DECLARE_BITWISE_COMPARABLE(float);
CUCO_DECLARE_BITWISE_COMPARABLE(double);

namespace cuvs::distance::detail::sparse {

template <typename ValueIdx, typename value_t, int tpb>  // NOLINT(readability-identifier-naming)
class hash_strategy : public coo_spmv_strategy<ValueIdx, value_t, tpb> {
 public:
  static constexpr ValueIdx kEmptyKeySentinel  = ValueIdx{-1};
  static constexpr value_t kEmptyValueSentinel = value_t{0};
  using probing_scheme_type = cuco::linear_probing<1, cuco::murmurhash3_32<ValueIdx>>;
  using storage_ref_type =
    cuco::bucket_storage_ref<cuco::pair<ValueIdx, value_t>, 1, cuco::extent<int>>;
  using map_type = cuco::static_map_ref<ValueIdx,
                                        value_t,
                                        cuda::thread_scope_block,
                                        cuda::std::equal_to<ValueIdx>,
                                        probing_scheme_type,
                                        storage_ref_type,
                                        cuco::op::insert_tag,
                                        cuco::op::find_tag>;

  explicit hash_strategy(const distances_config_t<ValueIdx, value_t>& config_,
                         float capacity_threshold_ = 0.5,
                         int map_size_             = get_map_size())
    : coo_spmv_strategy<ValueIdx, value_t, tpb>(config_),
      capacity_threshold_(capacity_threshold_),
      map_size_(map_size_)
  {
  }

  void chunking_needed(const ValueIdx* indptr_,
                       const ValueIdx n_rows,
                       rmm::device_uvector<ValueIdx>& mask_indptr,
                       std::tuple<ValueIdx, ValueIdx>& n_rows_divided,
                       cudaStream_t stream)
  {
    auto policy = raft::resource::get_thrust_policy(this->config_.handle);

    auto less                   = thrust::copy_if(policy,
                                thrust::make_counting_iterator(ValueIdx(0)),
                                thrust::make_counting_iterator(n_rows),
                                mask_indptr.data(),
                                fits_in_hash_table(indptr_, 0, capacity_threshold_ * map_size_));
    std::get<0>(n_rows_divided) = less - mask_indptr.data();

    auto more = thrust::copy_if(
      policy,
      thrust::make_counting_iterator(ValueIdx(0)),
      thrust::make_counting_iterator(n_rows),
      less,
      fits_in_hash_table(
        indptr_, capacity_threshold_ * map_size_, std::numeric_limits<ValueIdx>::max()));
    std::get<1>(n_rows_divided) = more - less;
  }

  template <typename ProductF, typename AccumF, typename WriteF>
  void dispatch(value_t* out_dists,
                ValueIdx* coo_rows_b,
                ProductF product_func,
                AccumF accum_func,
                WriteF write_func,
                int chunk_size)
  {
    auto n_blocks_per_row = raft::ceildiv(this->config_.b_nnz, chunk_size * tpb);
    rmm::device_uvector<ValueIdx> mask_indptr(
      this->config_.a_nrows, raft::resource::get_cuda_stream(this->config_.handle));
    std::tuple<ValueIdx, ValueIdx> n_rows_divided;

    chunking_needed(this->config_.a_indptr,
                    this->config_.a_nrows,
                    mask_indptr,
                    n_rows_divided,
                    raft::resource::get_cuda_stream(this->config_.handle));

    auto less_rows = std::get<0>(n_rows_divided);
    if (less_rows > 0) {
      mask_row_it<ValueIdx> less(this->config_.a_indptr, less_rows, mask_indptr.data());

      auto n_less_blocks = less_rows * n_blocks_per_row;
      this->dispatch_base(*this,
                          map_size_,
                          less,
                          out_dists,
                          coo_rows_b,
                          product_func,
                          accum_func,
                          write_func,
                          chunk_size,
                          n_less_blocks,
                          n_blocks_per_row);
    }

    auto more_rows = std::get<1>(n_rows_divided);
    if (more_rows > 0) {
      rmm::device_uvector<ValueIdx> n_chunks_per_row(
        more_rows + 1, raft::resource::get_cuda_stream(this->config_.handle));
      rmm::device_uvector<ValueIdx> chunk_indices(
        0, raft::resource::get_cuda_stream(this->config_.handle));
      chunked_mask_row_it<ValueIdx>::init(this->config_.a_indptr,
                                          mask_indptr.data() + less_rows,
                                          more_rows,
                                          capacity_threshold_ * map_size_,
                                          n_chunks_per_row,
                                          chunk_indices,
                                          raft::resource::get_cuda_stream(this->config_.handle));

      chunked_mask_row_it<ValueIdx> more(this->config_.a_indptr,
                                         more_rows,
                                         mask_indptr.data() + less_rows,
                                         capacity_threshold_ * map_size_,
                                         n_chunks_per_row.data(),
                                         chunk_indices.data(),
                                         raft::resource::get_cuda_stream(this->config_.handle));

      auto n_more_blocks = more.total_row_blocks * n_blocks_per_row;
      this->dispatch_base(*this,
                          map_size_,
                          more,
                          out_dists,
                          coo_rows_b,
                          product_func,
                          accum_func,
                          write_func,
                          chunk_size,
                          n_more_blocks,
                          n_blocks_per_row);
    }
  }

  template <typename ProductF, typename AccumF, typename WriteF>
  void dispatch_rev(value_t* out_dists,
                    ValueIdx* coo_rows_a,
                    ProductF product_func,
                    AccumF accum_func,
                    WriteF write_func,
                    int chunk_size)
  {
    auto n_blocks_per_row = raft::ceildiv(this->config_.a_nnz, chunk_size * tpb);
    rmm::device_uvector<ValueIdx> mask_indptr(
      this->config_.b_nrows, raft::resource::get_cuda_stream(this->config_.handle));
    std::tuple<ValueIdx, ValueIdx> n_rows_divided;

    chunking_needed(this->config_.b_indptr,
                    this->config_.b_nrows,
                    mask_indptr,
                    n_rows_divided,
                    raft::resource::get_cuda_stream(this->config_.handle));

    auto less_rows = std::get<0>(n_rows_divided);
    if (less_rows > 0) {
      mask_row_it<ValueIdx> less(this->config_.b_indptr, less_rows, mask_indptr.data());

      auto n_less_blocks = less_rows * n_blocks_per_row;
      this->dispatch_base_rev(*this,
                              map_size_,
                              less,
                              out_dists,
                              coo_rows_a,
                              product_func,
                              accum_func,
                              write_func,
                              chunk_size,
                              n_less_blocks,
                              n_blocks_per_row);
    }

    auto more_rows = std::get<1>(n_rows_divided);
    if (more_rows > 0) {
      rmm::device_uvector<ValueIdx> n_chunks_per_row(
        more_rows + 1, raft::resource::get_cuda_stream(this->config_.handle));
      rmm::device_uvector<ValueIdx> chunk_indices(
        0, raft::resource::get_cuda_stream(this->config_.handle));
      chunked_mask_row_it<ValueIdx>::init(this->config_.b_indptr,
                                          mask_indptr.data() + less_rows,
                                          more_rows,
                                          capacity_threshold_ * map_size_,
                                          n_chunks_per_row,
                                          chunk_indices,
                                          raft::resource::get_cuda_stream(this->config_.handle));

      chunked_mask_row_it<ValueIdx> more(this->config_.b_indptr,
                                         more_rows,
                                         mask_indptr.data() + less_rows,
                                         capacity_threshold_ * map_size_,
                                         n_chunks_per_row.data(),
                                         chunk_indices.data(),
                                         raft::resource::get_cuda_stream(this->config_.handle));

      auto n_more_blocks = more.total_row_blocks * n_blocks_per_row;
      this->dispatch_base_rev(*this,
                              map_size_,
                              more,
                              out_dists,
                              coo_rows_a,
                              product_func,
                              accum_func,
                              write_func,
                              chunk_size,
                              n_more_blocks,
                              n_blocks_per_row);
    }
  }

  __device__ inline auto init_map(void* storage, const ValueIdx& cache_size) -> map_type
  {
    auto map_ref =
      map_type{cuco::empty_key<ValueIdx>{kEmptyKeySentinel},
               cuco::empty_value<value_t>{kEmptyValueSentinel},
               cuda::std::equal_to<ValueIdx>{},
               probing_scheme_type{},
               cuco::cuda_thread_scope<cuda::thread_scope_block>{},
               storage_ref_type{cuco::extent<int>{cache_size},
                                static_cast<typename storage_ref_type::value_type*>(storage)}};
    map_ref.initialize(cooperative_groups::this_thread_block());

    return map_ref;
  }

  __device__ inline void insert(map_type& map_ref, const ValueIdx& key, const value_t& value)
  {
    map_ref.insert(cuco::pair{key, value});
  }

  // Note: init_find is now merged with init_map since the new API uses the same ref for both
  // operations

  __device__ inline auto find(map_type& map_ref, const ValueIdx& key) -> value_t
  {
    auto a_pair = map_ref.find(key);

    value_t a_col = 0.0;
    if (a_pair != map_ref.end()) { a_col = a_pair->second; }
    return a_col;
  }

  struct fits_in_hash_table {
   public:
    fits_in_hash_table(const ValueIdx* indptr_, ValueIdx degree_l_, ValueIdx degree_r_)
      : indptr_(indptr_), degree_l_(degree_l_), degree_r_(degree_r_)
    {
    }

    __host__ __device__ auto operator()(const ValueIdx& i) -> bool
    {
      auto degree = indptr_[i + 1] - indptr_[i];

      return degree >= degree_l_ && degree < degree_r_;
    }

   private:
    const ValueIdx* indptr_;
    const ValueIdx degree_l_, degree_r_;
  };

  inline static auto get_map_size() -> int
  {
    return (raft::getSharedMemPerBlock() - ((tpb / raft::warp_size()) * sizeof(value_t))) /
           sizeof(cuco::pair<ValueIdx, value_t>);
  }

 private:
  float capacity_threshold_;
  int map_size_;
};

}  // namespace cuvs::distance::detail::sparse

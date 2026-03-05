/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../detail/vpq_dataset.cuh"
#include "../../ivf_pq/ivf_pq_codepacking.cuh"
#include <chrono>
#include <cmath>
#include <cuvs/neighbors/common.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/matrix/init.cuh>

#include "scann_common.cuh"
using namespace cuvs::neighbors;

namespace cuvs::neighbors::experimental::scann::detail {

/** Fix the internal indexing type to avoid integer underflows/overflows */
using ix_t = int64_t;

template <uint32_t BlockSize,
          uint32_t PqBits,
          typename DataT,
          typename MathT,
          typename IdxT,
          typename LabelT>
__launch_bounds__(BlockSize) RAFT_KERNEL process_and_fill_codes_subspaces_kernel(
  raft::device_matrix_view<uint8_t, IdxT, raft::row_major> out_codes,
  raft::device_matrix_view<const DataT, IdxT, raft::row_major> dataset,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> vq_centers,
  raft::device_vector_view<const LabelT, IdxT, raft::row_major> vq_labels,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers)
{
  constexpr uint32_t kSubWarpSize = std::min<uint32_t>(raft::WarpSize, 1u << PqBits);
  using subwarp_align             = raft::Pow2<kSubWarpSize>;
  const IdxT row_ix = subwarp_align::div(IdxT{threadIdx.x} + IdxT{BlockSize} * IdxT{blockIdx.x});
  if (row_ix >= out_codes.extent(0)) { return; }

  const uint32_t pq_dim = raft::div_rounding_up_unsafe(vq_centers.extent(1), pq_centers.extent(1));

  const uint32_t lane_id = raft::Pow2<kSubWarpSize>::mod(threadIdx.x);
  const LabelT vq_label  = vq_labels(row_ix);

  // write label
  auto* out_label_ptr = reinterpret_cast<LabelT*>(&out_codes(row_ix, 0));
  if (lane_id == 0) { *out_label_ptr = vq_label; }

  auto* out_codes_ptr = reinterpret_cast<uint8_t*>(out_label_ptr + 1);
  cuvs::neighbors::ivf_pq::detail::bitfield_view_t<PqBits> code_view{out_codes_ptr};
  for (uint32_t j = 0; j < pq_dim; j++) {
    // find PQ label
    int subspace_offset   = j * pq_centers.extent(1) * (1 << PqBits);
    auto pq_subspace_view = raft::make_device_matrix_view(
      pq_centers.data_handle() + subspace_offset, (uint32_t)(1 << PqBits), pq_centers.extent(1));
    auto pq_centers_smem =
      raft::make_device_matrix_view<const MathT, uint32_t, raft::row_major>(nullptr, 0, 0);
    uint8_t code = cuvs::neighbors::detail::compute_code<kSubWarpSize, uint8_t>(
      dataset, vq_centers, pq_centers_smem, pq_subspace_view, row_ix, j, vq_label);
    // TODO: this writes in global memory one byte per warp, which is very slow.
    //  It's better to keep the codes in the shared memory or registers and dump them at once.
    if (lane_id == 0) { code_view[j] = code; }
  }
}

template <typename MathT, typename IdxT, typename DatasetT>
auto process_and_fill_codes_subspaces(
  const raft::resources& res,
  const vpq_params& params,
  const DatasetT& dataset,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> vq_centers,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers)
  -> raft::device_matrix<uint8_t, IdxT, raft::row_major>
{
  using data_t     = typename DatasetT::value_type;
  using cdataset_t = vpq_dataset<MathT, IdxT>;
  using label_t    = uint32_t;

  const ix_t n_rows       = dataset.extent(0);
  const ix_t dim          = dataset.extent(1);
  const ix_t pq_dim       = params.pq_dim;
  const ix_t pq_bits      = params.pq_bits;
  const ix_t pq_n_centers = ix_t{1} << pq_bits;
  // NB: codes must be aligned at least to sizeof(label_t) to be able to read labels.
  const ix_t codes_rowlen =
    sizeof(label_t) * (1 + raft::div_rounding_up_safe<ix_t>(pq_dim * pq_bits, 8 * sizeof(label_t)));

  auto codes = raft::make_device_matrix<uint8_t, IdxT, raft::row_major>(res, n_rows, codes_rowlen);

  auto stream = raft::resource::get_cuda_stream(res);

  // TODO: with scaling workspace we could choose the batch size dynamically
  constexpr ix_t kBlockSize  = 256;
  const ix_t threads_per_vec = std::min<ix_t>(raft::WarpSize, pq_n_centers);
  dim3 threads(kBlockSize, 1, 1);

  auto kernel = [](uint32_t pq_bits) {
    switch (pq_bits) {
      case 4:
        return process_and_fill_codes_subspaces_kernel<kBlockSize, 4, data_t, MathT, IdxT, label_t>;
      case 8:
        return process_and_fill_codes_subspaces_kernel<kBlockSize, 8, data_t, MathT, IdxT, label_t>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be 4 or 8", pq_bits);
    }
  }(pq_bits);

  auto labels = raft::make_device_vector<label_t, IdxT>(res, dataset.extent(0));
  cuvs::neighbors::detail::predict_vq<label_t>(res, dataset, vq_centers, labels.view());

  dim3 blocks(raft::div_rounding_up_safe<ix_t>(n_rows, kBlockSize / threads_per_vec), 1, 1);

  kernel<<<blocks, threads, 0, stream>>>(
    raft::make_device_matrix_view<uint8_t, IdxT>(codes.data_handle(), n_rows, codes_rowlen),
    dataset,
    vq_centers,
    raft::make_const_mdspan(labels.view()),
    pq_centers);

  RAFT_CUDA_TRY(cudaPeekAtLastError());

  return codes;
}

template <typename T>
auto create_pq_codebook(raft::resources const& res,
                        raft::device_matrix_view<const T, int64_t> residuals,
                        cuvs::neighbors::vpq_params ps)
  -> raft::device_matrix<T, uint32_t, raft::row_major>
{
  // Create codebooks (vq initialized to 0s since we don't need here)
  auto vq_code_book =
    raft::make_device_matrix<T, uint32_t, raft::row_major>(res, 1, residuals.extent(1));
  raft::linalg::map_offset(res, vq_code_book.view(), [] __device__(size_t i) { return 0; });

  auto pq_code_book = cuvs::neighbors::detail::train_pq<T>(
    res, ps, residuals, raft::make_const_mdspan(vq_code_book.view()));

  return pq_code_book;
}

/**
 * @brief Subtract cluster center coordinates from each dataset vector.
 *
 * residual[i, k] = dataset[i ,k] - centers[l, k],
 * where l = labels[i], the cluster label corresponding to vector i.
 *
 * @tparam T
 * @tparam LabelT
 * @param res raft resources
 * @param dataset dataset vectors, size [n_rows, dim]
 * @param centers cluster center coordinates, size [n_clusters, dim]
 * @param labels cluster labels, size [n_rows]
 * @return device matrix with the residuals, size [n_rows, dim]
 */
template <typename T, typename LabelT>
auto compute_residuals(raft::resources const& res,
                       raft::device_matrix_view<const T, int64_t> dataset,
                       raft::device_matrix_view<const T, int64_t> centers,
                       raft::device_vector_view<const LabelT, int64_t> labels)
  -> raft::device_matrix<T, int64_t, raft::row_major>
{
  auto dim       = dataset.extent(1);
  auto residuals = raft::make_device_matrix<T, int64_t>(res, labels.extent(0), dim);

  // compute residuals for AVQ assignments
  raft::linalg::map_offset(
    res, residuals.view(), [dataset, centers, labels, dim] __device__(size_t i) {
      int row_idx = i / dim;
      int el_idx  = i % dim;
      return dataset(row_idx, el_idx) - centers(labels(row_idx), el_idx);
    });

  return residuals;
}

/**}
 * @brief Generate PQ codes for residual vectors using codebook
 *
 * For each subspace, minimize L2 norm between residual vectors and
 * PQ centers to generate codes for residual vectors
 *
 * @tparam T
 * @tparam IdxT
 * @tparam LabelT
 * @param res raft resources
 * @param residuals the residual vectors we're quantizing, size [n_rows, dim]
 * @param pq_codebook the codebook of PQ centers size [dim, 1 << pq_bits]
 * @oaran ps parameters used with vpq_dataset for pq quantization
 * @return device matrix with (packed) codes from vpq, size [n_rows, 1 +ceil((dim / pq_dim *
 * pq_bits) /( 8 * sizeof(LabelT)))]
 */
template <typename T, typename IdxT, typename LabelT>
auto quantize_residuals(raft::resources const& res,
                        raft::device_matrix_view<const T, int64_t> residuals,
                        raft::device_matrix_view<T, uint32_t, raft::row_major> pq_codebook,
                        cuvs::neighbors::vpq_params ps)
  -> raft::device_matrix<uint8_t, IdxT, raft::row_major>
{
  auto dim = residuals.extent(1);

  // Using a single 0 vector for the vq_codebook, since we already have
  // vq centers and computed residuals w.r.t those centers
  auto vq_codebook = raft::make_device_matrix<T, uint32_t, raft::row_major>(res, 1, dim);

  raft::matrix::fill(res, vq_codebook.view(), T(0));

  auto codes = process_and_fill_codes_subspaces<T, IdxT>(
    res, ps, residuals, raft::make_const_mdspan(vq_codebook.view()), pq_codebook);

  return codes;
}

/**
 * @brief Unpack VPQ codes into 1-byte per code
 *
 * VPQ gives codes in a "packed" form. The first 4 bytes give the code for
 * vector quantization, and the remaining bytes the codes for subspace product
 * quantization. In the case of 4 bit PQ, each byte stores codes for 2 subspaces
 * in a packed form.
 *
 * This function unpacks the codes by discarding the VQ code (which we don't need,
 * since we use VPQ only for residual quantization) and (in the case of 4-bit PQ)
 * unpackes the subspace codes into one byte each. This is for interoperability
 * with open source ScaNN, which doesn't pack codes
 *
 * @tparam IdxT
 * @param res raft resources
 * @param unpacked_codes_view matrix of unpacked codes, size  [n_rows, dim / pq_dim]
 * @param codes_view packed codes from vpq, size [n_rows, 1 +ceil((dim / pq_dim * pq_bits) /( 8 *
 * sizeof(LabelT)))]
 * @param pq_bits number of bits used for PQ
 * @param num_subspaces the number of pq_subspaces (dim / pq_dim)
 */
template <typename IdxT>
void unpack_codes(raft::resources const& res,
                  raft::device_matrix_view<uint8_t, IdxT> unpacked_codes_view,
                  raft::device_matrix_view<const uint8_t, IdxT> codes_view,
                  int pq_bits,
                  int num_subspaces)
{
  if (pq_bits == 4) {
    raft::linalg::map_offset(
      res, unpacked_codes_view, [codes_view, num_subspaces] __device__(size_t i) {
        int64_t row_idx             = i / num_subspaces;
        int64_t subspace_idx        = i % num_subspaces;
        int64_t packed_subspace_idx = 4 + subspace_idx / 2;
        uint8_t mask                = subspace_idx % 2;

        uint8_t packed_labels = codes_view(row_idx, packed_subspace_idx);
        uint8_t first         = packed_labels >> 4;
        uint8_t second        = (packed_labels & 15);

        return (mask)*first + (1 - mask) * second;
      });

  } else {
    raft::matrix::slice_coordinates<IdxT> coords(0, 4, codes_view.extent(0), 4 + num_subspaces);
    raft::matrix::slice(res, raft::make_const_mdspan(codes_view), unpacked_codes_view, coords);
  }
}

/**
 * @brief compute eta for AVQ according to Theorem 3.4 in https://arxiv.org/abs/1908.10396
 *
 * @tparam IdxT
 * @param dim the dataset dimension
 * @param sq_norm the squared norm of the vector
 * @param noise_shaping_threshold the threshold T in the Theorem
 * @return eta
 */
template <typename IdxT>
__device__ inline float compute_avq_eta(IdxT dim, const float sq_norm, const float threshold)
{
  return (dim - 1) * (threshold * threshold / sq_norm) / (1 - threshold * threshold / sq_norm);
}

/**
 * @brief helper to convert a float to bfloat16 (represented as int16_t)
 *
 * @param f the float value
 * @return the bflaot16 value (as int16_t)
 */
__device__ inline int16_t float_to_bfloat16(const float& f)
{
  nv_bfloat16 val = __float2bfloat16(f);
  return reinterpret_cast<int16_t&>(val);
}

/**
 * @brief helper to convert a bfloat16 (represented as int16_t) to float
 *
 * @param bf16 the bf16 value (represented as int16_t)
 * @return the float value
 */
__device__ inline float bfloat16_to_float(int16_t& bf16)
{
  nv_bfloat16 nv_bf16 = reinterpret_cast<nv_bfloat16&>(bf16);
  return __bfloat162float(nv_bf16);
}

/**
 * @brief Select the next bfloat16 value to try during coordinate descent
 *
 * Based on the signs of the current residual and quantized value,
 * increment or decrement the quantized value to push residual closer to 0
 *
 * Note that the bfloat16 value is encoded as an int16_t, and the
 * increment/decrement is applied to encoded value. In terms of the float
 * representation, it is the mantissa that is being incremented/decremented,
 * which could carryover to the exponent
 *
 * @param res the float residual
 * @param current the current quantized dimension
 * @return the other possible quantized value
 */
__device__ inline int16_t bfloat16_next_delta(float& res, int16_t& current)
{
  uint32_t res_sign  = ((int32_t)res & (1u << 31) >> 31);
  uint32_t curr_sign = (current & (1 << 15)) >> 15;

  if (res_sign == curr_sign) { return current - 1; }

  return current + 1;
}

template <uint32_t BlockSize, typename IdxT>
__launch_bounds__(BlockSize) RAFT_KERNEL
  quantize_bfloat16_noise_shaped_kernel(raft::device_matrix_view<const float, IdxT> dataset,
                                        raft::device_matrix_view<int16_t, IdxT> bf16_dataset,
                                        raft::device_vector_view<const float, IdxT> sq_norms,
                                        float noise_shaping_threshold)
{
  IdxT row_idx = raft::Pow2<32>::div(IdxT{threadIdx.x} + IdxT{BlockSize} * IdxT{blockIdx.x});

  if (row_idx >= dataset.extent(0)) { return; }

  uint32_t lane_id = raft::Pow2<32>::mod(threadIdx.x);

  IdxT dim = dataset.extent(1);

  // 1 / ||x||
  float inv_norm = 1 / sqrtf(sq_norms[row_idx]);
  float eta      = compute_avq_eta(dim, sq_norms[row_idx], noise_shaping_threshold);

  // < r, x >
  float residual_dot = 0.0;

  for (int i = lane_id; i < dim; i += 32) {
    bf16_dataset(row_idx, i) = float_to_bfloat16(dataset(row_idx, i));

    float residual = dataset(row_idx, i) - bfloat16_to_float(bf16_dataset(row_idx, i));
    residual_dot += dataset(row_idx, i) * residual * inv_norm;
  }

  // reduce  and broadcast residual_dot across warp
  for (uint32_t offset = 16; offset > 0; offset >>= 1) {
    residual_dot += raft::shfl_xor(residual_dot, offset, 32);
  }

  constexpr uint32_t kMaxRounds = 10;

  bool round_changes = true;
  for (int round = 0; round < kMaxRounds && round_changes; round++) {
    round_changes = false;

    for (int i = lane_id; i < dim; i += 32) {
      // coaleseced reads of required data
      float original    = dataset(row_idx, i);
      int16_t quantized = bf16_dataset(row_idx, i);

      float old_residual    = original - bfloat16_to_float(quantized);
      int16_t quantized_new = bfloat16_next_delta(old_residual, quantized);

      float new_residual       = original - bfloat16_to_float(quantized_new);
      float residual_dot_delta = (new_residual - old_residual) * dataset(row_idx, i) * inv_norm;

      float residual_norm_delta = new_residual * new_residual - old_residual * old_residual;

      // we want to compute the change in cost = eta || r_parallel || ^2 + || r_perpendicular|| ^2
      // The change in || r_parallel ||^2 can be written (residual_dot + residual_dot_delta) ^ 2
      // the change in || r_perpendicular || ^2 can be written residual_norm_delta -
      // parallel_norm_delta Thus cost_delta = eta * (residual_dot + residual_dot_delta) ^2 +
      // (residual_norm_delta - (residual_dot + residual_dot_delta)^2 Expanding and simplying,
      // cost_delta = a + b * resdiaul_dot, where a and b are as below. Since only residual_dot is
      // unknown (because updates must be made synchronously) we can compute a and b in parallel
      // across threads in the warp and minimize computation in the update step of the coordinate
      // descent
      float a = residual_norm_delta + (eta - 1) * residual_dot_delta * residual_dot_delta;
      float b = 2 * (eta - 1) * residual_dot_delta;

      // Dim may not be divisible by 32
      // Only synchronize/shuffle for active threads
      int active_threads = std::min<int>(32, dim - i + lane_id);
      int mask           = (1 << active_threads) - 1;

      // Update step for coordinate descent. Compute the cost_delta for
      // each thread, update the quantized value and residual_dot if applicable,
      // then broadcast the new residual dot to the warp
      // AVQ loss the not separable, so we must optimize each dimension separately
      for (int j = 0; j < active_threads; j++) {
        if (lane_id == j) {
          // change in AVQ loss
          float cost_delta = b * residual_dot + a;

          if (cost_delta < 0.0) {
            quantized = quantized_new;
            residual_dot += residual_dot_delta;
            round_changes = true;
          }
        }

        // broadcast new dot product to all lanes
        residual_dot = raft::shfl(residual_dot, j, active_threads, mask);
      }

      // coalesced write of possibly updated quantized values
      bf16_dataset(row_idx, i) = quantized;
    }

    // reduce round_changes across warp
    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
      round_changes |= raft::shfl_xor(round_changes, offset, 32);
    }
  }
}

/**
 * @brief Quantized a float dataset as bfloat16, with noise shaping (AVQ)
 *
 * During quantization we replace each input vector coordinate `f` of type float32 with a bfloat16
 * coordinate `b`. One way to do this would be to simply assign the nearest bfloat16 value to
 * each coordinate. This would be the best way to quantize if we want to minimize the L2
 * distance between the quantized and the original vector.
 *
 * In the AVQ method, we use a different cost function. To minimize that, we consider nearest
 * representable bfloat16 values (`b1`, `b2`) around `f`, and select the one that minimizes the AVQ
 * cost function. In two dimensions we need to consider the four neighboring quantized vectors:
 * b1       b2
 *        f
 * b3      b4
 *
 * In N dimension we will select one the vertices of an N dimensional hypercube as the quantized
 * vector. To find the minimum without enumerating all the combinations, a coordinate descent
 * method is used.
 * @tparam IdxT
 * @param res raft resources
 * @param dataset the dataset (device only) size [n_rows, dim]
 * @param bf16_dataset the quantized dataset (device only) size [n_rows, dim]
 * @param noise_shaping_threshold the threshold for AVQ
 */
template <typename IdxT>
void quantize_bfloat16_noise_shaped(raft::resources const& res,
                                    raft::device_matrix_view<const float, IdxT> dataset,
                                    raft::device_matrix_view<int16_t, IdxT> bf16_dataset,
                                    float noise_shaping_threshold)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res);

  IdxT n_rows = dataset.extent(0);
  auto norms  = raft::make_device_vector<float, IdxT>(res, n_rows);

  // populate square norms
  raft::linalg::norm<raft::linalg::NormType::L2Norm, raft::Apply::ALONG_ROWS>(
    res, dataset, norms.view());

  constexpr int64_t kBlockSize = 256;

  dim3 threads(kBlockSize, 1, 1);
  dim3 blocks(raft::div_rounding_up_safe<ix_t>(n_rows, kBlockSize / 32), 1, 1);

  quantize_bfloat16_noise_shaped_kernel<kBlockSize, IdxT><<<blocks, threads, 0, stream>>>(
    dataset, bf16_dataset, raft::make_const_mdspan(norms.view()), noise_shaping_threshold);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief Quantized a float dataset as bfloat16, with optional noise shaping (AVQ)
 *
 * @tparam IdxT
 * @param res raft resources
 * @param dataset the dataset (device only) size [n_rows, dim]
 * @param bf16_dataset the quantized dataset (device only) size [n_rows, dim]
 * @param noise_shaping_threshold the threshold for AVQ (nan when not using AVQ)
 */
template <typename IdxT>
void quantize_bfloat16(raft::resources const& res,
                       raft::device_matrix_view<const float, IdxT> dataset,
                       raft::device_matrix_view<int16_t, IdxT> bf16_dataset,
                       float noise_shaping_threshold)
{
  if (!std::isnan(noise_shaping_threshold)) {
    quantize_bfloat16_noise_shaped(res, dataset, bf16_dataset, noise_shaping_threshold);
  } else {
    raft::linalg::map(
      res,
      raft::make_device_vector_view<int16_t, int64_t>(bf16_dataset.data_handle(),
                                                      (int64_t)bf16_dataset.size()),
      [] __device__(float x) { return float_to_bfloat16(x); },
      raft::make_const_mdspan(raft::make_device_vector_view<const float, int64_t>(
        dataset.data_handle(), (int64_t)dataset.size())));
  }
}

/**
 * @brief sample dataset vectors/labels and compute their residuals for PQ training
 *
 * @tparam T
 * @tparms LabelT
 * @tparam Accessor
 * @param res raft resources
 * @param random_state state for random generator
 * @param dataset the dataset (host or device), size [n_rows, dim]
 * @param centroids the centers from kmeans training, size [n_clusters, dim]
 * @param labels the idx of the nearest center for each dataset vector, size [n_rows]
 * @param n_samples number of samples
 * @return the sampled residuals for PQ training, size [n_samples, dim]
 */
template <typename T,
          typename LabelT,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
auto sample_training_residuals(
  raft::resources const& res,
  random::RngState random_state,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  raft::device_matrix_view<const T, int64_t> centroids,
  raft::device_vector_view<const LabelT, int64_t> labels,
  int64_t n_samples) -> raft::device_matrix<T, int64_t, raft::row_major>
{
  int64_t n_dim = dataset.extent(1);

  raft::device_vector<int64_t, int64_t> train_indices =
    raft::random::excess_subsample<int64_t, int64_t>(
      res, random_state, dataset.extent(0), n_samples);

  auto trainset = raft::make_device_matrix<T, int64_t>(res, n_samples, n_dim);

  gather_functor<T, int64_t>{}(res,
                               dataset,
                               raft::make_const_mdspan(train_indices.view()),
                               trainset.view(),
                               raft::resource::get_cuda_stream(res));

  // Considering labels as a single column matrix for use in gather
  auto labels_view =
    raft::make_device_matrix_view<const LabelT, int64_t>(labels.data_handle(), labels.extent(0), 1);
  auto trainset_labels = raft::make_device_matrix<LabelT, int64_t>(res, n_samples, 1);

  raft::matrix::gather(
    res, labels_view, raft::make_const_mdspan(train_indices.view()), trainset_labels.view());

  return compute_residuals<T, LabelT>(
    res,
    raft::make_const_mdspan(trainset.view()),
    centroids,
    raft::make_device_vector_view<const LabelT, int64_t>(trainset_labels.data_handle(), n_samples));
}

}  // namespace cuvs::neighbors::experimental::scann::detail

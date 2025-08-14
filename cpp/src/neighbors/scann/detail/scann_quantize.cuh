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

#include "../../detail/vpq_dataset.cuh"
#include <chrono>
#include <cuvs/neighbors/common.hpp>
#include <raft/matrix/gather.cuh>

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
    uint8_t code = cuvs::neighbors::detail::compute_code<kSubWarpSize>(
      dataset, vq_centers, pq_subspace_view, row_ix, j, vq_label);
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

  auto labels = cuvs::neighbors::detail::predict_vq<label_t>(res, dataset, vq_centers);

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

  RAFT_CUDA_TRY(cudaMemsetAsync(vq_codebook.data_handle(),
                                0,
                                vq_codebook.size() * sizeof(T),
                                raft::resource::get_cuda_stream(res)));

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
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
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

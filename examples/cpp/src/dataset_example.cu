/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/dataset.hpp>

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/math.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/stats/meanvar.cuh>

#include <cstdint>
#include <iostream>
#include <utility>
#include <variant>

/** The interface every spec provides, regardless of how the data is stored. */
template <typename DatasetT>
void print_dataset(const char* name, const DatasetT& dataset)
{
  std::cout << name << ": n_rows = " << dataset.n_rows() << ", dim = " << dataset.dim() << '\n';
}

/**
 * A transformation that takes its input by value and hands back the result. A dataset owns its data
 * uniquely, so the caller has to move it in and the transform is free to write over its own input:
 * `d = normalize(res, std::move(d))` allocates only the column statistics.
 */
template <typename T, typename IdxT>
auto normalize(const raft::resources& res, cuvs::device_contiguous_dataset<T, IdxT> data)
  -> cuvs::device_contiguous_dataset<T, IdxT>
{
  auto in   = data.data_const_view();
  auto mean = raft::make_device_vector<T, int64_t>(res, in.extent(1));
  auto var  = raft::make_device_vector<T, int64_t>(res, in.extent(1));
  // Both statistics come out of a single sweep over the data.
  raft::stats::meanvar(res, in, mean.view(), var.view(), false);
  auto mean_view = raft::make_const_mdspan(mean.view());
  auto var_view  = raft::make_const_mdspan(var.view());
  // The linewise op broadcasts the per-column statistics for us; a constant column stays at zero.
  auto standardize = [] __device__(T x, T mean, T var) -> T {
    return var > T{0} ? (x - mean) / raft::sqrt(var) : T{0};
  };

  // Nobody else can observe the buffer, so the transform writes over its own input.
  raft::linalg::matrix_vector_op<raft::Apply::ALONG_ROWS>(
    res, in, mean_view, var_view, data.data_view(), standardize);
  return data;
}

/* The dictionary slot is free for the datasets that have no dictionary. */
static_assert(!cuvs::device_contiguous_dataset<float, uint32_t>::is_compressed);
static_assert(
  std::same_as<cuvs::device_contiguous_dataset<float, uint32_t>::dictionary_type, std::monostate>);
static_assert(
  std::same_as<cuvs::device_csr_dataset<float, uint32_t>::dictionary_type, std::monostate>);
static_assert(std::same_as<cuvs::empty_dataset<float, uint32_t>::dictionary_type, std::monostate>);
static_assert(sizeof(cuvs::device_contiguous_dataset<float, uint32_t>) ==
              sizeof(typename cuvs::device_contiguous_dataset<float, uint32_t>::data_type));
static_assert(cuvs::device_vpq_dataset<float, uint32_t, float>::is_compressed);
/* Ownership is unique, so an owning dataset moves rather than copies, while its view is a handle.
 */
static_assert(!std::is_copy_constructible_v<cuvs::device_contiguous_dataset<float, uint32_t>>);
static_assert(std::is_copy_constructible_v<
              typename cuvs::device_contiguous_dataset<float, uint32_t>::view_type>);
/* The read-only flavor of the view is the same type with a const element type, in which the two
 * data views coincide. A mutable view decays into it, but not the other way round.
 */
static_assert(
  std::same_as<
    typename cuvs::device_contiguous_dataset<float, uint32_t>::const_view_type::data_view_type,
    typename cuvs::device_contiguous_dataset<float, uint32_t>::const_data_view_type>);
static_assert(
  std::convertible_to<typename cuvs::device_contiguous_dataset<float, uint32_t>::view_type,
                      typename cuvs::device_contiguous_dataset<float, uint32_t>::const_view_type>);
static_assert(
  !std::convertible_to<typename cuvs::device_contiguous_dataset<float, uint32_t>::const_view_type,
                       typename cuvs::device_contiguous_dataset<float, uint32_t>::view_type>);

int main()
{
  raft::resources res;

  int64_t n_rows = 8;
  int64_t dim    = 4;

  // An empty dataset carries the dimension only; there is no storage behind it.
  cuvs::empty_dataset<float, uint32_t> empty{{static_cast<uint32_t>(dim)}};
  print_dataset("empty", empty);

  // A host dataset takes ownership of a raft::mdarray of a matching layout and storage policy.
  auto host_data = raft::make_host_matrix<float, int64_t>(res, n_rows, dim);
  for (int64_t i = 0; i < n_rows; i++) {
    for (int64_t j = 0; j < dim; j++) {
      host_data(i, j) = static_cast<float>(i * dim + j);
    }
  }
  cuvs::host_contiguous_dataset<float, uint32_t> host_dataset{std::move(host_data)};
  print_dataset("host contiguous", host_dataset);
  std::cout << "  last element = " << host_dataset.data_view()(n_rows - 1, dim - 1) << '\n';

  // The device variant differs only in the storage policy baked into the spec.
  cuvs::device_contiguous_dataset<float, uint32_t> device_dataset{
    raft::make_device_matrix<float, int64_t>(res, n_rows, dim)};
  print_dataset("device contiguous", device_dataset);

  // The dataset has to be moved into `normalize`, which therefore scales the data in-place.
  raft::linalg::map_offset(res, device_dataset.data_view(), raft::cast_op<float>{});
  const auto* original_data = device_dataset.data_const_view().data_handle();
  device_dataset            = normalize(res, std::move(device_dataset));
  std::cout << "  reused the input buffer: " << std::boolalpha
            << (device_dataset.data_const_view().data_handle() == original_data) << '\n';

  // A view is the non-owning counterpart of a dataset: it answers the same questions, but the data
  // stays with the dataset it was taken from.
  typename cuvs::device_contiguous_dataset<float, uint32_t>::view_type device_view =
    device_dataset.view();
  print_dataset("device contiguous (view)", device_view);
  // The view hands out a read-only mdspan over the same buffer; rebuilding one by hand would mean
  // spelling out the const accessor.
  std::cout << "  views the same buffer: "
            << (device_view.data_const_view().data_handle() == original_data) << '\n';

  // The constness of the dataset picks the flavor of the view: a const dataset hands out the
  // read-only one, whose data view is already an mdspan of `const float`.
  const auto& const_dataset = device_dataset;
  typename cuvs::device_contiguous_dataset<float, uint32_t>::const_view_type const_view =
    const_dataset.view();
  print_dataset("device contiguous (const view)", const_view);
  // ... and a mutable view decays into that same type on its own.
  const_view = device_view;
  std::cout << "  views the same buffer: "
            << (const_view.data_view().data_handle() == original_data) << '\n';

  // The same dataset over a padded layout: the spec, not the dataset, selects the layout.
  using padded_dataset_type = cuvs::device_padded_dataset<float, uint32_t>;
  using padded_data_type    = typename padded_dataset_type::data_type;
  padded_dataset_type padded_dataset{padded_data_type{
    res,
    typename padded_data_type::mapping_type{raft::make_extents<int64_t>(n_rows, dim)},
    typename padded_data_type::container_policy_type{}}};
  print_dataset("device padded", padded_dataset);

  // A sparse dataset: one CSR row per vector, so `dim` is the number of features.
  uint64_t nnz = static_cast<uint64_t>(n_rows) * 2;
  cuvs::device_csr_dataset<float, uint32_t> csr_dataset{
    raft::make_device_csr_matrix<float, int64_t, uint32_t, uint64_t>(
      res, n_rows, static_cast<uint32_t>(dim), nnz)};
  print_dataset("device csr", csr_dataset);
  std::cout << "  nnz = " << csr_dataset.data_view().structure_view().get_nnz() << '\n';

  // The same dataset as a list of (row, column, value) triples.
  cuvs::device_coo_dataset<float, uint32_t> coo_dataset{
    raft::make_device_coo_matrix<float, uint32_t, uint32_t, uint64_t>(
      res, static_cast<uint32_t>(n_rows), static_cast<uint32_t>(dim), nnz)};
  print_dataset("device coo", coo_dataset);

  // A compressed dataset fills the second slot with a dictionary; for VPQ that is the pair of
  // codebooks. Note the dimension of the dataset comes from the VQ codebook here, since the encoded
  // rows know only how many bytes it took to compress a vector.
  using vpq_dataset_type    = cuvs::device_vpq_dataset<float, uint32_t, float>;
  using vpq_dictionary_type = typename vpq_dataset_type::dictionary_type;

  int64_t vq_n_centers = 4;
  int64_t pq_n_centers = 256;
  int64_t pq_len       = 2;
  // One byte per code at pq_bits == 8, prefixed with the inlined VQ label.
  int64_t encoded_row_length = sizeof(uint32_t) + dim / pq_len;

  vpq_dictionary_type codebooks{
    raft::make_device_matrix<float, int64_t>(res, vq_n_centers, dim),
    // A single codebook shared by all the subspaces; `pq_dim` of them would be per-subspace.
    raft::make_device_mdarray<float, int64_t>(
      res, raft::make_extents<int64_t>(1, pq_n_centers, pq_len))};

  vpq_dataset_type vpq_dataset{
    raft::make_device_matrix<uint8_t, int64_t>(res, n_rows, encoded_row_length),
    std::move(codebooks)};
  print_dataset("device vpq", vpq_dataset);
  auto books = vpq_dataset.dictionary_view();
  std::cout << "  vq_n_centers = " << books.vq_n_centers() << ", pq_bits = " << books.pq_bits()
            << ", pq_dim = " << books.pq_dim() << ", per_subspace = " << books.per_subspace()
            << '\n';
  // The data slot is a plain dense matrix of bytes, hence the shared spec.
  std::cout << "  encoded row length = " << vpq_dataset.data_view().extent(1) << '\n';

  // The view of a compressed dataset carries the dictionary along, so it can still tell the
  // dimension that the encoded rows alone do not reveal.
  auto vpq_view = vpq_dataset.view();
  print_dataset("device vpq (view)", vpq_view);
  std::cout << "  views the same codebook: "
            << (vpq_view.dictionary_view().vq_code_book.data_handle() ==
                books.vq_code_book.data_handle())
            << '\n';

  // Scalar quantization keeps one code per component, so the dimension is still a property of the
  // data and the dictionary is only the interval to dequantize into.
  using sq_dataset_type    = cuvs::device_sq_dataset<float, uint32_t, float>;
  using sq_dictionary_type = typename sq_dataset_type::dictionary_type;
  sq_dataset_type sq_dataset{raft::make_device_matrix<int8_t, int64_t>(res, n_rows, dim),
                             sq_dictionary_type{-1.0F, 1.0F}};
  print_dataset("device sq", sq_dataset);
  std::cout << "  range = [" << sq_dataset.dictionary_view().min_ << ", "
            << sq_dataset.dictionary_view().max_ << "]\n";

  return 0;
}

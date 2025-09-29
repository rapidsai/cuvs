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

#include "../test_utils.cuh"
#include <chrono>
#include <cuvs/preprocessing/quantize/product.hpp>
#include <cuvs/stats/trustworthiness_score.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/gather.cuh>
#include <raft/matrix/init.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>

namespace cuvs::preprocessing::quantize::product {

template <typename T>
struct ProductQuantizationInputs {
  int n_samples;                                      // Number of samples in the dataset
  int n_features;                                     // Number of features in the dataset
  uint32_t pq_bits;                                   // PQ bits
  uint32_t pq_dim;                                    // PQ dimension
  cuvs::cluster::kmeans::kmeans_type pq_kmeans_type;  // PQ kmeans type
  uint64_t seed;                                      // Random seed
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const ProductQuantizationInputs<T>& inputs)
{
  return os << "n_samples:" << inputs.n_samples << " n_features:" << inputs.n_features
            << " pq_bits:" << inputs.pq_bits << " pq_dim:" << inputs.pq_dim
            << " pq_kmeans_type:" << (int)inputs.pq_kmeans_type << " seed:" << inputs.seed;
}

template <typename T>
void compare_vectors_l2(const raft::resources& res,
                        T a,
                        T b,
                        double compression_ratio,
                        double eps,
                        bool print_data = false)
{
  auto n_rows = a.extent(0);
  auto dim    = a.extent(1);
  rmm::mr::managed_memory_resource managed_memory;
  auto dist =
    raft::make_device_mdarray<double>(res, &managed_memory, raft::make_extents<uint32_t>(n_rows));
  raft::linalg::map_offset(res, dist.view(), [a, b, dim] __device__(uint32_t i) {
    double d = 0.0f;
    for (uint32_t j = 0; j < dim; j++) {
      double t = static_cast<float>(a(i, j)) - static_cast<float>(b(i, j));
      d += t * t;
    }
    return sqrt(d / double(dim));
  });
  raft::resource::sync_stream(res);
  double mean_dist = 0.0;
  for (uint32_t i = 0; i < n_rows; i++) {
    double d = dist(i);
    if ((print_data && i < 5) || (d > 1.2 * eps * std::pow(2.0, compression_ratio))) {
      auto dim_print = std::min<uint32_t>(dim, 10);
      raft::print_vector("original", a.data_handle() + i * dim, dim_print, std::cout);
      raft::print_vector("reconstructed", b.data_handle() + i * dim, dim_print, std::cout);
      std::cout << "dist: " << d << std::endl;
    }
    mean_dist += d;
    // The theoretical estimate of the error is hard to come up with,
    // the estimate below is based on experimentation + curse of dimensionality
    ASSERT_LE(d, 1.2 * eps * std::pow(2.0, compression_ratio))
      << " (ix = " << i << ", eps = " << eps << ", compression_ratio = " << compression_ratio
      << ")";
  }
  mean_dist /= n_rows;
  if (print_data) {
    std::cout << "mean_dist: " << mean_dist << ", compression_ratio = " << compression_ratio
              << std::endl;
  }
}

/**
 * Copy from cuvs::neighbors::ivf_pq::detail::bitfield_ref_t to be used in this test.
 */
template <uint32_t Bits>
struct bitfield_ref_t {
  static_assert(Bits <= 8 && Bits > 0, "Bit code must fit one byte");
  constexpr static uint8_t kMask = static_cast<uint8_t>((1u << Bits) - 1u);
  uint8_t* ptr;
  uint32_t offset;

  constexpr operator uint8_t()  // NOLINT
  {
    auto pair = static_cast<uint16_t>(ptr[0]);
    if (offset + Bits > 8) { pair |= static_cast<uint16_t>(ptr[1]) << 8; }
    return static_cast<uint8_t>((pair >> offset) & kMask);
  }

  constexpr auto operator=(uint8_t code) -> bitfield_ref_t&
  {
    if (offset + Bits > 8) {
      auto pair = static_cast<uint16_t>(ptr[0]);
      pair |= static_cast<uint16_t>(ptr[1]) << 8;
      pair &= ~(static_cast<uint16_t>(kMask) << offset);
      pair |= static_cast<uint16_t>(code) << offset;
      ptr[0] = static_cast<uint8_t>(raft::Pow2<256>::mod(pair));
      ptr[1] = static_cast<uint8_t>(raft::Pow2<256>::div(pair));
    } else {
      ptr[0] = (ptr[0] & ~(kMask << offset)) | (code << offset);
    }
    return *this;
  }
};

/**
 * Copy from cuvs::neighbors::ivf_pq::detail::bitfield_view_t to be used in this test.
 */
template <uint32_t Bits>
struct bitfield_view_t {
  static_assert(Bits <= 8 && Bits > 0, "Bit code must fit one byte");
  uint8_t* raw;

  constexpr auto operator[](uint32_t i) -> bitfield_ref_t<Bits>
  {
    uint32_t bit_offset = i * Bits;
    return bitfield_ref_t<Bits>{raw + raft::Pow2<8>::div(bit_offset),
                                raft::Pow2<8>::mod(bit_offset)};
  }
};

template <uint32_t BlockSize, uint32_t PqBits, typename DataT, typename MathT, typename IdxT>
__launch_bounds__(BlockSize) RAFT_KERNEL reconstruct_vectors_kernel(
  raft::device_matrix_view<uint8_t, IdxT, raft::row_major> codes,
  raft::device_matrix_view<DataT, IdxT, raft::row_major> dataset,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  const uint32_t pq_dim)
{
  constexpr uint32_t kSubWarpSize = std::min<uint32_t>(raft::WarpSize, 1u << PqBits);
  using subwarp_align             = raft::Pow2<kSubWarpSize>;
  const IdxT row_ix = subwarp_align::div(IdxT{threadIdx.x} + IdxT{BlockSize} * IdxT{blockIdx.x});
  if (row_ix >= codes.extent(0)) { return; }

  auto* out_codes_ptr = &codes(row_ix, 0);
  bitfield_view_t<PqBits> code_view{out_codes_ptr};
  for (uint32_t j = 0; j < pq_dim; j++) {
    uint8_t code = code_view[j];
    for (uint32_t k = 0; k < pq_centers.extent(1); k++) {
      dataset(row_ix, j * pq_centers.extent(1) + k) = pq_centers(code, k);
    }
  }
}

template <typename DataT, typename MathT, typename IdxT>
auto reconstruct_vectors(
  const raft::resources& res,
  const params& params,
  raft::device_matrix_view<uint8_t, IdxT, raft::row_major> codes,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  raft::device_matrix_view<DataT, IdxT, raft::row_major> out_vectors)
{
  using data_t = DataT;
  using ix_t   = IdxT;

  const ix_t n_rows       = out_vectors.extent(0);
  const ix_t dim          = out_vectors.extent(1);
  const ix_t pq_dim       = params.pq_dim;
  const ix_t pq_bits      = params.pq_bits;
  const ix_t pq_n_centers = ix_t{1} << pq_bits;
  const ix_t codes_rowlen = raft::div_rounding_up_safe<ix_t>(pq_dim * pq_bits, 8);

  auto stream = raft::resource::get_cuda_stream(res);

  constexpr ix_t kBlockSize  = 256;
  const ix_t threads_per_vec = std::min<ix_t>(raft::WarpSize, pq_n_centers);
  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [](uint32_t pq_bits) {
    switch (pq_bits) {
      case 4: return reconstruct_vectors_kernel<kBlockSize, 4, data_t, MathT, IdxT>;
      case 5: return reconstruct_vectors_kernel<kBlockSize, 5, data_t, MathT, IdxT>;
      case 6: return reconstruct_vectors_kernel<kBlockSize, 6, data_t, MathT, IdxT>;
      case 7: return reconstruct_vectors_kernel<kBlockSize, 7, data_t, MathT, IdxT>;
      case 8: return reconstruct_vectors_kernel<kBlockSize, 8, data_t, MathT, IdxT>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }(pq_bits);
  dim3 blocks(raft::div_rounding_up_safe<ix_t>(n_rows, kBlockSize / threads_per_vec), 1, 1);
  kernel<<<blocks, threads, 0, stream>>>(codes, out_vectors, pq_centers, pq_dim);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  return codes;
}

template <typename T>
class ProductQuantizationTest : public ::testing::TestWithParam<ProductQuantizationInputs<T>> {
 public:
  ProductQuantizationTest()
    : params_(::testing::TestWithParam<ProductQuantizationInputs<T>>::GetParam()),
      stream(raft::resource::get_cuda_stream(handle)),
      dataset_(raft::make_device_matrix<T, int64_t, raft::row_major>(
        handle, params_.n_samples, params_.n_features)),
      labels_(raft::make_device_vector<int64_t, int64_t>(handle, params_.n_samples)),
      n_samples_(params_.n_samples),
      n_features_(params_.n_features)
  {
  }

 protected:
  void SetUp() override
  {
    if constexpr (std::is_same_v<T, half>) {
      raft::random::RngState r(params_.seed);
      raft::random::uniform(
        handle, r, dataset_.data_handle(), n_samples_ * n_features_, T(0.1), T(2.0));
    } else {
      raft::random::make_blobs<T, int64_t, raft::row_major>(
        handle,
        dataset_.view(),
        labels_.view(),
        5,                      // Number of clusters
        std::nullopt,           // Generate random centers
        std::nullopt,           // Use scalar std
        static_cast<T>(1.0),    // Cluster std
        true,                   // Shuffle
        static_cast<T>(-10.0),  // Center box min
        static_cast<T>(10.0),   // Center box max
        params_.seed);          // Random seed
    }
  }

  void TearDown() override {}

  void check_reconstruction(const cuvs::preprocessing::quantize::product::quantizer<T>& quantizer,
                            raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
                            double compression_ratio,
                            uint32_t n_take)
  {
    auto dim = n_features_;

    if (n_take == 0) { return; }

    n_take = std::min<uint32_t>(n_take, codes.extent(0));

    auto rec_data  = raft::make_device_matrix<T, int64_t>(handle, n_take, dim);
    auto orig_data = raft::make_device_matrix_view<T, int64_t>(dataset_.data_handle(), n_take, dim);

    reconstruct_vectors(handle,
                        quantizer.params_quantizer,
                        codes,
                        raft::make_const_mdspan(quantizer.vpq_codebooks.pq_code_book.view()),
                        rec_data.view());

    compare_vectors_l2(handle, orig_data, rec_data.view(), compression_ratio, 0.04, false);
  }

  void testProductQuantizationFromDataset()
  {
    config_ = {params_.pq_bits, params_.pq_dim, 1, 25, 0, 0, params_.pq_kmeans_type};
    raft::resource::sync_stream(handle);
    auto pq = train(handle, config_, dataset_.view());

    auto n_encoded_cols =
      raft::div_rounding_up_safe(pq.params_quantizer.pq_dim * pq.params_quantizer.pq_bits, 8u);
    auto d_quantized_output =
      raft::make_device_matrix<uint8_t, int64_t>(handle, n_samples_, n_encoded_cols);

    transform(handle, pq, dataset_.view(), d_quantized_output.view());

    // 1. Verify that the quantized output is not all zeros or NaNs
    {
      auto h_quantized_output =
        raft::make_host_matrix<uint8_t, int, raft::col_major>(n_samples_, n_encoded_cols);
      raft::update_host(h_quantized_output.data_handle(),
                        d_quantized_output.data_handle(),
                        n_samples_ * n_encoded_cols,
                        stream);
      raft::resource::sync_stream(handle, stream);

      bool all_zeros = true;
      bool has_nan   = false;

      for (int i = 0; i < h_quantized_output.extent(0) * h_quantized_output.extent(1); i++) {
        if (h_quantized_output.data_handle()[i] != 0) { all_zeros = false; }
        if (std::isnan(h_quantized_output.data_handle()[i])) {
          has_nan = true;
          break;
        }
      }
      ASSERT_FALSE(all_zeros) << "Quantized output contains all zeros";
      ASSERT_FALSE(has_nan) << "Quantized output contains NaN values";
    }

    // 2. Verify that the quantized output is consistent with the input
    double compression_ratio =
      static_cast<double>(n_features_ * 8) /
      static_cast<double>(pq.params_quantizer.pq_dim * pq.params_quantizer.pq_bits);

    check_reconstruction(pq, d_quantized_output.view(), compression_ratio, 500);
  }

 private:
  raft::resources handle;
  cudaStream_t stream;

  ProductQuantizationInputs<T> params_;
  int n_samples_;
  int n_features_;

  raft::device_matrix<T, int64_t, raft::row_major> dataset_;
  raft::device_vector<int64_t, int64_t, raft::row_major> labels_;

  params config_;
};

// Define test cases with different parameters
template <typename T>
const std::vector<ProductQuantizationInputs<T>> inputs = {
  // Small dataset
  {100, 32, 4, 4, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 42ULL},

  // Small dataset with bigger dims
  {100, 90, 6, 10, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 42ULL},
  {100, 91, 7, 7, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 42ULL},

  // Medium dataset
  {500, 40, 5, 8, cuvs::cluster::kmeans::kmeans_type::KMeans, 42ULL},
  {500, 60, 6, 6, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 42ULL},

  // Larger dataset
  {1000, 40, 4, 10, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 42ULL},
  {3000, 1024, 4, 32, cuvs::cluster::kmeans::kmeans_type::KMeans, 42ULL},
  {1000, 2048, 4, 128, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 42ULL},

  // Benchmark datasets
  {50000, 1024, 8, 128, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 42ULL},
  {50000, 2048, 8, 128, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 42ULL}};

typedef ProductQuantizationTest<float> ProductQuantizationTestF;
TEST_P(ProductQuantizationTestF, Result) { this->testProductQuantizationFromDataset(); }

typedef ProductQuantizationTest<double> ProductQuantizationTestD;
TEST_P(ProductQuantizationTestD, Result) { this->testProductQuantizationFromDataset(); }

INSTANTIATE_TEST_CASE_P(ProductQuantizationTests,
                        ProductQuantizationTestF,
                        ::testing::ValuesIn(inputs<float>));
INSTANTIATE_TEST_CASE_P(ProductQuantizationTests,
                        ProductQuantizationTestD,
                        ::testing::ValuesIn(inputs<double>));

}  // namespace cuvs::preprocessing::quantize::product

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
  int n_queries;                                      // Number of queries in the dataset
  uint32_t pq_bits;                                   // PQ bits
  uint32_t pq_dim;                                    // PQ dimension
  cuvs::cluster::kmeans::kmeans_type pq_kmeans_type;  // PQ kmeans type
  uint32_t n_vq_centers;                              // Number of VQ centers
  uint64_t seed;                                      // Random seed
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const ProductQuantizationInputs<T>& inputs)
{
  return os << "n_samples:" << inputs.n_samples << " n_features:" << inputs.n_features
            << " n_queries:" << inputs.n_queries << " pq_bits:" << inputs.pq_bits
            << " pq_dim:" << inputs.pq_dim << " pq_kmeans_type:" << (int)inputs.pq_kmeans_type
            << " n_vq_centers:" << inputs.n_vq_centers << " seed:" << inputs.seed;
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

template <typename T>
class ProductQuantizationTest : public ::testing::TestWithParam<ProductQuantizationInputs<T>> {
 public:
  ProductQuantizationTest()
    : params_(::testing::TestWithParam<ProductQuantizationInputs<T>>::GetParam()),
      stream(raft::resource::get_cuda_stream(handle)),
      dataset_(raft::make_device_matrix<T, int64_t, raft::row_major>(
        handle, params_.n_samples, params_.n_features)),
      queries_(raft::make_device_matrix<T, int64_t, raft::row_major>(
        handle, params_.n_queries, params_.n_features)),
      labels_(raft::make_device_vector<int64_t, int64_t>(handle, params_.n_samples)),
      n_samples_(params_.n_samples),
      n_features_(params_.n_features),
      n_queries_(params_.n_queries)
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
      raft::random::RngState r(params_.seed);
      raft::random::uniform(
        handle, r, queries_.data_handle(), n_queries_ * n_features_, T(-10.0), T(10.0));
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
    std::optional<raft::device_matrix_view<const T, uint32_t, raft::row_major>> vq_centers_opt =
      std::nullopt;
    if (params_.n_vq_centers > 1) {
      vq_centers_opt = raft::make_const_mdspan(quantizer.vpq_codebooks.vq_code_book.view());
    }

    inverse_transform(handle,
                      quantizer,
                      raft::device_matrix_view<const uint8_t, int64_t>(
                        codes.data_handle(), n_take, codes.extent(1)),
                      rec_data.view());

    compare_vectors_l2(handle, orig_data, rec_data.view(), compression_ratio, 0.04, false);
  }

  void testProductQuantizationFromDataset()
  {
    using LabelT = uint32_t;
    config_      = {
      params_.pq_bits, params_.pq_dim, params_.n_vq_centers, 25, 0, 0, params_.pq_kmeans_type};
    raft::resource::sync_stream(handle);
    auto pq = train(handle, config_, dataset_.view());

    auto n_encoded_cols =
      sizeof(LabelT) *
      (1 + raft::div_rounding_up_safe<int64_t>(
             pq.params_quantizer.pq_dim * pq.params_quantizer.pq_bits, 8 * sizeof(LabelT)));
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
  int n_queries_;

  raft::device_matrix<T, int64_t, raft::row_major> dataset_;
  raft::device_vector<int64_t, int64_t, raft::row_major> labels_;
  raft::device_matrix<T, int64_t, raft::row_major> queries_;
  params config_;
};

// Define test cases with different parameters
template <typename T>
const std::vector<ProductQuantizationInputs<T>> inputs = {
  // Small dataset
  {100, 32, 10, 4, 4, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 1, 42ULL},

  // Small dataset with bigger dims
  {100, 90, 10, 6, 10, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 1, 42ULL},
  {100, 91, 50, 7, 7, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 3, 42ULL},

  // Medium dataset
  {500, 40, 100, 5, 8, cuvs::cluster::kmeans::kmeans_type::KMeans, 3, 42ULL},
  {500, 60, 100, 6, 6, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 4, 42ULL},

  // Larger dataset
  {1000, 40, 100, 4, 10, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 5, 42ULL},
  {3000, 1024, 100, 4, 32, cuvs::cluster::kmeans::kmeans_type::KMeans, 10, 42ULL},
  {1000, 2048, 100, 4, 128, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 1, 42ULL},

  // Benchmark datasets
  {50000, 1024, 100, 8, 128, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 1, 42ULL},
  {50000, 2048, 100, 8, 128, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 5, 42ULL}};

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

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
#include <cuvs/preprocessing/quantize/product.hpp>
#include <cuvs/stats/trustworthiness_score.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/gather.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>

namespace cuvs::preprocessing::quantize::product {

template <typename T>
struct ProductQuantizationInputs {
  int n_samples;                                        // Number of samples in the dataset
  int n_features;                                       // Number of features in the dataset
  int pq_bits;                                          // PQ bits
  int pq_dim;                                           // PQ dimension
  cuvs::neighbors::ivf_pq::codebook_gen codebook_kind;  // Codebook generation method
  bool force_random_rotation;                           // Whether to force random rotation
  uint32_t max_train_points_per_pq_code;                // Max training points per PQ code
  uint64_t seed;                                        // Random seed
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const ProductQuantizationInputs<T>& inputs)
{
  return os << "n_samples:" << inputs.n_samples << " n_features:" << inputs.n_features
            << " pq_bits:" << inputs.pq_bits << " pq_dim:" << inputs.pq_dim
            << " codebook_kind:" << static_cast<int>(inputs.codebook_kind)
            << " force_random_rotation:" << inputs.force_random_rotation
            << " max_train_points_per_pq_code:" << inputs.max_train_points_per_pq_code
            << " seed:" << inputs.seed;
}

template <typename T>
struct config {};

template <>
struct config<double> {
  using value_t                    = double;
  static constexpr double kDivisor = 1.0;
};
template <>
struct config<float> {
  using value_t                    = float;
  static constexpr double kDivisor = 1.0;
};
template <>
struct config<half> {
  using value_t                    = half;
  static constexpr double kDivisor = 1.0;
};
template <>
struct config<uint8_t> {
  using value_t                    = uint32_t;
  static constexpr double kDivisor = 256.0;
};
template <>
struct config<int8_t> {
  using value_t                    = int32_t;
  static constexpr double kDivisor = 128.0;
};

template <typename T>
struct mapping {
  /**
   * @defgroup
   * @brief Cast and possibly scale a value of the source type `S` to the target type `T`.
   *
   * @tparam S source type
   * @param x source value
   * @{
   */
  template <typename S>
  HDI constexpr auto operator()(const S& x) const -> std::enable_if_t<std::is_same_v<S, T>, T>
  {
    return x;
  };

  template <typename S>
  HDI constexpr auto operator()(const S& x) const -> std::enable_if_t<!std::is_same_v<S, T>, T>
  {
    constexpr double kMult = config<T>::kDivisor / config<S>::kDivisor;
    if constexpr (std::is_floating_point_v<S>) { return static_cast<T>(x * static_cast<S>(kMult)); }
    if constexpr (std::is_floating_point_v<T>) { return static_cast<T>(x) * static_cast<T>(kMult); }
    return static_cast<T>(static_cast<float>(x) * static_cast<float>(kMult));
  };
  /** @} */
};

template <typename T>
void compare_vectors_l2(const raft::resources& res,
                        T a,
                        T b,
                        uint32_t label,
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
    mapping<float> f{};
    double d = 0.0f;
    for (uint32_t j = 0; j < dim; j++) {
      double t = f(a(i, j)) - f(b(i, j));
      d += t * t;
    }
    return sqrt(d / double(dim));
  });
  raft::resource::sync_stream(res);
  for (uint32_t i = 0; i < n_rows; i++) {
    double d = dist(i);
    // The theoretical estimate of the error is hard to come up with,
    // the estimate below is based on experimentation + curse of dimensionality
    ASSERT_LE(d, 1.2 * eps * std::pow(2.0, compression_ratio))
      << " (label = " << label << ", ix = " << i << ", eps = " << eps << ")";
    if (print_data && i < 5) {
      auto dim_print = std::min<uint32_t>(dim, 10);
      raft::print_vector("original", a.data_handle() + i * dim, dim_print, std::cout);
      raft::print_vector("reconstructed", b.data_handle() + i * dim, dim_print, std::cout);
      std::cout << "dist: " << d << std::endl;
    }
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
      labels_(raft::make_device_vector<int64_t, int64_t>(handle, params_.n_samples)),
      pq_dim_(params_.pq_dim),
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

  void check_reconstruction(const cuvs::neighbors::ivf_pq::index<int64_t>& index,
                            double compression_ratio,
                            uint32_t label,
                            uint32_t n_take,
                            uint32_t n_skip)
  {
    // the original data cannot be reconstructed since the dataset was normalized
    if (index.metric() == cuvs::distance::DistanceType::CosineExpanded) { return; }
    auto& rec_list = index.lists()[label];
    // If the data is unbalanced the list might be empty, which is actually nullptr
    if (!rec_list) { return; }
    auto dim = index.dim();
    n_take   = std::min<uint32_t>(n_take, rec_list->size.load());
    n_skip   = std::min<uint32_t>(n_skip, rec_list->size.load() - n_take);

    if (n_take == 0) { return; }

    auto rec_data  = raft::make_device_matrix<T>(handle, n_take, dim);
    auto orig_data = raft::make_device_matrix<T>(handle, n_take, dim);

    cuvs::neighbors::ivf_pq::helpers::codepacker::reconstruct_list_data(
      handle, index, rec_data.view(), label, n_skip);

    raft::matrix::gather(dataset_.data_handle(),
                         int64_t{dim},
                         int64_t{n_take},
                         rec_list->indices.data_handle() + n_skip,
                         int64_t{n_take},
                         orig_data.data_handle(),
                         raft::resource::get_cuda_stream(handle));
    compare_vectors_l2(
      handle, rec_data.view(), orig_data.view(), label, compression_ratio, 0.04, false);
  }

  void testProductQuantizationFromDataset()
  {
    auto pq = train(handle, config_, dataset_.view());
    auto d_quantized_output =
      raft::make_device_matrix<uint8_t, int64_t>(handle, n_samples_, pq_dim_);
    transform(handle, pq, dataset_.view(), d_quantized_output.view());

    // 1. Verify that the quantized output is not all zeros or NaNs
    auto h_quantized_output =
      raft::make_host_matrix<uint8_t, int, raft::col_major>(n_samples_, pq_dim_);
    raft::update_host(h_quantized_output.data_handle(),
                      d_quantized_output.data_handle(),
                      n_samples_ * pq_dim_,
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

    // 3. Verify that the quantized output is consistent with the input
    double compression_ratio =
      static_cast<double>(n_features_ * 8) / static_cast<double>(pq_dim_ * config_.pq_bits);
    std::optional<raft::device_vector_view<const int64_t, int64_t>> indices_view_opt = std::nullopt;
    auto reconstruction_index = cuvs::neighbors::ivf_pq::extend(
      handle, raft::make_const_mdspan(dataset_.view()), indices_view_opt, pq.pq_index);
    for (uint32_t i = 0; i < reconstruction_index.n_lists(); i++) {
      check_reconstruction(reconstruction_index, compression_ratio, i, 800, 0);
    }
  }

 private:
  raft::resources handle;
  cudaStream_t stream;

  ProductQuantizationInputs<T> params_;
  int n_samples_;
  int n_features_;
  int pq_dim_;

  raft::device_matrix<T, int64_t, raft::row_major> dataset_;
  raft::device_vector<int64_t, int64_t, raft::row_major> labels_;

  params config_;
};

// Define test cases with different parameters
template <typename T>
const std::vector<ProductQuantizationInputs<T>> inputs = {
  // Small dataset
  {100, 30, 4, 4, cuvs::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE, false, 10, 42ULL},

  // Small dataset with bigger dims
  {100, 90, 6, 8, cuvs::neighbors::ivf_pq::codebook_gen::PER_CLUSTER, true, 10, 42ULL},

  // Medium dataset
  {500, 40, 5, 8, cuvs::neighbors::ivf_pq::codebook_gen::PER_CLUSTER, false, 10, 42ULL},
  {500, 60, 7, 8, cuvs::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE, false, 10, 42ULL},

  // Larger dataset
  {1000, 40, 4, 4, cuvs::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE, true, 20, 42ULL},
  {3000, 1024, 4, 20, cuvs::neighbors::ivf_pq::codebook_gen::PER_CLUSTER, false, 10, 42ULL},
  {1000, 2048, 4, 20, cuvs::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE, false, 10, 42ULL},

  // Benchmark datasets
  //{500000, 2048, 8, 20, cuvs::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE, false, 1000, 42ULL},
  //{500000, 2048, 8, 20, cuvs::neighbors::ivf_pq::codebook_gen::PER_CLUSTER, false, 1000, 42ULL}
};

typedef ProductQuantizationTest<float> ProductQuantizationTestF;
TEST_P(ProductQuantizationTestF, Result) { this->testProductQuantizationFromDataset(); }

INSTANTIATE_TEST_CASE_P(ProductQuantizationTests,
                        ProductQuantizationTestF,
                        ::testing::ValuesIn(inputs<float>));

}  // namespace cuvs::preprocessing::quantize::product

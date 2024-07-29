/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include "./knn_utils.cuh"
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/selection/select_k.hpp>

#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/init.cuh>

namespace cuvs::neighbors::brute_force {
struct KNNInputs {
  std::vector<std::vector<float>> input;
  int k;
  std::vector<int> labels;
};

template <typename IdxT>
RAFT_KERNEL build_actual_output(
  int* output, int n_rows, int k, const int* idx_labels, const IdxT* indices)
{
  int element = threadIdx.x + blockDim.x * blockIdx.x;
  if (element >= n_rows * k) return;

  output[element] = idx_labels[indices[element]];
}

RAFT_KERNEL build_expected_output(int* output, int n_rows, int k, const int* labels)
{
  int row = threadIdx.x + blockDim.x * blockIdx.x;
  if (row >= n_rows) return;

  int cur_label = labels[row];
  for (int i = 0; i < k; i++) {
    output[row * k + i] = cur_label;
  }
}

template <typename T, typename IdxT>
class KNNTest : public ::testing::TestWithParam<KNNInputs> {
 public:
  KNNTest()
    : params_(::testing::TestWithParam<KNNInputs>::GetParam()),
      stream(raft::resource::get_cuda_stream(handle)),
      actual_labels_(0, stream),
      expected_labels_(0, stream),
      input_(0, stream),
      search_data_(0, stream),
      indices_(0, stream),
      distances_(0, stream),
      search_labels_(0, stream)
  {
  }

 protected:
  void testBruteForce()
  {
    // #if (RAFT_ACTIVE_LEVEL >= RAFT_LEVEL_DEBUG)
    raft::print_device_vector("Input array: ", input_.data(), rows_ * cols_, std::cout);
    std::cout << "K: " << k_ << std::endl;
    raft::print_device_vector("Labels array: ", search_labels_.data(), rows_, std::cout);
    // #endif

    auto index = raft::make_device_matrix_view<const T, IdxT, raft::row_major>(
      (const T*)(input_.data()), rows_, cols_);
    auto search = raft::make_device_matrix_view<const T, IdxT, raft::row_major>(
      (const T*)(search_data_.data()), rows_, cols_);
    auto indices =
      raft::make_device_matrix_view<IdxT, IdxT, raft::row_major>(indices_.data(), rows_, k_);
    auto distances =
      raft::make_device_matrix_view<T, IdxT, raft::row_major>(distances_.data(), rows_, k_);

    auto metric = cuvs::distance::DistanceType::L2Unexpanded;
    auto idx    = cuvs::neighbors::brute_force::build(handle, index, metric);
    cuvs::neighbors::brute_force::search(handle, idx, search, indices, distances, std::nullopt);

    build_actual_output<<<raft::ceildiv(rows_ * k_, 32), 32, 0, stream>>>(
      actual_labels_.data(), rows_, k_, search_labels_.data(), indices_.data());

    build_expected_output<<<raft::ceildiv(rows_ * k_, 32), 32, 0, stream>>>(
      expected_labels_.data(), rows_, k_, search_labels_.data());

    ASSERT_TRUE(devArrMatch(
      expected_labels_.data(), actual_labels_.data(), rows_ * k_, cuvs::Compare<int>(), stream));
  }

  void SetUp() override
  {
    rows_ = params_.input.size();
    cols_ = params_.input[0].size();
    k_    = params_.k;

    actual_labels_.resize(rows_ * k_, stream);
    expected_labels_.resize(rows_ * k_, stream);
    input_.resize(rows_ * cols_, stream);
    search_data_.resize(rows_ * cols_, stream);
    indices_.resize(rows_ * k_, stream);
    distances_.resize(rows_ * k_, stream);
    search_labels_.resize(rows_, stream);

    RAFT_CUDA_TRY(
      cudaMemsetAsync(actual_labels_.data(), 0, actual_labels_.size() * sizeof(int), stream));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(expected_labels_.data(), 0, expected_labels_.size() * sizeof(int), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(input_.data(), 0, input_.size() * sizeof(float), stream));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(search_data_.data(), 0, search_data_.size() * sizeof(float), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(indices_.data(), 0, indices_.size() * sizeof(IdxT), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(distances_.data(), 0, distances_.size() * sizeof(float), stream));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(search_labels_.data(), 0, search_labels_.size() * sizeof(int), stream));

    std::vector<float> row_major_input;
    for (std::size_t i = 0; i < params_.input.size(); ++i) {
      for (std::size_t j = 0; j < params_.input[i].size(); ++j) {
        row_major_input.push_back(params_.input[i][j]);
      }
    }
    rmm::device_buffer input_d =
      rmm::device_buffer(row_major_input.data(), row_major_input.size() * sizeof(float), stream);
    float* input_ptr = static_cast<float*>(input_d.data());

    rmm::device_buffer labels_d =
      rmm::device_buffer(params_.labels.data(), params_.labels.size() * sizeof(int), stream);
    int* labels_ptr = static_cast<int*>(labels_d.data());

    raft::copy(input_.data(), input_ptr, rows_ * cols_, stream);
    raft::copy(search_data_.data(), input_ptr, rows_ * cols_, stream);
    raft::copy(search_labels_.data(), labels_ptr, rows_, stream);
    raft::resource::sync_stream(handle, stream);
  }

 private:
  raft::resources handle;
  cudaStream_t stream;

  KNNInputs params_;
  int rows_;
  int cols_;
  rmm::device_uvector<float> input_;
  rmm::device_uvector<float> search_data_;
  rmm::device_uvector<IdxT> indices_;
  rmm::device_uvector<float> distances_;
  int k_;

  rmm::device_uvector<int> search_labels_;
  rmm::device_uvector<int> actual_labels_;
  rmm::device_uvector<int> expected_labels_;
};

const std::vector<KNNInputs> inputs = {
  // 2D
  {{
     {2.7810836, 2.550537003},
     {1.465489372, 2.362125076},
     {3.396561688, 4.400293529},
     {1.38807019, 1.850220317},
     {3.06407232, 3.005305973},
     {7.627531214, 2.759262235},
     {5.332441248, 2.088626775},
     {6.922596716, 1.77106367},
     {8.675418651, -0.242068655},
     {7.673756466, 3.508563011},
   },
   2,
   {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}}};

typedef KNNTest<float, int64_t> KNNTestFint64_t;
TEST_P(KNNTestFint64_t, BruteForce) { this->testBruteForce(); }

INSTANTIATE_TEST_CASE_P(KNNTest, KNNTestFint64_t, ::testing::ValuesIn(inputs));

// Also test with larger random inputs, including col-major inputs
struct RandomKNNInputs {
  int num_queries;
  int num_db_vecs;
  int dim;
  int k;
  cuvs::distance::DistanceType metric;
  bool row_major;
};

std::ostream& operator<<(std::ostream& os, const RandomKNNInputs& input)
{
  return os << "num_queries:" << input.num_queries << " num_vecs:" << input.num_db_vecs
            << " dim:" << input.dim << " k:" << input.k
            << " metric:" << cuvs::neighbors::print_metric{input.metric}
            << " row_major:" << input.row_major;
}

template <typename T>
class RandomBruteForceKNNTest : public ::testing::TestWithParam<RandomKNNInputs> {
 public:
  RandomBruteForceKNNTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      params_(::testing::TestWithParam<RandomKNNInputs>::GetParam()),
      database(params_.num_db_vecs * params_.dim, stream_),
      search_queries(params_.num_queries * params_.dim, stream_),
      cuvs_indices_(params_.num_queries * params_.k, stream_),
      cuvs_distances_(params_.num_queries * params_.k, stream_),
      ref_indices_(params_.num_queries * params_.k, stream_),
      ref_distances_(params_.num_queries * params_.k, stream_)
  {
    raft::matrix::fill(
      handle_,
      raft::make_device_matrix_view(database.data(), params_.num_db_vecs, params_.dim),
      T{0.0});
    raft::matrix::fill(
      handle_,
      raft::make_device_matrix_view(search_queries.data(), params_.num_queries, params_.dim),
      T{0.0});
    raft::matrix::fill(
      handle_,
      raft::make_device_matrix_view(cuvs_distances_.data(), params_.num_queries, params_.k),
      T{0.0});
    raft::matrix::fill(
      handle_,
      raft::make_device_matrix_view(ref_distances_.data(), params_.num_queries, params_.k),
      T{0.0});
  }

 protected:
  void testBruteForce()
  {
    float metric_arg = 3.0;

    // calculate the naive knn, by calculating the full pairwise distances and doing a k-select
    rmm::device_uvector<T> temp_distances(num_db_vecs * num_queries, stream_);
    rmm::device_uvector<char> workspace(0, stream_);

    auto temp_dist = temp_distances.data();
    rmm::device_uvector<T> temp_row_major_dist(num_db_vecs * num_queries, stream_);

    if (params_.row_major) {
      distance::pairwise_distance(
        handle_,
        raft::make_device_matrix_view<const T, int64_t>(
          search_queries.data(), params_.num_queries, params_.dim),
        raft::make_device_matrix_view<const T, int64_t>(
          database.data(), params_.num_db_vecs, params_.dim),
        raft::make_device_matrix_view<T, int64_t>(temp_distances.data(), num_queries, num_db_vecs),
        metric,
        metric_arg);

    } else {
      distance::pairwise_distance(handle_,
                                  raft::make_device_matrix_view<const T, int64_t, raft::col_major>(
                                    search_queries.data(), params_.num_queries, params_.dim),
                                  raft::make_device_matrix_view<const T, int64_t, raft::col_major>(
                                    database.data(), params_.num_db_vecs, params_.dim),
                                  raft::make_device_matrix_view<T, int64_t, raft::col_major>(
                                    temp_distances.data(), num_queries, num_db_vecs),
                                  metric,
                                  metric_arg);

      // the pairwisse_distance call assumes that the inputs and outputs are all either row-major
      // or col-major - meaning we have to transpose the output back for col-major queries
      // for comparison
      raft::linalg::transpose(
        handle_, temp_dist, temp_row_major_dist.data(), num_queries, num_db_vecs, stream_);
      temp_dist = temp_row_major_dist.data();
    }

    cuvs::selection::select_k(
      handle_,
      raft::make_device_matrix_view<const T, int64_t>(temp_dist, num_queries, num_db_vecs),
      std::nullopt,
      raft::make_device_matrix_view(ref_distances_.data(), params_.num_queries, params_.k),
      raft::make_device_matrix_view(ref_indices_.data(), params_.num_queries, params_.k),
      cuvs::distance::is_min_close(metric),
      true);

    auto indices = raft::make_device_matrix_view<int64_t, int64_t>(
      cuvs_indices_.data(), params_.num_queries, params_.k);
    auto distances = raft::make_device_matrix_view<T, int64_t>(
      cuvs_distances_.data(), params_.num_queries, params_.k);

    if (params_.row_major) {
      auto idx =
        cuvs::neighbors::brute_force::build(handle_,
                                            raft::make_device_matrix_view<const T, int64_t>(
                                              database.data(), params_.num_db_vecs, params_.dim),
                                            metric,
                                            metric_arg);

      cuvs::neighbors::brute_force::search(
        handle_,
        idx,
        raft::make_device_matrix_view<const T, int64_t>(
          search_queries.data(), params_.num_queries, params_.dim),
        indices,
        distances,
        std::nullopt);
    } else {
      auto idx = cuvs::neighbors::brute_force::build(
        handle_,
        raft::make_device_matrix_view<const T, int64_t, raft::col_major>(
          database.data(), params_.num_db_vecs, params_.dim),
        metric,
        metric_arg);

      cuvs::neighbors::brute_force::search(
        handle_,
        idx,
        raft::make_device_matrix_view<const T, int64_t, raft::col_major>(
          search_queries.data(), params_.num_queries, params_.dim),
        indices,
        distances,
        std::nullopt);
    }

    ASSERT_TRUE(cuvs::neighbors::devArrMatchKnnPair(ref_indices_.data(),
                                                    cuvs_indices_.data(),
                                                    ref_distances_.data(),
                                                    cuvs_distances_.data(),
                                                    num_queries,
                                                    k_,
                                                    float(0.001),
                                                    stream_,
                                                    true));
  }

  void SetUp() override
  {
    num_queries = params_.num_queries;
    num_db_vecs = params_.num_db_vecs;
    dim         = params_.dim;
    k_          = params_.k;
    metric      = params_.metric;

    unsigned long long int seed = 1234ULL;
    raft::random::RngState r(seed);

    // JensenShannon distance requires positive values
    T min_val = metric == cuvs::distance::DistanceType::JensenShannon ? T(0.0) : T(-1.0);
    uniform(handle_, r, database.data(), num_db_vecs * dim, min_val, T(1.0));
    uniform(handle_, r, search_queries.data(), num_queries * dim, min_val, T(1.0));
  }

 private:
  raft::resources handle_;
  cudaStream_t stream_ = 0;
  RandomKNNInputs params_;
  int num_queries;
  int num_db_vecs;
  int dim;
  rmm::device_uvector<T> database;
  rmm::device_uvector<T> search_queries;
  rmm::device_uvector<int64_t> cuvs_indices_;
  rmm::device_uvector<T> cuvs_distances_;
  rmm::device_uvector<int64_t> ref_indices_;
  rmm::device_uvector<T> ref_distances_;
  int k_;
  cuvs::distance::DistanceType metric;
};

const std::vector<RandomKNNInputs> random_inputs = {
  // test each distance metric on a small-ish input, with row-major inputs
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2Expanded, true},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2Unexpanded, true},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2SqrtExpanded, true},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2SqrtUnexpanded, true},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L1, true},
  {256, 512, 16, 8, cuvs::distance::DistanceType::Linf, true},
  {256, 512, 16, 8, cuvs::distance::DistanceType::InnerProduct, true},
  {256, 512, 16, 8, cuvs::distance::DistanceType::CorrelationExpanded, true},
  {256, 512, 16, 8, cuvs::distance::DistanceType::CosineExpanded, true},
  {256, 512, 16, 8, cuvs::distance::DistanceType::LpUnexpanded, true},
  {256, 512, 16, 8, cuvs::distance::DistanceType::JensenShannon, true},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2SqrtExpanded, true},
  {256, 512, 16, 8, cuvs::distance::DistanceType::Canberra, true},
  // test each distance metric with col-major inputs
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2Expanded, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2Unexpanded, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2SqrtExpanded, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2SqrtUnexpanded, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L1, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::Linf, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::InnerProduct, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::CorrelationExpanded, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::CosineExpanded, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::LpUnexpanded, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::JensenShannon, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2SqrtExpanded, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::Canberra, false},
  // larger tests on different sized data / k values
  {10000, 40000, 32, 30, cuvs::distance::DistanceType::L2Expanded, false},
  {345, 1023, 16, 128, cuvs::distance::DistanceType::CosineExpanded, true},
  {789, 20516, 64, 256, cuvs::distance::DistanceType::L2SqrtExpanded, false},
  {1000, 500000, 128, 128, cuvs::distance::DistanceType::L2Expanded, true},
  {1000, 500000, 128, 128, cuvs::distance::DistanceType::L2Expanded, false},
  {1000, 5000, 128, 128, cuvs::distance::DistanceType::LpUnexpanded, true},
  {1000, 5000, 128, 128, cuvs::distance::DistanceType::L2SqrtExpanded, false},
  {1000, 5000, 128, 128, cuvs::distance::DistanceType::InnerProduct, false}};

typedef RandomBruteForceKNNTest<float> RandomBruteForceKNNTestF;
TEST_P(RandomBruteForceKNNTestF, BruteForce) { this->testBruteForce(); }

INSTANTIATE_TEST_CASE_P(RandomBruteForceKNNTest,
                        RandomBruteForceKNNTestF,
                        ::testing::ValuesIn(random_inputs));

}  // namespace cuvs::neighbors::brute_force

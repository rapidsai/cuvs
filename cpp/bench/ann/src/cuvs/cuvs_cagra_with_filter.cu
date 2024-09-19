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
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/init.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/random/rng.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/itertools.hpp>

#include <thrust/sequence.h>

#include <chrono>
#include <cuda_fp16.h>
#include <iomanip>
#include <iostream>

namespace cuvs::neighbors::cagra {

struct print_metric {
  cuvs::distance::DistanceType value;
};

struct RandomKNNInputs {
  int num_queries;
  int num_db_vecs;
  int dim;
  int k;
  float sparsity;
  float threshold_to_bf;
  cuvs::distance::DistanceType metric;
  search_algo algo;
  bool row_major;
};

inline auto operator<<(std::ostream& os, const print_metric& p) -> std::ostream&
{
  switch (p.value) {
    case cuvs::distance::DistanceType::L2Expanded: os << "L2Expanded"; break;
    case cuvs::distance::DistanceType::L2SqrtExpanded: os << "L2SqrtExpanded"; break;
    case cuvs::distance::DistanceType::CosineExpanded: os << "CosineExpanded"; break;
    case cuvs::distance::DistanceType::L1: os << "L1"; break;
    case cuvs::distance::DistanceType::L2Unexpanded: os << "L2Unexpanded"; break;
    case cuvs::distance::DistanceType::L2SqrtUnexpanded: os << "L2SqrtUnexpanded"; break;
    case cuvs::distance::DistanceType::InnerProduct: os << "InnerProduct"; break;
    case cuvs::distance::DistanceType::Linf: os << "Linf"; break;
    case cuvs::distance::DistanceType::Canberra: os << "Canberra"; break;
    case cuvs::distance::DistanceType::LpUnexpanded: os << "LpUnexpanded"; break;
    case cuvs::distance::DistanceType::CorrelationExpanded: os << "CorrelationExpanded"; break;
    case cuvs::distance::DistanceType::JaccardExpanded: os << "JaccardExpanded"; break;
    case cuvs::distance::DistanceType::HellingerExpanded: os << "HellingerExpanded"; break;
    case cuvs::distance::DistanceType::Haversine: os << "Haversine"; break;
    case cuvs::distance::DistanceType::BrayCurtis: os << "BrayCurtis"; break;
    case cuvs::distance::DistanceType::JensenShannon: os << "JensenShannon"; break;
    case cuvs::distance::DistanceType::HammingUnexpanded: os << "HammingUnexpanded"; break;
    case cuvs::distance::DistanceType::KLDivergence: os << "KLDivergence"; break;
    case cuvs::distance::DistanceType::RusselRaoExpanded: os << "RusselRaoExpanded"; break;
    case cuvs::distance::DistanceType::DiceExpanded: os << "DiceExpanded"; break;
    case cuvs::distance::DistanceType::Precomputed: os << "Precomputed"; break;
    default: RAFT_FAIL("unreachable code");
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const RandomKNNInputs& input)
{
  return os << "num_queries:" << input.num_queries << " num_vecs:" << input.num_db_vecs
            << " dim:" << input.dim << " k:" << input.k << " metric:" << print_metric{input.metric}
            << " row_major:" << input.row_major;
}

template <typename T, typename DistT = T>
class CagraWithFilterKNNBenchmark {
 public:
  CagraWithFilterKNNBenchmark(const RandomKNNInputs& params, const std::string& type_str)
    : stream_(raft::resource::get_cuda_stream(handle_)),
      params_(params),
      type_str_(type_str),
      database(params_.num_db_vecs * params_.dim, stream_),
      search_queries(params_.num_queries * params_.dim, stream_),
      cuvs_indices_(params_.num_queries * params_.k, stream_),
      cuvs_distances_(params_.num_queries * params_.k, stream_),
      cuvs_indices_expected_(params_.num_queries * params_.k, stream_),
      cuvs_distances_expected_(params_.num_queries * params_.k, stream_)
  {
    int64_t dataset_size = params_.num_db_vecs * params_.dim;
    int64_t queries_size = params_.num_queries * params_.dim;

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
      DistT{0.0});

    auto X      = raft::make_device_matrix<T, int64_t>(handle_, 1, dataset_size + queries_size);
    auto labels = raft::make_device_vector<int64_t, int64_t>(handle_, 1);

    raft::random::make_blobs<T, int64_t>(X.data_handle(),
                                         labels.data_handle(),
                                         1,
                                         dataset_size + queries_size,
                                         1,
                                         stream_,
                                         false,
                                         nullptr,
                                         nullptr,
                                         T(1.0),
                                         false,
                                         T(-1.0f),
                                         T(1.0f),
                                         uint64_t(2024));
    raft::copy(database.data(), X.data_handle(), dataset_size, stream_);
    raft::copy(search_queries.data(), X.data_handle() + dataset_size, queries_size, stream_);
    raft::resource::sync_stream(handle_);
  }

  auto calc_recall(const std::vector<uint32_t>& expected_idx,
                   const std::vector<uint32_t>& actual_idx,
                   size_t rows,
                   size_t cols)
  {
    size_t match_count = 0;
    size_t total_count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t k = 0; k < cols; ++k) {
        size_t idx_k = i * cols + k;  // row major assumption!
        auto act_idx = actual_idx[idx_k];
        for (size_t j = 0; j < cols; ++j) {
          size_t idx   = i * cols + j;  // row major assumption!
          auto exp_idx = expected_idx[idx];
          if (act_idx == exp_idx) {
            match_count++;
            break;
          }
        }
      }
    }
    return std::make_tuple(
      static_cast<double>(match_count * 100.0) / static_cast<double>(total_count),
      match_count,
      total_count);
  }

  int64_t create_sparse_bitset(int64_t total, float sparsity, std::vector<uint32_t>& bitset)
  {
    int64_t num_ones = static_cast<int64_t>((total * 1.0f) * (1.0f - sparsity));
    int64_t res      = num_ones;

    for (auto& item : bitset) {
      item = static_cast<uint32_t>(0);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dis(0, total - 1);

    while (num_ones > 0) {
      int64_t index = dis(gen);

      uint32_t& element    = bitset[index / (8 * sizeof(uint32_t))];
      int64_t bit_position = index % (8 * sizeof(uint32_t));

      if (((element >> bit_position) & 1) == 0) {
        element |= (static_cast<uint32_t>(1) << bit_position);
        num_ones--;
      }
    }
    return res;
  }

  void runBenchmark()
  {
    DistT metric_arg = 3.0;
    rmm::device_uvector<char> workspace(0, stream_);

    std::chrono::duration<double, std::milli> build_dur;
    std::chrono::duration<double, std::milli> search_dur;

    auto index_params   = cuvs::neighbors::cagra::index_params();
    index_params.metric = cuvs::distance::DistanceType::InnerProduct;

    auto search_params = cuvs::neighbors::cagra::search_params();
    search_params.algo = cuvs::neighbors::cagra::search_algo::SINGLE_CTA;

    auto indices = raft::make_device_matrix_view<uint32_t, int64_t, raft::row_major>(
      cuvs_indices_.data(), params_.num_queries, params_.k);
    auto distances = raft::make_device_matrix_view<DistT, int64_t, raft::row_major>(
      cuvs_distances_.data(), params_.num_queries, params_.k);

    auto indices_expected = raft::make_device_matrix_view<uint32_t, int64_t, raft::row_major>(
      cuvs_indices_expected_.data(), params_.num_queries, params_.k);
    auto distances_expected = raft::make_device_matrix_view<DistT, int64_t, raft::row_major>(
      cuvs_distances_expected_.data(), params_.num_queries, params_.k);

    cuvs::core::bitset<std::uint32_t, int64_t> bitset_filter(handle_, params_.num_db_vecs, false);

    std::vector<uint32_t> bitset_cpu(bitset_filter.n_elements());

    create_sparse_bitset(bitset_filter.size(), params_.sparsity, bitset_cpu);
    raft::copy(bitset_filter.data(), bitset_cpu.data(), bitset_filter.n_elements(), stream_);

    auto filter =
      std::make_optional(cuvs::neighbors::filtering::bitset_filter(bitset_filter.view()));
    raft::resource::sync_stream(handle_, stream_);

    {
      auto idx_warm =
        cuvs::neighbors::cagra::build(handle_,
                                      index_params,
                                      raft::make_device_matrix_view<const T, uint32_t>(
                                        database.data(), params_.num_db_vecs, params_.dim));
      cuvs::neighbors::cagra::search(handle_,
                                     search_params,
                                     idx_warm,
                                     raft::make_device_matrix_view<const T, int64_t>(
                                       search_queries.data(), params_.num_queries, params_.dim),
                                     indices,
                                     distances,
                                     filter,
                                     params_.threshold_to_bf);
      flush_l2_cache();
      raft::resource::sync_stream(handle_, stream_);
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto idx   = cuvs::neighbors::cagra::build(handle_,
                                             index_params,
                                             raft::make_device_matrix_view<const T, int64_t>(
                                               database.data(), params_.num_db_vecs, params_.dim));
    raft::resource::sync_stream(handle_, stream_);
    auto end  = std::chrono::high_resolution_clock::now();
    build_dur = end - start;

    start = std::chrono::high_resolution_clock::now();
    cuvs::neighbors::cagra::search(handle_,
                                   search_params,
                                   idx,
                                   raft::make_device_matrix_view<const T, int64_t>(
                                     search_queries.data(), params_.num_queries, params_.dim),
                                   indices,
                                   distances,
                                   filter,
                                   params_.threshold_to_bf);
    raft::resource::sync_stream(handle_, stream_);
    end        = std::chrono::high_resolution_clock::now();
    search_dur = end - start;

    // calc recall
    std::vector<uint32_t> actual_idx_cpu(indices.size());
    std::vector<uint32_t> expected_idx_cpu(indices.size());
    {
      auto idx =
        cuvs::neighbors::cagra::build(handle_,
                                      index_params,
                                      raft::make_device_matrix_view<const T, uint32_t>(
                                        database.data(), params_.num_db_vecs, params_.dim));
      cuvs::neighbors::cagra::search(handle_,
                                     search_params,
                                     idx,
                                     raft::make_device_matrix_view<const T, int64_t>(
                                       search_queries.data(), params_.num_queries, params_.dim),
                                     indices_expected,
                                     distances_expected,
                                     filter,
                                     0.0);
      raft::resource::sync_stream(handle_, stream_);
    }

    raft::copy(actual_idx_cpu.data(), indices.data_handle(), indices.size(), stream_);
    raft::copy(expected_idx_cpu.data(), indices_expected.data_handle(), indices.size(), stream_);
    raft::resource::sync_stream(handle_, stream_);

    auto [actual_recall, match_count, total_count] =
      calc_recall(expected_idx_cpu, actual_idx_cpu, params_.num_queries, params_.k);

    double total_dur  = build_dur.count() + search_dur.count();
    double throughput = static_cast<double>(params_.num_queries) / (total_dur / 1000.0);
    printResult(
      params_, build_dur.count(), search_dur.count(), total_dur, throughput, actual_recall);
  }

  void setUp()
  {
    unsigned long long int seed = 1234ULL;
    raft::random::RngState r(seed);

    // JensenShannon distance requires positive values
    T min_val = params_.metric == cuvs::distance::DistanceType::JensenShannon ? T(0.0) : T(-1.0);
    uniform(handle_, r, database.data(), params_.num_db_vecs * params_.dim, min_val, T(1.0));
    uniform(handle_, r, search_queries.data(), params_.num_queries * params_.dim, min_val, T(1.0));
  }

 private:
  void flush_l2_cache()
  {
    int l2_cache_size = 0;
    int device_id     = 0;
    RAFT_CUDA_TRY(cudaGetDevice(&device_id));
    RAFT_CUDA_TRY(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, device_id));
    scratch_buf_ = rmm::device_buffer(l2_cache_size * 3, stream_);
    RAFT_CUDA_TRY(cudaMemsetAsync(scratch_buf_.data(), 0, scratch_buf_.size(), stream_));
  };

  void printResult(const RandomKNNInputs& params,
                   double build_time,
                   double search_time,
                   double total_time,
                   double throughput,
                   double recall)
  {
    std::cout << std::left << std::setw(15) << type_str_ << std::setw(10) << params.num_queries
              << std::setw(10) << params.num_db_vecs << std::setw(10) << params.dim << std::setw(10)
              << params.k << std::setw(20) << print_metric{params.metric} << std::setw(15)
              << (params.row_major ? "row" : "col") << std::right << std::setw(10) << std::fixed
              << (params.sparsity >= params.threshold_to_bf ? "Brute" : "Cagra") << std::setw(20)
              << std::fixed << std::setprecision(3) << params.sparsity << std::right
              << std::setw(20) << std::fixed << std::setprecision(3) << recall << std::right
              << std::setw(20) << std::fixed << std::setprecision(3) << build_time << std::right
              << std::setw(20) << std::fixed << std::setprecision(3) << search_time << std::right
              << std::setw(20) << std::fixed << std::setprecision(3) << total_time << std::right
              << std::setw(20) << std::fixed << std::setprecision(3) << throughput << "\n";
  }
  raft::resources handle_;
  cudaStream_t stream_ = 0;
  RandomKNNInputs params_;
  rmm::device_uvector<T> database;
  rmm::device_uvector<T> search_queries;
  rmm::device_uvector<uint32_t> cuvs_indices_;
  rmm::device_uvector<DistT> cuvs_distances_;
  rmm::device_uvector<uint32_t> cuvs_indices_expected_;
  rmm::device_uvector<DistT> cuvs_distances_expected_;
  rmm::device_buffer scratch_buf_;
  std::string type_str_;
};

static std::vector<RandomKNNInputs> getInputs()
{
  std::vector<RandomKNNInputs> param_vec;
  struct TestParams {
    int num_queries;
    int num_db_vecs;
    int dim;
    int k;
    float sparsity;
    cuvs::distance::DistanceType metric;
    search_algo algo;
    bool row_major;
  };
  {
    const std::vector<TestParams> params_group =
      raft::util::itertools::product<TestParams>({int(100)},
                                                 {int(1024 * 1024)},
                                                 {int(32)},
                                                 {int(64)},
                                                 {0.1f, 0.3f, 0.5f, 0.8f, 0.9f, 0.99f},
                                                 {cuvs::distance::DistanceType::InnerProduct},
                                                 {search_algo::SINGLE_CTA},
                                                 {true});

    param_vec.reserve(params_group.size());
    for (TestParams params : params_group) {
      param_vec.push_back(RandomKNNInputs({params.num_queries,
                                           params.num_db_vecs,
                                           params.dim,
                                           params.k,
                                           params.sparsity + 0.0001f,
                                           params.sparsity,  // threshold_to_bf
                                           params.metric,
                                           params.algo,
                                           params.row_major}));
      // add case for original `CAGRA`.
      param_vec.push_back(RandomKNNInputs({params.num_queries,
                                           params.num_db_vecs,
                                           params.dim,
                                           params.k,
                                           params.sparsity,
                                           1.0f,  // threshold_to_bf
                                           params.metric,
                                           params.algo,
                                           params.row_major}));
    }
  }
  return param_vec;
}

void printHeader()
{
  std::cout << std::left << std::setw(15) << "Type" << std::setw(10) << "Queries" << std::setw(10)
            << "Vectors" << std::setw(10) << "Dim" << std::setw(10) << "K" << std::setw(20)
            << "Metric" << std::setw(15) << "Layout" << std::right << std::setw(10)
            << "BruteF/Cagra" << std::setw(20) << "Sparsity" << std::right << std::setw(20)
            << "Recall rate (%)" << std::right << std::setw(20) << "Build Time (ms)" << std::right
            << std::setw(20) << "Search Time (ms)" << std::right << std::setw(20)
            << "Total Time (ms)" << std::right << std::setw(20) << "Throughput (q/s)"
            << "\n";
  std::cout << std::string(200, '-') << "\n";
}

void runBenchmarkForType()
{
  auto selected_inputs = getInputs();
  for (const auto& input : selected_inputs) {
    CagraWithFilterKNNBenchmark<float, float> benchmark(input, "float");
    benchmark.setUp();
    benchmark.runBenchmark();
  }
}

}  // namespace cuvs::neighbors::cagra

int main()
{
  cuvs::neighbors::cagra::printHeader();
  cuvs::neighbors::cagra::runBenchmarkForType();
  return 0;
}

/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#pragma once

#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include <raft/core/resource/cuda_stream.hpp>

#include "naive_knn.cuh"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/normalize.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/itertools.hpp>

#include <rmm/device_buffer.hpp>

#include <gtest/gtest.h>

#include <thrust/sequence.h>

#include <cstddef>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace cuvs::neighbors::cagra {
namespace {

/** Xorshift rondem number generator.
 *
 * See https://en.wikipedia.org/wiki/Xorshift#xorshift for reference.
 */
_RAFT_HOST_DEVICE inline uint64_t xorshift64(uint64_t u)
{
  u ^= u >> 12;
  u ^= u << 25;
  u ^= u >> 27;
  return u * 0x2545F4914F6CDD1DULL;
}

// For sort_knn_graph test
template <typename IdxT>
void RandomSuffle(raft::host_matrix_view<IdxT, int64_t> index)
{
  for (IdxT i = 0; i < index.extent(0); i++) {
    uint64_t rand       = i;
    IdxT* const row_ptr = index.data_handle() + i * index.extent(1);
    for (unsigned j = 0; j < index.extent(1); j++) {
      // Swap two indices at random
      rand          = xorshift64(rand);
      const auto i0 = rand % index.extent(1);
      rand          = xorshift64(rand);
      const auto i1 = rand % index.extent(1);

      const auto tmp = row_ptr[i0];
      row_ptr[i0]    = row_ptr[i1];
      row_ptr[i1]    = tmp;
    }
  }
}

template <typename DistanceT, typename DatatT, typename IdxT>
testing::AssertionResult CheckOrder(raft::host_matrix_view<IdxT, int64_t> index_test,
                                    raft::host_matrix_view<DatatT, int64_t> dataset)
{
  for (IdxT i = 0; i < index_test.extent(0); i++) {
    const DatatT* const base_vec = dataset.data_handle() + i * dataset.extent(1);
    const IdxT* const index_row  = index_test.data_handle() + i * index_test.extent(1);
    DistanceT prev_distance      = 0;
    for (unsigned j = 0; j < index_test.extent(1) - 1; j++) {
      const DatatT* const target_vec = dataset.data_handle() + index_row[j] * dataset.extent(1);
      DistanceT distance             = 0;
      for (unsigned l = 0; l < dataset.extent(1); l++) {
        const auto diff =
          static_cast<DistanceT>(target_vec[l]) - static_cast<DistanceT>(base_vec[l]);
        distance += diff * diff;
      }
      if (prev_distance > distance) {
        return testing::AssertionFailure()
               << "Wrong index order (row = " << i << ", neighbor_id = " << j
               << "). (distance[neighbor_id-1] = " << prev_distance
               << "should be larger than distance[neighbor_id] = " << distance << ")";
      }
      prev_distance = distance;
    }
  }
  return testing::AssertionSuccess();
}

template <typename T>
struct fpi_mapper {};

template <>
struct fpi_mapper<double> {
  using type                         = int64_t;
  static constexpr int kBitshiftBase = 53;
};

template <>
struct fpi_mapper<float> {
  using type                         = int32_t;
  static constexpr int kBitshiftBase = 24;
};

template <>
struct fpi_mapper<half> {
  using type                         = int16_t;
  static constexpr int kBitshiftBase = 11;
};

// Generate dataset to ensure no rounding error occurs in the norm computation of any two vectors.
// When testing the CAGRA index sorting function, rounding errors can affect the norm and alter the
// order of the index. To ensure the accuracy of the test, we utilize the dataset. The generation
// method is based on the error-free transformation (EFT) method.
template <typename T>
RAFT_KERNEL GenerateRoundingErrorFreeDataset_kernel(T* const ptr,
                                                    const uint32_t size,
                                                    const typename fpi_mapper<T>::type resolution)
{
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) { return; }

  const float u32 = *reinterpret_cast<const typename fpi_mapper<T>::type*>(ptr + tid);
  ptr[tid]        = u32 / resolution;
}

template <typename T>
void GenerateRoundingErrorFreeDataset(
  const raft::resources& handle,
  T* const ptr,
  const uint32_t n_row,
  const uint32_t dim,
  raft::random::RngState& rng,
  const bool diff_flag  // true if compute the norm between two vectors
)
{
  using mapper_type         = fpi_mapper<T>;
  using int_type            = typename mapper_type::type;
  auto cuda_stream          = raft::resource::get_cuda_stream(handle);
  const uint32_t size       = n_row * dim;
  const uint32_t block_size = 256;
  const uint32_t grid_size  = (size + block_size - 1) / block_size;

  const auto bitshift = (mapper_type::kBitshiftBase - std::log2(dim) - (diff_flag ? 1 : 0)) / 2;
  // Skip the test when `dim` is too big for type `T` to allow rounding error-free test.
  if (bitshift <= 1) { GTEST_SKIP(); }
  const int_type resolution = int_type{1} << static_cast<unsigned>(std::floor(bitshift));
  raft::random::uniformInt<int_type>(
    handle, rng, reinterpret_cast<int_type*>(ptr), size, -resolution, resolution - 1);

  GenerateRoundingErrorFreeDataset_kernel<T>
    <<<grid_size, block_size, 0, cuda_stream>>>(ptr, size, resolution);
}

template <class DataT>
void InitDataset(const raft::resources& handle,
                 DataT* const datatset_ptr,
                 std::uint32_t size,
                 std::uint32_t dim,
                 distance::DistanceType metric,
                 raft::random::RngState& r)
{
  if constexpr (std::is_same_v<DataT, float> || std::is_same_v<DataT, half>) {
    GenerateRoundingErrorFreeDataset(handle, datatset_ptr, size, dim, r, true);

    if (metric == InnerProduct) {
      auto dataset_view = raft::make_device_matrix_view(datatset_ptr, size, dim);
      raft::linalg::row_normalize(
        handle, raft::make_const_mdspan(dataset_view), dataset_view, raft::linalg::L2Norm);
    }
  } else if constexpr (std::is_same_v<DataT, std::uint8_t> || std::is_same_v<DataT, std::int8_t>) {
    if constexpr (std::is_same_v<DataT, std::int8_t>) {
      raft::random::uniformInt(handle, r, datatset_ptr, size * dim, DataT(-10), DataT(10));
    } else {
      raft::random::uniformInt(handle, r, datatset_ptr, size * dim, DataT(1), DataT(20));
    }

    if (metric == InnerProduct) {
      // TODO (enp1s0): Change this once row_normalize supports (u)int8 matrices.
      // https://github.com/rapidsai/raft/issues/2291

      using ComputeT    = float;
      auto dataset_view = raft::make_device_matrix_view(datatset_ptr, size, dim);
      auto dev_row_norm = raft::make_device_vector<ComputeT>(handle, size);
      const auto normalized_norm =
        (std::is_same_v<DataT, std::uint8_t> ? 40 : 20) * std::sqrt(static_cast<ComputeT>(dim));

      raft::linalg::reduce(dev_row_norm.data_handle(),
                           datatset_ptr,
                           dim,
                           size,
                           0.f,
                           true,
                           true,
                           raft::resource::get_cuda_stream(handle),
                           false,
                           raft::sq_op(),
                           raft::add_op(),
                           raft::sqrt_op());
      raft::linalg::matrix_vector_op(
        handle,
        raft::make_const_mdspan(dataset_view),
        raft::make_const_mdspan(dev_row_norm.view()),
        dataset_view,
        raft::linalg::Apply::ALONG_COLUMNS,
        [normalized_norm] __device__(DataT elm, ComputeT norm) {
          const ComputeT v           = elm / norm * normalized_norm;
          const ComputeT max_v_range = std::numeric_limits<DataT>::max();
          const ComputeT min_v_range = std::numeric_limits<DataT>::min();
          return static_cast<DataT>(std::min(max_v_range, std::max(min_v_range, v)));
        });
    }
  }
}

enum class graph_build_algo {
  /* Use IVF-PQ to build all-neighbors knn graph */
  IVF_PQ,
  /* Experimental, use NN-Descent to build all-neighbors knn graph */
  NN_DESCENT,
  /* Choose default automatically */
  AUTO
};

}  // namespace

struct AnnCagraInputs {
  int n_queries;
  int n_rows;
  int dim;
  int k;
  graph_build_algo build_algo;
  search_algo algo;
  int max_queries;
  int team_size;
  int itopk_size;
  int search_width;
  cuvs::distance::DistanceType metric;
  bool host_dataset;
  bool include_serialized_dataset;
  // std::optional<double>
  double min_recall;  // = std::nullopt;
  std::optional<float> ivf_pq_search_refine_ratio = std::nullopt;
  std::optional<vpq_params> compression           = std::nullopt;

  std::optional<bool> non_owning_memory_buffer_flag = std::nullopt;
};

inline ::std::ostream& operator<<(::std::ostream& os, const AnnCagraInputs& p)
{
  const auto metric_str = [](const cuvs::distance::DistanceType dist) -> std::string {
    switch (dist) {
      case InnerProduct: return "InnerProduct";
      case L2Expanded: return "L2";
      default: break;
    }
    return "Unknown";
  };

  std::vector<std::string> algo       = {"single-cta", "multi_cta", "multi_kernel", "auto"};
  std::vector<std::string> build_algo = {"IVF_PQ", "NN_DESCENT", "AUTO"};
  os << "{n_queries=" << p.n_queries << ", dataset shape=" << p.n_rows << "x" << p.dim
     << ", k=" << p.k << ", " << algo.at((int)p.algo) << ", max_queries=" << p.max_queries
     << ", itopk_size=" << p.itopk_size << ", search_width=" << p.search_width
     << ", metric=" << metric_str(p.metric) << ", " << (p.host_dataset ? "host" : "device")
     << ", build_algo=" << build_algo.at((int)p.build_algo);
  if ((int)p.build_algo == 0 && p.ivf_pq_search_refine_ratio) {
    os << "(refine_rate=" << *p.ivf_pq_search_refine_ratio << ')';
  }
  if (p.compression.has_value()) {
    auto vpq = p.compression.value();
    os << ", pq_bits=" << vpq.pq_bits << ", pq_dim=" << vpq.pq_dim
       << ", vq_n_centers=" << vpq.vq_n_centers;
  }
  os << '}' << std::endl;
  return os;
}

template <typename DistanceT, typename DataT, typename IdxT>
class AnnCagraTest : public ::testing::TestWithParam<AnnCagraInputs> {
 public:
  AnnCagraTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnCagraInputs>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

 protected:
  void testCagra()
  {
    size_t queries_size = ps.n_queries * ps.k;
    std::vector<IdxT> indices_Cagra(queries_size);
    std::vector<IdxT> indices_naive(queries_size);
    std::vector<DistanceT> distances_Cagra(queries_size);
    std::vector<DistanceT> distances_naive(queries_size);

    {
      rmm::device_uvector<DistanceT> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);

      cuvs::neighbors::naive_knn<DistanceT, DataT, IdxT>(handle_,
                                                         distances_naive_dev.data(),
                                                         indices_naive_dev.data(),
                                                         search_queries.data(),
                                                         database.data(),
                                                         ps.n_queries,
                                                         ps.n_rows,
                                                         ps.dim,
                                                         ps.k,
                                                         ps.metric);
      raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      raft::resource::sync_stream(handle_);
    }

    {
      rmm::device_uvector<DistanceT> distances_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_dev(queries_size, stream_);

      {
        cagra::index_params index_params;
        index_params.metric = ps.metric;  // Note: currently ony the cagra::index_params metric is
                                          // not used for knn_graph building.
        switch (ps.build_algo) {
          case graph_build_algo::IVF_PQ:
            index_params.graph_build_params =
              graph_build_params::ivf_pq_params(raft::matrix_extent<int64_t>(ps.n_rows, ps.dim));
            if (ps.ivf_pq_search_refine_ratio) {
              std::get<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(
                index_params.graph_build_params)
                .refinement_rate = *ps.ivf_pq_search_refine_ratio;
            }
            break;
          case graph_build_algo::NN_DESCENT: {
            index_params.graph_build_params =
              graph_build_params::nn_descent_params(index_params.intermediate_graph_degree);
            break;
          }
          case graph_build_algo::AUTO:
            // do nothing
            break;
        };

        index_params.compression = ps.compression;
        cagra::search_params search_params;
        search_params.algo        = ps.algo;
        search_params.max_queries = ps.max_queries;
        search_params.team_size   = ps.team_size;

        auto database_view = raft::make_device_matrix_view<const DataT, int64_t>(
          (const DataT*)database.data(), ps.n_rows, ps.dim);

        {
          cagra::index<DataT, IdxT> index(handle_);
          if (ps.host_dataset) {
            auto database_host = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
            raft::copy(database_host.data_handle(), database.data(), database.size(), stream_);
            auto database_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
              (const DataT*)database_host.data_handle(), ps.n_rows, ps.dim);

            index = cagra::build(handle_, index_params, database_host_view);
          } else {
            index = cagra::build(handle_, index_params, database_view);
          };

          cagra::serialize(handle_, "cagra_index", index, ps.include_serialized_dataset);
        }

        cagra::index<DataT, IdxT> index(handle_);
        cagra::deserialize(handle_, "cagra_index", &index);

        if (!ps.include_serialized_dataset) { index.update_dataset(handle_, database_view); }

        auto search_queries_view = raft::make_device_matrix_view<const DataT, int64_t>(
          search_queries.data(), ps.n_queries, ps.dim);
        auto indices_out_view =
          raft::make_device_matrix_view<IdxT, int64_t>(indices_dev.data(), ps.n_queries, ps.k);
        auto dists_out_view = raft::make_device_matrix_view<DistanceT, int64_t>(
          distances_dev.data(), ps.n_queries, ps.k);

        cagra::search(
          handle_, search_params, index, search_queries_view, indices_out_view, dists_out_view);
        raft::update_host(distances_Cagra.data(), distances_dev.data(), queries_size, stream_);
        raft::update_host(indices_Cagra.data(), indices_dev.data(), queries_size, stream_);

        raft::resource::sync_stream(handle_);
      }

      // for (int i = 0; i < min(ps.n_queries, 10); i++) {
      //   //  std::cout << "query " << i << std::end;
      //   print_vector("T", indices_naive.data() + i * ps.k, ps.k, std::cout);
      //   print_vector("C", indices_Cagra.data() + i * ps.k, ps.k, std::cout);
      //   print_vector("T", distances_naive.data() + i * ps.k, ps.k, std::cout);
      //   print_vector("C", distances_Cagra.data() + i * ps.k, ps.k, std::cout);
      // }
      double min_recall = ps.min_recall;
      EXPECT_TRUE(eval_neighbours(indices_naive,
                                  indices_Cagra,
                                  distances_naive,
                                  distances_Cagra,
                                  ps.n_queries,
                                  ps.k,
                                  0.003,
                                  min_recall));
      if (!ps.compression.has_value()) {
        // Don't evaluate distances for CAGRA-Q for now as the error can be somewhat large
        EXPECT_TRUE(eval_distances(handle_,
                                   database.data(),
                                   search_queries.data(),
                                   indices_dev.data(),
                                   distances_dev.data(),
                                   ps.n_rows,
                                   ps.dim,
                                   ps.n_queries,
                                   ps.k,
                                   ps.metric,
                                   1.0e-4));
      }
    }
  }

  void SetUp() override
  {
    database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    search_queries.resize(ps.n_queries * ps.dim, stream_);
    raft::random::RngState r(1234ULL);
    InitDataset(handle_, database.data(), ps.n_rows, ps.dim, ps.metric, r);
    InitDataset(handle_, search_queries.data(), ps.n_queries, ps.dim, ps.metric, r);
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnCagraInputs ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;
};

template <typename DistanceT, typename DataT, typename IdxT>
class AnnCagraAddNodesTest : public ::testing::TestWithParam<AnnCagraInputs> {
 public:
  AnnCagraAddNodesTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnCagraInputs>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

 protected:
  void testCagra()
  {
    // TODO (tarang-jain): remove when NN Descent index building support InnerProduct. Reference
    // issue: https://github.com/rapidsai/raft/issues/2276
    if (ps.metric == InnerProduct && ps.build_algo == graph_build_algo::NN_DESCENT) GTEST_SKIP();
    if (ps.compression != std::nullopt) GTEST_SKIP();

    size_t queries_size = ps.n_queries * ps.k;
    std::vector<IdxT> indices_Cagra(queries_size);
    std::vector<IdxT> indices_naive(queries_size);
    std::vector<DistanceT> distances_Cagra(queries_size);
    std::vector<DistanceT> distances_naive(queries_size);

    {
      rmm::device_uvector<DistanceT> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);

      cuvs::neighbors::naive_knn<DistanceT, DataT, IdxT>(handle_,
                                                         distances_naive_dev.data(),
                                                         indices_naive_dev.data(),
                                                         search_queries.data(),
                                                         database.data(),
                                                         ps.n_queries,
                                                         ps.n_rows,
                                                         ps.dim,
                                                         ps.k,
                                                         ps.metric);
      raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      raft::resource::sync_stream(handle_);
    }

    {
      rmm::device_uvector<DistanceT> distances_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_dev(queries_size, stream_);

      {
        cagra::index_params index_params;
        index_params.metric = ps.metric;  // Note: currently ony the cagra::index_params metric is
                                          // not used for knn_graph building.

        switch (ps.build_algo) {
          case graph_build_algo::IVF_PQ:
            index_params.graph_build_params =
              graph_build_params::ivf_pq_params(raft::matrix_extent<int64_t>(ps.n_rows, ps.dim));
            if (ps.ivf_pq_search_refine_ratio) {
              std::get<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(
                index_params.graph_build_params)
                .refinement_rate = *ps.ivf_pq_search_refine_ratio;
            }
            break;
          case graph_build_algo::NN_DESCENT: {
            index_params.graph_build_params =
              graph_build_params::nn_descent_params(index_params.intermediate_graph_degree);
            break;
          }
          case graph_build_algo::AUTO:
            // do nothing
            break;
        };

        cagra::search_params search_params;
        search_params.algo        = ps.algo;
        search_params.max_queries = ps.max_queries;
        search_params.team_size   = ps.team_size;
        search_params.itopk_size  = ps.itopk_size;

        const double initial_dataset_ratio      = 0.90;
        const std::size_t initial_database_size = ps.n_rows * initial_dataset_ratio;

        auto initial_database_view = raft::make_device_matrix_view<const DataT, int64_t>(
          (const DataT*)database.data(), initial_database_size, ps.dim);

        cagra::index<DataT, IdxT> index(handle_);
        if (ps.host_dataset) {
          auto database_host = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
          raft::copy(
            database_host.data_handle(), database.data(), initial_database_view.size(), stream_);
          auto database_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
            (const DataT*)database_host.data_handle(), initial_database_size, ps.dim);
          index = cagra::build(handle_, index_params, database_host_view);
        } else {
          index = cagra::build(handle_, index_params, initial_database_view);
        };

        auto additional_dataset =
          raft::make_host_matrix<DataT, int64_t>(ps.n_rows - initial_database_size, index.dim());
        raft::copy(additional_dataset.data_handle(),
                   static_cast<const DataT*>(database.data()) + initial_database_view.size(),
                   additional_dataset.size(),
                   stream_);

        auto new_dataset_buffer = raft::make_device_matrix<DataT, int64_t>(handle_, 0, 0);
        auto new_graph_buffer   = raft::make_device_matrix<IdxT, int64_t>(handle_, 0, 0);
        std::optional<raft::device_matrix_view<DataT, int64_t, raft::layout_stride>>
          new_dataset_buffer_view                                                    = std::nullopt;
        std::optional<raft::device_matrix_view<IdxT, int64_t>> new_graph_buffer_view = std::nullopt;
        if (ps.non_owning_memory_buffer_flag.has_value() &&
            ps.non_owning_memory_buffer_flag.value()) {
          const auto stride =
            dynamic_cast<const cuvs::neighbors::strided_dataset<DataT, int64_t>*>(&index.data())
              ->stride();
          new_dataset_buffer = raft::make_device_matrix<DataT, int64_t>(handle_, ps.n_rows, stride);
          new_graph_buffer =
            raft::make_device_matrix<IdxT, int64_t>(handle_, ps.n_rows, index.graph_degree());

          new_dataset_buffer_view = raft::make_device_strided_matrix_view<DataT, int64_t>(
            new_dataset_buffer.data_handle(), ps.n_rows, ps.dim, stride);
          new_graph_buffer_view = new_graph_buffer.view();
        }

        cagra::extend_params extend_params;
        cagra::extend(handle_,
                      extend_params,
                      raft::make_const_mdspan(additional_dataset.view()),
                      index,
                      new_dataset_buffer_view,
                      new_graph_buffer_view);

        auto search_queries_view = raft::make_device_matrix_view<const DataT, int64_t>(
          search_queries.data(), ps.n_queries, ps.dim);
        auto indices_out_view =
          raft::make_device_matrix_view<IdxT, int64_t>(indices_dev.data(), ps.n_queries, ps.k);
        auto dists_out_view = raft::make_device_matrix_view<DistanceT, int64_t>(
          distances_dev.data(), ps.n_queries, ps.k);

        cagra::search(
          handle_, search_params, index, search_queries_view, indices_out_view, dists_out_view);
        raft::update_host(distances_Cagra.data(), distances_dev.data(), queries_size, stream_);
        raft::update_host(indices_Cagra.data(), indices_dev.data(), queries_size, stream_);
        raft::resource::sync_stream(handle_);
      }

      double min_recall = ps.min_recall;
      EXPECT_TRUE(eval_neighbours(indices_naive,
                                  indices_Cagra,
                                  distances_naive,
                                  distances_Cagra,
                                  ps.n_queries,
                                  ps.k,
                                  0.006,
                                  min_recall));
      EXPECT_TRUE(eval_distances(handle_,
                                 database.data(),
                                 search_queries.data(),
                                 indices_dev.data(),
                                 distances_dev.data(),
                                 ps.n_rows,
                                 ps.dim,
                                 ps.n_queries,
                                 ps.k,
                                 ps.metric,
                                 1.0e-4));
    }
  }

  void SetUp() override
  {
    database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    search_queries.resize(ps.n_queries * ps.dim, stream_);
    raft::random::RngState r(1234ULL);
    InitDataset(handle_, database.data(), ps.n_rows, ps.dim, ps.metric, r);
    InitDataset(handle_, search_queries.data(), ps.n_queries, ps.dim, ps.metric, r);
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnCagraInputs ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;
};

inline std::vector<AnnCagraInputs> generate_inputs()
{
  // TODO(tfeher): test MULTI_CTA kernel with search_width > 1 to allow multiple CTA per queries
  std::vector<AnnCagraInputs> inputs = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {1000},
    {1, 8, 17},
    {1, 16},  // k
    {graph_build_algo::IVF_PQ, graph_build_algo::NN_DESCENT},
    {search_algo::SINGLE_CTA, search_algo::MULTI_CTA, search_algo::MULTI_KERNEL},
    {0, 1, 10, 100},  // query size
    {0},
    {256},
    {1},
    {cuvs::distance::DistanceType::L2Expanded},
    {false},
    {true},
    {0.995});

  auto inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {1000},
    {1, 3, 5, 7, 8, 17, 64, 128, 137, 192, 256, 512, 619, 1024},  // dim
    {16},                                                         // k
    {graph_build_algo::IVF_PQ, graph_build_algo::NN_DESCENT},
    {search_algo::AUTO},
    {10},
    {0},
    {64},
    {1},
    {cuvs::distance::DistanceType::L2Expanded},
    {false},
    {true},
    {0.995});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());
  inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {1000},
    {64},
    {16},
    {graph_build_algo::IVF_PQ, graph_build_algo::NN_DESCENT},
    {search_algo::AUTO},
    {10},
    {0, 4, 8, 16, 32},  // team_size
    {64},
    {1},
    {cuvs::distance::DistanceType::L2Expanded},
    {false},
    {false},
    {0.995});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {1000},
    {64},
    {16},
    {graph_build_algo::IVF_PQ, graph_build_algo::NN_DESCENT},
    {search_algo::AUTO},
    {10},
    {0},  // team_size
    {32, 64, 128, 256, 512, 768},
    {1},
    {cuvs::distance::DistanceType::L2Expanded},
    {false},
    {true},
    {0.995});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  inputs2 =
    raft::util::itertools::product<AnnCagraInputs>({100},
                                                   {10000, 20000},
                                                   {32},
                                                   {10},
                                                   {graph_build_algo::AUTO},
                                                   {search_algo::AUTO},
                                                   {10},
                                                   {0},  // team_size
                                                   {64},
                                                   {1},
                                                   {cuvs::distance::DistanceType::L2Expanded},
                                                   {false, true},
                                                   {false},
                                                   {0.985});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  // a few PQ configurations
  inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {10000},
    {64, 128, 192, 256, 512, 1024},  // dim
    {16},                            // k
    {graph_build_algo::IVF_PQ},
    {search_algo::AUTO},
    {10},
    {0},
    {64},
    {1},
    {cuvs::distance::DistanceType::L2Expanded},
    {false},
    {true},
    {0.6});                      // don't demand high recall without refinement
  for (uint32_t pq_len : {2}) {  // for now, only pq_len = 2 is supported, more options coming soon
    for (uint32_t vq_n_centers : {100, 1000}) {
      for (auto input : inputs2) {
        vpq_params ps{};
        ps.pq_dim       = input.dim / pq_len;
        ps.vq_n_centers = vq_n_centers;
        input.compression.emplace(ps);
        inputs.push_back(input);
      }
    }
  }

  // refinement options
  inputs2 =
    raft::util::itertools::product<AnnCagraInputs>({100},
                                                   {5000},
                                                   {32, 64},
                                                   {16},
                                                   {graph_build_algo::IVF_PQ},
                                                   {search_algo::AUTO},
                                                   {10},
                                                   {0},  // team_size
                                                   {64},
                                                   {1},
                                                   {cuvs::distance::DistanceType::L2Expanded},
                                                   {false, true},
                                                   {false},
                                                   {0.99},
                                                   {1.0f, 2.0f, 3.0f});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {1000},
    {1, 3, 5, 7, 8, 17, 64, 128, 137, 192, 256, 512, 619, 1024},  // dim
    {10},
    {graph_build_algo::IVF_PQ},
    {search_algo::AUTO},
    {10},
    {0},  // team_size
    {64},
    {1},
    {cuvs::distance::DistanceType::L2Expanded},
    {false},
    {false},
    {0.995});
  for (auto input : inputs2) {
    input.non_owning_memory_buffer_flag = true;
    inputs.push_back(input);
  }

  return inputs;
}

const std::vector<AnnCagraInputs> inputs = generate_inputs();

}  // namespace cuvs::neighbors::cagra

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

#pragma once

#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include <raft/core/resource/cuda_stream.hpp>

#include "naive_knn.cuh"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/vamana.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/linalg/add.cuh>
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

namespace cuvs::neighbors::vamana {

struct AnnVamanaInputs {
  int n_rows;
  int dim;
  int graph_degree;
  int visited_size;
  double max_batchsize;
  cuvs::distance::DistanceType metric;
  bool host_dataset;
};

template<typename DataT, typename IdxT>
inline void CheckGraph(vamana::index<DataT, IdxT>* index_, AnnVamanaInputs inputs, cudaStream_t stream) {

	EXPECT_TRUE(index_->graph().size() == (inputs.n_rows * inputs.graph_degree));
	EXPECT_TRUE(index_->graph().extent(0) == inputs.n_rows);
	EXPECT_TRUE(index_->graph().extent(1) == inputs.graph_degree);
	
	// Copy graph to host
	auto h_graph = raft::make_host_matrix<IdxT, int64_t>(inputs.n_rows, inputs.graph_degree);
        raft::copy(h_graph.data_handle(), index_->graph().data_handle(), index_->graph().size(), stream);

	size_t edge_count=0;
	int max_degree=0;
	for(int i=0; i<h_graph.extent(0); i++) {
          int temp_degree=0;
	  for(int j=0; j<h_graph.extent(1); j++) {
            if(h_graph(i,j) < (uint32_t)(inputs.n_rows)) temp_degree++;
	  }
	  if(temp_degree > max_degree) max_degree = temp_degree;
	  edge_count += (size_t)temp_degree;
	}

	// Tests for acceptable range of edges - low dim can also impact this
        // Minimum expected maximum degree across the whole graph
	EXPECT_TRUE(max_degree >= std::min(inputs.graph_degree, inputs.dim)); 

	float max_edges = (float)(inputs.n_rows * std::min(inputs.graph_degree,inputs.dim));

	RAFT_LOG_INFO("dim:%d, degree:%d, visited_size:%d, edge_count:%lu, max_edges:%lu\n", inputs.dim, inputs.graph_degree, inputs.visited_size, edge_count, (size_t)max_edges);

        // Graph won't always be full, but <75% is very unlikely
	EXPECT_TRUE(((float)edge_count / max_edges) > 0.75); 

	// TODO - Anything else we can test without search


}

template <typename DistanceT, typename DataT, typename IdxT>
class AnnVamanaTest : public ::testing::TestWithParam<AnnVamanaInputs> {
 public:
  AnnVamanaTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnVamanaInputs>::GetParam()),
      database(0, stream_)
  {
  }

 protected:
  void testVamana()
  {
        vamana::index_params index_params;
        index_params.metric = ps.metric;  
        index_params.graph_degree = ps.graph_degree;
	index_params.visited_size = ps.visited_size;
	index_params.max_batchsize = ps.max_batchsize;
	
                                          
        auto database_view = raft::make_device_matrix_view<const DataT, int64_t>(
          (const DataT*)database.data(), ps.n_rows, ps.dim);

//        {
          vamana::index<DataT, IdxT> index(handle_);
          if (ps.host_dataset) {
            auto database_host = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
            raft::copy(database_host.data_handle(), database.data(), database.size(), stream_);
            auto database_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
              (const DataT*)database_host.data_handle(), ps.n_rows, ps.dim);

            index = vamana::build(handle_, index_params, database_host_view);
          } else {
            index = vamana::build(handle_, index_params, database_view);
          };

	  CheckGraph<DataT, IdxT>(&index, ps, stream_);

	  // Can we test serialize here without deserialize implemented?
//          cagra::serialize(handle_, "cagra_index", index, ps.include_serialized_dataset);
//        }
  }

  void SetUp() override
  {
    database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      raft::random::normal(handle_, r, database.data(), ps.n_rows * ps.dim, DataT(0.1), DataT(2.0));
    } else {
      raft::random::uniformInt(
        handle_, r, database.data(), ps.n_rows * ps.dim, DataT(1), DataT(20));
    }
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnVamanaInputs ps;
  rmm::device_uvector<DataT> database;
};

inline std::vector<AnnVamanaInputs> generate_inputs()
{

  std::vector<AnnVamanaInputs> inputs = raft::util::itertools::product<AnnVamanaInputs>(
    {1000},
//    {1, 3, 5, 7, 8, 17, 64, 128, 137, 192, 256, 512, 619, 1024},  // TODO - fix alignment issue for odd dims
    {2, 8, 16, 32, 64, 128, 192, 256, 512, 1024},  // dim
    {32}, // graph degree
    {64,128,256}, // visited_size
    {0.06, 0.2},
    {cuvs::distance::DistanceType::L2Expanded},
    {false});

  std::vector<AnnVamanaInputs> inputs2 = raft::util::itertools::product<AnnVamanaInputs>(
    {1000},
    {2, 8, 16, 32, 64, 128, 192, 256, 512, 1024},  // dim
    {64}, // graph degree
    {128,256,512}, // visited_size
    {0.06, 0.2},
    {cuvs::distance::DistanceType::L2Expanded},
    {false});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  inputs2 = raft::util::itertools::product<AnnVamanaInputs>(
    {1000},
    {2, 8, 16, 32, 64, 128, 192, 256, 512, 1024},  // dim
    {128}, // graph degree
    {256,512}, // visited_size
    {0.06, 0.2},
    {cuvs::distance::DistanceType::L2Expanded},
    {false});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  inputs2 = raft::util::itertools::product<AnnVamanaInputs>(
    {1000},
    {2, 8, 16, 32, 64, 128, 192, 256, 512, 1024},  // dim
    {256}, // graph degree
    {512,1024}, // visited_size
    {0.06, 0.2},
    {cuvs::distance::DistanceType::L2Expanded},
    {false});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  return inputs;
}


const std::vector<AnnVamanaInputs> inputs = generate_inputs();

}  // namespace cuvs::neighbors::vamana

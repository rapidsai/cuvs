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

#include <gtest/gtest.h>

#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/rng.cuh>

#include <cstdint>

namespace cuvs::neighbors::cagra {

class cagra_extreme_inputs_oob_test : public ::testing::Test {
 public:
  using data_type = float;

 protected:
  void run()
  {
    cagra::index_params ix_ps;
    graph_build_params::ivf_pq_params gb_params{};
    gb_params.refinement_rate       = 2;
    ix_ps.graph_build_params        = gb_params;
    ix_ps.graph_degree              = 64;
    ix_ps.intermediate_graph_degree = 128;

    [[maybe_unused]] auto ix = cagra::build(res, ix_ps, raft::make_const_mdspan(dataset->view()));
    raft::resource::sync_stream(res);
  }

  void SetUp() override
  {
    dataset.emplace(raft::make_device_matrix<data_type, int64_t>(res, n_samples, n_dim));
    raft::random::RngState r(1234ULL);
    raft::random::normal(
      res, r, dataset->data_handle(), n_samples * n_dim, data_type(0), data_type(1e20));
    raft::resource::sync_stream(res);
  }

  void TearDown() override
  {
    dataset.reset();
    raft::resource::sync_stream(res);
  }

 private:
  raft::resources res;
  std::optional<raft::device_matrix<data_type, int64_t>> dataset = std::nullopt;

  constexpr static int64_t n_samples                   = 100000;
  constexpr static int64_t n_dim                       = 200;
  constexpr static cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
};

TEST_F(cagra_extreme_inputs_oob_test, cagra_extreme_inputs_oob_test) { this->run(); }

}  // namespace cuvs::neighbors::cagra

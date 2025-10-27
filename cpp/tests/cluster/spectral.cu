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

#include <cuvs/cluster/spectral.hpp>
#include <cuvs/preprocessing/spectral_embedding.hpp>
#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/stats/adjusted_rand_index.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace cuvs {

struct SpectralClusteringInputs {
  int n_row;
  int n_col;
  int n_clusters;
  int n_components;
  int n_neighbors;
  int n_init;
  float cluster_std;
  uint64_t seed;
};

class SpectralClusteringTest : public ::testing::TestWithParam<SpectralClusteringInputs> {
 public:
  SpectralClusteringTest()
    : d_labels(0, raft::resource::get_cuda_stream(handle)),
      d_labels_ref(0, raft::resource::get_cuda_stream(handle))
  {
  }

  void basicTest()
  {
    testparams = ::testing::TestWithParam<SpectralClusteringInputs>::GetParam();

    int n_samples  = testparams.n_row;
    int n_features = testparams.n_col;

    cluster::spectral::params params;
    params.n_clusters   = testparams.n_clusters;
    params.n_components = testparams.n_components;
    params.n_neighbors  = testparams.n_neighbors;
    params.n_init       = testparams.n_init;
    params.rng_state    = raft::random::RngState(testparams.seed);

    auto X      = raft::make_device_matrix<float, int>(handle, n_samples, n_features);
    auto labels = raft::make_device_vector<int, int>(handle, n_samples);
    auto stream = raft::resource::get_cuda_stream(handle);

    raft::random::make_blobs<float, int>(X.data_handle(),
                                         labels.data_handle(),
                                         n_samples,
                                         n_features,
                                         params.n_clusters,
                                         stream,
                                         true,
                                         nullptr,
                                         nullptr,
                                         testparams.cluster_std,
                                         false,
                                         -10.0f,
                                         10.0f,
                                         testparams.seed);

    d_labels.resize(n_samples, stream);
    d_labels_ref.resize(n_samples, stream);

    raft::copy(d_labels_ref.data(), labels.data_handle(), n_samples, stream);

    auto connectivity_graph =
      raft::make_device_coo_matrix<float, int, int, int>(handle, n_samples, n_samples);

    cuvs::preprocessing::spectral_embedding::params embed_params;
    embed_params.n_neighbors = params.n_neighbors;
    embed_params.seed        = params.rng_state.seed;

    cuvs::preprocessing::spectral_embedding::helpers::create_connectivity_graph(
      handle, embed_params, X.view(), connectivity_graph);

    cluster::spectral::fit_predict(
      handle,
      params,
      connectivity_graph.view(),
      raft::make_device_vector_view<int, int>(d_labels.data(), n_samples));

    raft::resource::sync_stream(handle, stream);

    score =
      raft::stats::adjusted_rand_index(d_labels_ref.data(), d_labels.data(), n_samples, stream);

    if (score < 0.8) {
      std::stringstream ss;
      ss << "Expected: " << raft::arr2Str(d_labels_ref.data(), 25, "d_labels_ref", stream);
      std::cout << (ss.str().c_str()) << '\n';
      ss.str(std::string());
      ss << "Actual: " << raft::arr2Str(d_labels.data(), 25, "d_labels", stream);
      std::cout << (ss.str().c_str()) << '\n';
      std::cout << "Score = " << score << '\n';
    }
  }

  void SetUp() override { basicTest(); }

 protected:
  raft::resources handle;
  SpectralClusteringInputs testparams;
  rmm::device_uvector<int> d_labels;
  rmm::device_uvector<int> d_labels_ref;
  double score;
  cluster::spectral::params params;
};

const std::vector<SpectralClusteringInputs> inputs = {
  // Small datasets with well-separated clusters
  {100, 10, 2, 2, 10, 3, 0.3f, 42ULL},  // Tighter clusters for better separation
  {200, 20, 3, 3, 15, 3, 0.3f, 123ULL},
  {500, 15, 4, 4, 20, 3, 0.3f, 456ULL},

  // Medium datasets
  {1000, 32, 5, 5, 25, 5, 0.3f, 789ULL},
  {2000, 50, 6, 6, 30, 5, 0.3f, 111ULL},

  // Larger datasets with more clusters
  {5000, 100, 8, 8, 40, 5, 0.3f, 222ULL},
  {10000, 50, 10, 10, 50, 5, 0.3f, 333ULL},

  {1000, 30, 5, 5, 20, 3, 0.3f, 444ULL},
  {1000, 30, 3, 3, 20, 3, 0.3f, 555ULL},

  // Varying cluster separation
  {500, 20, 3, 3, 15, 3, 0.2f, 666ULL},  // Very tight clusters
  {500, 20, 3, 3, 15, 3, 0.5f, 777ULL},  // More spread but still reasonable
};

TEST_P(SpectralClusteringTest, Result)
{
  ASSERT_GT(score, 0.8) << "Adjusted Rand Index is too low: " << score;
}

INSTANTIATE_TEST_CASE_P(SpectralClusteringTests,
                        SpectralClusteringTest,
                        ::testing::ValuesIn(inputs));

}  // namespace cuvs

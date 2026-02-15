/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/nn_descent.hpp>

namespace cuvs::neighbors::nn_descent {

index_params::index_params(size_t graph_degree, cuvs::distance::DistanceType metric)
{
  this->graph_degree              = graph_degree;
  this->intermediate_graph_degree = 1.5 * graph_degree;
  this->metric                    = metric;
}
}  // namespace cuvs::neighbors::nn_descent

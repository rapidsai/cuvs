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

#include "./detail/quantization.cuh"

#include <cuvs/neighbors/quantization.hpp>

namespace cuvs::neighbors::quantization {

template <typename T, typename QuantI>
void ScalarQuantizer<T, QuantI>::train(raft::resources const& res,
                                       sq_params params,
                                       raft::device_matrix_view<const T, int64_t> dataset)
{
  RAFT_EXPECTS(params.quantile > 0.0 && params.quantile <= 1.0,
               "quantile for scalar quantization needs to be within (0, 1] but is %f",
               params.quantile);

  // conditional: search for quantiles / min / max
  if (!is_trained_) {
    auto [min, max] = detail::quantile_min_max(res, dataset, params.quantile);

    // persist settings in params
    constexpr int64_t range_q_type = static_cast<int64_t>(std::numeric_limits<QuantI>::max()) -
                                     static_cast<int64_t>(std::numeric_limits<QuantI>::min());
    min_        = min;
    max_        = max;
    is_trained_ = true;
    RAFT_LOG_DEBUG("ScalarQuantizer train min=%lf max=%lf.", double(min_), double(max_));
  }
}

template <typename T, typename QuantI>
void ScalarQuantizer<T, QuantI>::train(raft::resources const& res,
                                       sq_params params,
                                       raft::host_matrix_view<const T, int64_t> dataset)
{
  RAFT_EXPECTS(params.quantile > 0.0 && params.quantile <= 1.0,
               "quantile for scalar quantization needs to be within (0, 1] but is %f",
               params.quantile);

  // conditional: search for quantiles / min / max
  if (!is_trained_) {
    auto [min, max] = detail::quantile_min_max(res, dataset, params.quantile);

    // persist settings in params
    constexpr int64_t range_q_type = static_cast<int64_t>(std::numeric_limits<QuantI>::max()) -
                                     static_cast<int64_t>(std::numeric_limits<QuantI>::min());
    min_        = min;
    max_        = max;
    is_trained_ = true;
    RAFT_LOG_DEBUG("ScalarQuantizer train min=%lf max=%lf.", double(min_), double(max_));
  }
}

template <typename T, typename QuantI>
raft::device_matrix<QuantI, int64_t> ScalarQuantizer<T, QuantI>::transform(
  raft::resources const& res, raft::device_matrix_view<const T, int64_t> dataset)
{
  RAFT_EXPECTS(is_trained_, "ScalarQuantizer needs to be trained first!");
  return detail::scalar_transform<T, QuantI>(res, dataset, min_, max_);
}

template <typename T, typename QuantI>
raft::host_matrix<QuantI, int64_t> ScalarQuantizer<T, QuantI>::transform(
  raft::resources const& res, raft::host_matrix_view<const T, int64_t> dataset)
{
  RAFT_EXPECTS(is_trained_, "ScalarQuantizer needs to be trained first!");
  return detail::scalar_transform<T, QuantI>(res, dataset, min_, max_);
}

template <typename T, typename QuantI>
raft::device_matrix<T, int64_t> ScalarQuantizer<T, QuantI>::inverse_transform(
  raft::resources const& res, raft::device_matrix_view<const QuantI, int64_t> dataset)
{
  RAFT_EXPECTS(is_trained_, "ScalarQuantizer needs to be trained first!");
  return detail::inverse_scalar_transform<T, QuantI>(res, dataset, min_, max_);
}

template <typename T, typename QuantI>
raft::host_matrix<T, int64_t> ScalarQuantizer<T, QuantI>::inverse_transform(
  raft::resources const& res, raft::host_matrix_view<const QuantI, int64_t> dataset)
{
  RAFT_EXPECTS(is_trained_, "ScalarQuantizer needs to be trained first!");
  return detail::inverse_scalar_transform<T, QuantI>(res, dataset, min_, max_);
}

template <typename T, typename QuantI>
_RAFT_HOST_DEVICE bool ScalarQuantizer<T, QuantI>::is_trained() const
{
  return is_trained_;
}

template <typename T, typename QuantI>
_RAFT_HOST_DEVICE bool ScalarQuantizer<T, QuantI>::operator==(
  const ScalarQuantizer<T, QuantI>& other) const
{
  return (!is_trained() && !other.is_trained()) ||
         (is_trained() == other.is_trained() && detail::fp_equals(min(), other.min()) &&
          detail::fp_equals(max(), other.max()));
}

template <typename T, typename QuantI>
_RAFT_HOST_DEVICE T ScalarQuantizer<T, QuantI>::min() const
{
  return min_;
}

template <typename T, typename QuantI>
_RAFT_HOST_DEVICE T ScalarQuantizer<T, QuantI>::max() const
{
  return max_;
}

#define CUVS_INST_QUANTIZATION(T, QuantI) \
  template struct cuvs::neighbors::quantization::ScalarQuantizer<T, QuantI>;

CUVS_INST_QUANTIZATION(double, int8_t);
CUVS_INST_QUANTIZATION(float, int8_t);
CUVS_INST_QUANTIZATION(half, int8_t);

#undef CUVS_INST_QUANTIZATION

}  // namespace cuvs::neighbors::quantization
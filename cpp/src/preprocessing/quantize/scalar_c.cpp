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

#include <cstdint>
#include <dlpack/dlpack.h>

#include <cuvs/core/c_api.h>
#include <cuvs/core/exceptions.hpp>
#include <cuvs/core/interop.hpp>
#include <cuvs/preprocessing/quantize/scalar.h>
#include <cuvs/preprocessing/quantize/scalar.hpp>

namespace {

template <typename T>
void _train(cuvsResources_t res,
            cuvsScalarQuantizerParams params,
            DLManagedTensor* dataset_tensor,
            cuvsScalarQuantizer_t quantizer)
{
  auto dataset = dataset_tensor->dl_tensor;

  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  auto quantizer_params     = cuvs::preprocessing::quantize::scalar::params();
  quantizer_params.quantile = params.quantile;

  cuvs::preprocessing::quantize::scalar::quantizer<T> ret;

  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    ret = cuvs::preprocessing::quantize::scalar::train(*res_ptr, quantizer_params, mds);
  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    ret = cuvs::preprocessing::quantize::scalar::train(*res_ptr, quantizer_params, mds);
  } else {
    RAFT_FAIL("dataset must be accessible on host or device memory");
  }

  quantizer->min_ = ret.min_;
  quantizer->max_ = ret.max_;
}

template <typename T, typename OutputT = int8_t>
void _transform(cuvsResources_t res,
                cuvsScalarQuantizer_t quantizer_,
                DLManagedTensor* dataset_tensor,
                DLManagedTensor* out_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  cuvs::preprocessing::quantize::scalar::quantizer<T> quantizer;
  quantizer.min_ = quantizer_->min_;
  quantizer.max_ = quantizer_->max_;

  auto dataset = dataset_tensor->dl_tensor;
  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type     = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    using out_mdspan_type = raft::device_matrix_view<OutputT, int64_t, raft::row_major>;

    cuvs::preprocessing::quantize::scalar::transform(
      *res_ptr,
      quantizer,
      cuvs::core::from_dlpack<mdspan_type>(dataset_tensor),
      cuvs::core::from_dlpack<out_mdspan_type>(out_tensor));

  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using mdspan_type     = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    using out_mdspan_type = raft::host_matrix_view<OutputT, int64_t, raft::row_major>;

    cuvs::preprocessing::quantize::scalar::transform(
      *res_ptr,
      quantizer,
      cuvs::core::from_dlpack<mdspan_type>(dataset_tensor),
      cuvs::core::from_dlpack<out_mdspan_type>(out_tensor));
  } else {
    RAFT_FAIL("dataset must be accessible on host or device memory");
  }
}

template <typename OutputT, typename InputT = int8_t>
void _inverse_transform(cuvsResources_t res,
                        cuvsScalarQuantizer_t quantizer_,
                        DLManagedTensor* dataset_tensor,
                        DLManagedTensor* out_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  cuvs::preprocessing::quantize::scalar::quantizer<OutputT> quantizer;
  quantizer.min_ = quantizer_->min_;
  quantizer.max_ = quantizer_->max_;

  auto dataset = dataset_tensor->dl_tensor;
  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type     = raft::device_matrix_view<InputT const, int64_t, raft::row_major>;
    using out_mdspan_type = raft::device_matrix_view<OutputT, int64_t, raft::row_major>;

    cuvs::preprocessing::quantize::scalar::inverse_transform(
      *res_ptr,
      quantizer,
      cuvs::core::from_dlpack<mdspan_type>(dataset_tensor),
      cuvs::core::from_dlpack<out_mdspan_type>(out_tensor));

  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using mdspan_type     = raft::host_matrix_view<InputT const, int64_t, raft::row_major>;
    using out_mdspan_type = raft::host_matrix_view<OutputT, int64_t, raft::row_major>;

    cuvs::preprocessing::quantize::scalar::inverse_transform(
      *res_ptr,
      quantizer,
      cuvs::core::from_dlpack<mdspan_type>(dataset_tensor),
      cuvs::core::from_dlpack<out_mdspan_type>(out_tensor));
  } else {
    RAFT_FAIL("dataset must be accessible on host or device memory");
  }
}
}  // namespace

extern "C" cuvsError_t cuvsScalarQuantizerParamsCreate(cuvsScalarQuantizerParams_t* params)
{
  return cuvs::core::translate_exceptions(
    [=] { *params = new cuvsScalarQuantizerParams{.quantile = 0.99}; });
}

extern "C" cuvsError_t cuvsScalarQuantizerParamsDestroy(cuvsScalarQuantizerParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsScalarQuantizerCreate(cuvsScalarQuantizer_t* quantizer)
{
  return cuvs::core::translate_exceptions([=] { *quantizer = new cuvsScalarQuantizer{}; });
}

extern "C" cuvsError_t cuvsScalarQuantizerDestroy(cuvsScalarQuantizer_t quantizer)
{
  return cuvs::core::translate_exceptions([=] { delete quantizer; });
}

extern "C" cuvsError_t cuvsScalarQuantizerTrain(cuvsResources_t res,
                                                cuvsScalarQuantizerParams_t params,
                                                DLManagedTensor* dataset_tensor,
                                                cuvsScalarQuantizer_t quantizer)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = dataset_tensor->dl_tensor;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      _train<float>(res, *params, dataset_tensor, quantizer);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 64) {
      _train<double>(res, *params, dataset_tensor, quantizer);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsScalarQuantizerTransform(cuvsResources_t res,
                                                    cuvsScalarQuantizer_t quantizer,
                                                    DLManagedTensor* dataset_tensor,
                                                    DLManagedTensor* out_tensor)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = dataset_tensor->dl_tensor;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      _transform<float>(res, quantizer, dataset_tensor, out_tensor);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 64) {
      _transform<double>(res, quantizer, dataset_tensor, out_tensor);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

cuvsError_t cuvsScalarQuantizerInverseTransform(cuvsResources_t res,
                                                cuvsScalarQuantizer_t quantizer,
                                                DLManagedTensor* dataset,
                                                DLManagedTensor* out)
{
  return cuvs::core::translate_exceptions([=] {
    auto dtype = out->dl_tensor.dtype;
    if (dtype.code == kDLFloat && dtype.bits == 32) {
      _inverse_transform<float>(res, quantizer, dataset, out);
    } else if (dtype.code == kDLFloat && dtype.bits == 64) {
      _inverse_transform<double>(res, quantizer, dataset, out);
    } else {
      RAFT_FAIL(
        "Unsupported output dataset DLtensor dtype: %d and bits: %d", dtype.code, dtype.bits);
    }
  });
}

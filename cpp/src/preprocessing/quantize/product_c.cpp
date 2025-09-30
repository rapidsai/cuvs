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
#include <cuvs/preprocessing/quantize/product.h>
#include <cuvs/preprocessing/quantize/product.hpp>

namespace {

template <typename T, typename OutputT = uint8_t>
void _transform(cuvsResources_t res,
                cuvsProductQuantizer_t quantizer,
                DLManagedTensor* dataset_tensor,
                DLManagedTensor* out_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto q = reinterpret_cast<cuvs::preprocessing::quantize::product::quantizer<T>*>(quantizer->addr);

  auto dataset = dataset_tensor->dl_tensor;
  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type     = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    using out_mdspan_type = raft::device_matrix_view<OutputT, int64_t, raft::row_major>;

    cuvs::preprocessing::quantize::product::transform(
      *res_ptr,
      *q,
      cuvs::core::from_dlpack<mdspan_type>(dataset_tensor),
      cuvs::core::from_dlpack<out_mdspan_type>(out_tensor));

  } else {
    RAFT_FAIL("dataset must be accessible on device memory");
  }
}

template <typename T>
void* _train(cuvsResources_t res,
             cuvsProductQuantizerParams_t params,
             DLManagedTensor* dataset_tensor)
{
  auto dataset = dataset_tensor->dl_tensor;

  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  auto quantizer_params                        = cuvs::preprocessing::quantize::product::params();
  quantizer_params.pq_bits                     = params->pq_bits;
  quantizer_params.pq_dim                      = params->pq_dim;
  quantizer_params.kmeans_n_iters              = params->kmeans_n_iters;
  quantizer_params.pq_kmeans_trainset_fraction = params->pq_kmeans_trainset_fraction;
  quantizer_params.vq_n_centers                = 1;
  quantizer_params.pq_kmeans_type =
    static_cast<cuvs::cluster::kmeans::kmeans_type>(params->pq_kmeans_type);

  cuvs::preprocessing::quantize::product::quantizer<T>* ret = nullptr;

  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    auto return_value =
      cuvs::preprocessing::quantize::product::train(*res_ptr, quantizer_params, mds);
    ret = new cuvs::preprocessing::quantize::product::quantizer<T>(return_value);
  } else {
    RAFT_FAIL("dataset must be accessible on device memory");
  }
  return ret;
}

}  // namespace

extern "C" cuvsError_t cuvsProductQuantizerParamsCreate(cuvsProductQuantizerParams_t* params)
{
  return cuvs::core::translate_exceptions([=] { *params = new cuvsProductQuantizerParams; });
}

extern "C" cuvsError_t cuvsProductQuantizerParamsDestroy(cuvsProductQuantizerParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsProductQuantizerCreate(cuvsProductQuantizer_t* quantizer)
{
  return cuvs::core::translate_exceptions([=] { *quantizer = new cuvsProductQuantizer; });
}

extern "C" cuvsError_t cuvsProductQuantizerDestroy(cuvsProductQuantizer_t quantizer)
{
  return cuvs::core::translate_exceptions([=] { delete quantizer; });
}

extern "C" cuvsError_t cuvsProductQuantizerTrain(cuvsResources_t res,
                                                 cuvsProductQuantizerParams_t params,
                                                 DLManagedTensor* dataset_tensor,
                                                 cuvsProductQuantizer_t quantizer)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset     = dataset_tensor->dl_tensor;
    quantizer->dtype = dataset.dtype;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 64) {
      quantizer->addr = reinterpret_cast<uintptr_t>(_train<double>(res, params, dataset_tensor));
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      quantizer->addr = reinterpret_cast<uintptr_t>(_train<float>(res, params, dataset_tensor));
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsProductQuantizerTransform(cuvsResources_t res,
                                                     cuvsProductQuantizer_t quantizer,
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

extern "C" cuvsError_t cuvsProductQuantizerGetPqBits(cuvsProductQuantizer_t quantizer,
                                                     uint32_t* pq_bits)
{
  return cuvs::core::translate_exceptions([=] {
    if (quantizer != nullptr) {
      auto quant_addr = quantizer->addr;
      if (quantizer->dtype.code == kDLFloat && quantizer->dtype.bits == 32) {
        *pq_bits =
          (reinterpret_cast<cuvs::preprocessing::quantize::product::quantizer<float>*>(quant_addr))
            ->params_quantizer.pq_bits;
      } else if (quantizer->dtype.code == kDLFloat && quantizer->dtype.bits == 64) {
        *pq_bits =
          (reinterpret_cast<cuvs::preprocessing::quantize::product::quantizer<double>*>(quant_addr))
            ->params_quantizer.pq_bits;
      } else {
        RAFT_FAIL("Unsupported quantizer dtype: %d and bits: %d",
                  quantizer->dtype.code,
                  quantizer->dtype.bits);
      }
    } else {
      RAFT_FAIL("quantizer is not initialized");
    }
  });
}

extern "C" cuvsError_t cuvsProductQuantizerGetPqDim(cuvsProductQuantizer_t quantizer,
                                                    uint32_t* pq_dim)
{
  return cuvs::core::translate_exceptions([=] {
    if (quantizer != nullptr) {
      auto quant_addr = quantizer->addr;
      if (quantizer->dtype.code == kDLFloat && quantizer->dtype.bits == 32) {
        *pq_dim =
          (reinterpret_cast<cuvs::preprocessing::quantize::product::quantizer<float>*>(quant_addr))
            ->params_quantizer.pq_dim;
      } else if (quantizer->dtype.code == kDLFloat && quantizer->dtype.bits == 64) {
        *pq_dim =
          (reinterpret_cast<cuvs::preprocessing::quantize::product::quantizer<double>*>(quant_addr))
            ->params_quantizer.pq_dim;
      } else {
        RAFT_FAIL("Unsupported quantizer dtype: %d and bits: %d",
                  quantizer->dtype.code,
                  quantizer->dtype.bits);
      }
    } else {
      RAFT_FAIL("quantizer is not initialized");
    }
  });
}

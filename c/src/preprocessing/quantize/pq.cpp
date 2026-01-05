/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <dlpack/dlpack.h>

#include <cuvs/core/c_api.h>
#include <cuvs/preprocessing/quantize/pq.h>
#include <cuvs/preprocessing/quantize/pq.hpp>
#include "../../core/exceptions.hpp"
#include "../../core/interop.hpp"

namespace {

template <typename T, typename OutputT = uint8_t>
void _transform(cuvsResources_t res,
                cuvsProductQuantizer_t quantizer,
                DLManagedTensor* dataset_tensor,
                DLManagedTensor* out_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto q = reinterpret_cast<cuvs::preprocessing::quantize::pq::quantizer<T>*>(quantizer->addr);

  auto dataset = dataset_tensor->dl_tensor;
  using mdspan_type     = raft::device_matrix_view<T const, int64_t, raft::row_major>;
  using out_mdspan_type = raft::device_matrix_view<OutputT, int64_t, raft::row_major>;
  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    cuvs::preprocessing::quantize::pq::transform(
      *res_ptr,
      *q,
      cuvs::core::from_dlpack<mdspan_type>(dataset_tensor),
      cuvs::core::from_dlpack<out_mdspan_type>(out_tensor));

  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using host_mdspan_type     = raft::host_matrix_view<T const, int64_t, raft::row_major>;

    cuvs::preprocessing::quantize::pq::transform(
      *res_ptr,
      *q,
      cuvs::core::from_dlpack<host_mdspan_type>(dataset_tensor),
      cuvs::core::from_dlpack<out_mdspan_type>(out_tensor));
  } else {
    RAFT_FAIL("dataset must be accessible on host or device memory");
  }
}

template <typename T>
void* _train(cuvsResources_t res,
             cuvsProductQuantizerParams_t params,
             DLManagedTensor* dataset_tensor)
{
  auto dataset = dataset_tensor->dl_tensor;

  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  auto quantizer_params                        = cuvs::preprocessing::quantize::pq::params();
  quantizer_params.pq_bits                     = params->pq_bits;
  quantizer_params.pq_dim                      = params->pq_dim;
  quantizer_params.vq_n_centers                = params->vq_n_centers;
  quantizer_params.kmeans_n_iters              = params->kmeans_n_iters;
  quantizer_params.pq_kmeans_trainset_fraction = params->pq_kmeans_trainset_fraction;
  quantizer_params.vq_kmeans_trainset_fraction = params->vq_kmeans_trainset_fraction;
  quantizer_params.pq_kmeans_type =
    static_cast<cuvs::cluster::kmeans::kmeans_type>(params->pq_kmeans_type);
  quantizer_params.max_train_points_per_pq_code = params->max_train_points_per_pq_code;
  quantizer_params.use_vq = params->use_vq;
  quantizer_params.use_subspaces = params->use_subspaces;

  cuvs::preprocessing::quantize::pq::quantizer<T>* ret = nullptr;

  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    ret = new cuvs::preprocessing::quantize::pq::quantizer<T>{
      cuvs::preprocessing::quantize::pq::train(*res_ptr, quantizer_params, mds)};
  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using host_mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<host_mdspan_type>(dataset_tensor);
    ret = new cuvs::preprocessing::quantize::pq::quantizer<T>{
      cuvs::preprocessing::quantize::pq::train(*res_ptr, quantizer_params, mds)};
  } else {
    RAFT_FAIL("dataset must be accessible on host or device memory");
  }
  return ret;
}

template <typename DataT, typename QuantT = uint8_t>
void _inverse_transform(cuvsResources_t res,
                cuvsProductQuantizer_t quantizer,
                DLManagedTensor* codes_tensor,
                DLManagedTensor* out_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto q = reinterpret_cast<cuvs::preprocessing::quantize::pq::quantizer<DataT>*>(quantizer->addr);

  auto codes = codes_tensor->dl_tensor;
  if (cuvs::core::is_dlpack_device_compatible(codes)) {
    using codes_mdspan_type     = raft::device_matrix_view<QuantT const, int64_t, raft::row_major>;
    using data_mdspan_type = raft::device_matrix_view<DataT, int64_t, raft::row_major>;

    cuvs::preprocessing::quantize::pq::inverse_transform(
      *res_ptr,
      *q,
      cuvs::core::from_dlpack<codes_mdspan_type>(codes_tensor),
      cuvs::core::from_dlpack<data_mdspan_type>(out_tensor));

  } else {
    RAFT_FAIL("codes must be accessible on device memory");
  }
}
}  // namespace

extern "C" cuvsError_t cuvsProductQuantizerParamsCreate(cuvsProductQuantizerParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params = new cuvsProductQuantizerParams{
      .pq_bits = 8, .pq_dim = 0, .vq_n_centers = 0, .kmeans_n_iters = 25,
      .vq_kmeans_trainset_fraction = 0, .pq_kmeans_trainset_fraction = 0,
      .pq_kmeans_type = cuvsKMeansType::KMeansBalanced, .max_train_points_per_pq_code = 256,
      .use_vq = false, .use_subspaces = true}; });
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
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
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
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsProductQuantizerInverseTransform(cuvsResources_t res,
                                                            cuvsProductQuantizer_t quantizer,
                                                            DLManagedTensor* codes_tensor,
                                                            DLManagedTensor* out_tensor)
{
  return cuvs::core::translate_exceptions([=] {
    auto out_dtype = out_tensor->dl_tensor.dtype;
    if (out_dtype.code == kDLFloat && out_dtype.bits == 32) {
      _inverse_transform<float>(res, quantizer, codes_tensor, out_tensor);
    } else {
      RAFT_FAIL("Unsupported out DLtensor dtype: %d and bits: %d",
                out_dtype.code,
                out_dtype.bits);
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
          (reinterpret_cast<cuvs::preprocessing::quantize::pq::quantizer<float>*>(quant_addr))
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
          (reinterpret_cast<cuvs::preprocessing::quantize::pq::quantizer<float>*>(quant_addr))
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

extern "C" cuvsError_t cuvsProductQuantizerGetPqCodebook(cuvsProductQuantizer_t quantizer,
                                                         DLManagedTensor* pq_codebook)
{
  return cuvs::core::translate_exceptions([=] {
    if (quantizer != nullptr) {
      auto quant_addr = quantizer->addr;
      if (quantizer->dtype.code == kDLFloat && quantizer->dtype.bits == 32) {
        auto pq_mdspan =
          (reinterpret_cast<cuvs::preprocessing::quantize::pq::quantizer<float>*>(quant_addr))
            ->vpq_codebooks.pq_code_book.view();
        cuvs::core::to_dlpack(pq_mdspan, pq_codebook);
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

extern "C" cuvsError_t cuvsProductQuantizerGetVqCodebook(cuvsProductQuantizer_t quantizer,
                                                         DLManagedTensor* vq_codebook)
{
  return cuvs::core::translate_exceptions([=] {
    if (quantizer != nullptr) {
      auto quant_addr = quantizer->addr;
      if (quantizer->dtype.code == kDLFloat && quantizer->dtype.bits == 32) {
        auto pq_mdspan =
          (reinterpret_cast<cuvs::preprocessing::quantize::pq::quantizer<float>*>(quant_addr))
            ->vpq_codebooks.vq_code_book.view();
        cuvs::core::to_dlpack(pq_mdspan, vq_codebook);
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

extern "C" cuvsError_t cuvsProductQuantizerGetEncodedDim(cuvsProductQuantizer_t quantizer,
                                                        uint32_t* encoded_dim)
{
  return cuvs::core::translate_exceptions([=] {
    if (quantizer != nullptr) {
      auto quant_addr = quantizer->addr;
      if (quantizer->dtype.code == kDLFloat && quantizer->dtype.bits == 32) {
        *encoded_dim = cuvs::preprocessing::quantize::pq::get_quantized_dim(
          reinterpret_cast<cuvs::preprocessing::quantize::pq::quantizer<float>*>(quant_addr)->params_quantizer);
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

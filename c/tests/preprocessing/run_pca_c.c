/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/preprocessing/pca.h>
#include <stdbool.h>
#include <stdint.h>

void run_pca(int64_t n_rows,
             int64_t n_cols,
             int n_components,
             float* input_data,
             float* trans_data,
             float* components_data,
             float* explained_var_data,
             float* explained_var_ratio_data,
             float* singular_vals_data,
             float* mu_data,
             float* noise_vars_data,
             float* output_data)
{
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  cuvsPcaParams_t params;
  cuvsPcaParamsCreate(&params);
  params->n_components = n_components;

  /* col-major 2D tensors need explicit strides: [1, n_rows] */

  /* input [n_rows x n_cols], col-major */
  DLManagedTensor input_tensor;
  input_tensor.dl_tensor.data               = input_data;
  input_tensor.dl_tensor.device.device_type  = kDLCUDA;
  input_tensor.dl_tensor.device.device_id    = 0;
  input_tensor.dl_tensor.ndim               = 2;
  input_tensor.dl_tensor.dtype.code         = kDLFloat;
  input_tensor.dl_tensor.dtype.bits         = 32;
  input_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t input_shape[2]                    = {n_rows, n_cols};
  int64_t input_strides[2]                  = {1, n_rows};
  input_tensor.dl_tensor.shape              = input_shape;
  input_tensor.dl_tensor.strides            = input_strides;
  input_tensor.dl_tensor.byte_offset        = 0;
  input_tensor.manager_ctx                  = NULL;
  input_tensor.deleter                      = NULL;

  /* trans_input [n_rows x n_components], col-major */
  DLManagedTensor trans_tensor;
  trans_tensor.dl_tensor.data               = trans_data;
  trans_tensor.dl_tensor.device.device_type  = kDLCUDA;
  trans_tensor.dl_tensor.device.device_id    = 0;
  trans_tensor.dl_tensor.ndim               = 2;
  trans_tensor.dl_tensor.dtype.code         = kDLFloat;
  trans_tensor.dl_tensor.dtype.bits         = 32;
  trans_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t trans_shape[2]                    = {n_rows, n_components};
  int64_t trans_strides[2]                  = {1, n_rows};
  trans_tensor.dl_tensor.shape              = trans_shape;
  trans_tensor.dl_tensor.strides            = trans_strides;
  trans_tensor.dl_tensor.byte_offset        = 0;
  trans_tensor.manager_ctx                  = NULL;
  trans_tensor.deleter                      = NULL;

  /* components [n_components x n_cols], col-major */
  DLManagedTensor comp_tensor;
  comp_tensor.dl_tensor.data               = components_data;
  comp_tensor.dl_tensor.device.device_type  = kDLCUDA;
  comp_tensor.dl_tensor.device.device_id    = 0;
  comp_tensor.dl_tensor.ndim               = 2;
  comp_tensor.dl_tensor.dtype.code         = kDLFloat;
  comp_tensor.dl_tensor.dtype.bits         = 32;
  comp_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t comp_shape[2]                    = {n_components, n_cols};
  int64_t comp_strides[2]                  = {1, n_components};
  comp_tensor.dl_tensor.shape              = comp_shape;
  comp_tensor.dl_tensor.strides            = comp_strides;
  comp_tensor.dl_tensor.byte_offset        = 0;
  comp_tensor.manager_ctx                  = NULL;
  comp_tensor.deleter                      = NULL;

  /* explained_var [n_components], 1D */
  DLManagedTensor ev_tensor;
  ev_tensor.dl_tensor.data               = explained_var_data;
  ev_tensor.dl_tensor.device.device_type  = kDLCUDA;
  ev_tensor.dl_tensor.device.device_id    = 0;
  ev_tensor.dl_tensor.ndim               = 1;
  ev_tensor.dl_tensor.dtype.code         = kDLFloat;
  ev_tensor.dl_tensor.dtype.bits         = 32;
  ev_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t ev_shape[1]                    = {n_components};
  ev_tensor.dl_tensor.shape              = ev_shape;
  ev_tensor.dl_tensor.strides            = NULL;
  ev_tensor.dl_tensor.byte_offset        = 0;
  ev_tensor.manager_ctx                  = NULL;
  ev_tensor.deleter                      = NULL;

  /* explained_var_ratio [n_components], 1D */
  DLManagedTensor evr_tensor;
  evr_tensor.dl_tensor.data               = explained_var_ratio_data;
  evr_tensor.dl_tensor.device.device_type  = kDLCUDA;
  evr_tensor.dl_tensor.device.device_id    = 0;
  evr_tensor.dl_tensor.ndim               = 1;
  evr_tensor.dl_tensor.dtype.code         = kDLFloat;
  evr_tensor.dl_tensor.dtype.bits         = 32;
  evr_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t evr_shape[1]                    = {n_components};
  evr_tensor.dl_tensor.shape              = evr_shape;
  evr_tensor.dl_tensor.strides            = NULL;
  evr_tensor.dl_tensor.byte_offset        = 0;
  evr_tensor.manager_ctx                  = NULL;
  evr_tensor.deleter                      = NULL;

  /* singular_vals [n_components], 1D */
  DLManagedTensor sv_tensor;
  sv_tensor.dl_tensor.data               = singular_vals_data;
  sv_tensor.dl_tensor.device.device_type  = kDLCUDA;
  sv_tensor.dl_tensor.device.device_id    = 0;
  sv_tensor.dl_tensor.ndim               = 1;
  sv_tensor.dl_tensor.dtype.code         = kDLFloat;
  sv_tensor.dl_tensor.dtype.bits         = 32;
  sv_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t sv_shape[1]                    = {n_components};
  sv_tensor.dl_tensor.shape              = sv_shape;
  sv_tensor.dl_tensor.strides            = NULL;
  sv_tensor.dl_tensor.byte_offset        = 0;
  sv_tensor.manager_ctx                  = NULL;
  sv_tensor.deleter                      = NULL;

  /* mu [n_cols], 1D */
  DLManagedTensor mu_tensor;
  mu_tensor.dl_tensor.data               = mu_data;
  mu_tensor.dl_tensor.device.device_type  = kDLCUDA;
  mu_tensor.dl_tensor.device.device_id    = 0;
  mu_tensor.dl_tensor.ndim               = 1;
  mu_tensor.dl_tensor.dtype.code         = kDLFloat;
  mu_tensor.dl_tensor.dtype.bits         = 32;
  mu_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t mu_shape[1]                    = {n_cols};
  mu_tensor.dl_tensor.shape              = mu_shape;
  mu_tensor.dl_tensor.strides            = NULL;
  mu_tensor.dl_tensor.byte_offset        = 0;
  mu_tensor.manager_ctx                  = NULL;
  mu_tensor.deleter                      = NULL;

  /* noise_vars [1], scalar stored as 1D */
  DLManagedTensor nv_tensor;
  nv_tensor.dl_tensor.data               = noise_vars_data;
  nv_tensor.dl_tensor.device.device_type  = kDLCUDA;
  nv_tensor.dl_tensor.device.device_id    = 0;
  nv_tensor.dl_tensor.ndim               = 1;
  nv_tensor.dl_tensor.dtype.code         = kDLFloat;
  nv_tensor.dl_tensor.dtype.bits         = 32;
  nv_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t nv_shape[1]                    = {1};
  nv_tensor.dl_tensor.shape              = nv_shape;
  nv_tensor.dl_tensor.strides            = NULL;
  nv_tensor.dl_tensor.byte_offset        = 0;
  nv_tensor.manager_ctx                  = NULL;
  nv_tensor.deleter                      = NULL;

  /* output [n_rows x n_cols], col-major */
  DLManagedTensor output_tensor;
  output_tensor.dl_tensor.data               = output_data;
  output_tensor.dl_tensor.device.device_type  = kDLCUDA;
  output_tensor.dl_tensor.device.device_id    = 0;
  output_tensor.dl_tensor.ndim               = 2;
  output_tensor.dl_tensor.dtype.code         = kDLFloat;
  output_tensor.dl_tensor.dtype.bits         = 32;
  output_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t output_shape[2]                    = {n_rows, n_cols};
  int64_t output_strides[2]                  = {1, n_rows};
  output_tensor.dl_tensor.shape              = output_shape;
  output_tensor.dl_tensor.strides            = output_strides;
  output_tensor.dl_tensor.byte_offset        = 0;
  output_tensor.manager_ctx                  = NULL;
  output_tensor.deleter                      = NULL;

  cuvsPcaFitTransform(res,
                      params,
                      &input_tensor,
                      &trans_tensor,
                      &comp_tensor,
                      &ev_tensor,
                      &evr_tensor,
                      &sv_tensor,
                      &mu_tensor,
                      &nv_tensor,
                      false);

  cuvsPcaInverseTransform(
    res, params, &trans_tensor, &comp_tensor, &sv_tensor, &mu_tensor, &output_tensor);

  cuvsPcaParamsDestroy(params);
  cuvsResourcesDestroy(res);
}

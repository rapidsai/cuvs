/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

 #include <cuvs/cluster/kmeans.h>
 #include <cuvs/core/c_api.h>
 #include <dlpack/dlpack.h>
 #include <stdbool.h>
 #include <stddef.h>
 #include <stdint.h>

 static void fill_matrix_tensor(DLManagedTensor* t,
                                void* data,
                                int64_t* shape,
                                DLDeviceType device_type,
                                uint8_t code,
                                uint8_t bits)
 {
   t->dl_tensor.data               = data;
   t->dl_tensor.device.device_type = device_type;
   t->dl_tensor.device.device_id   = 0;
   t->dl_tensor.ndim               = 2;
   t->dl_tensor.dtype.code         = code;
   t->dl_tensor.dtype.bits         = bits;
   t->dl_tensor.dtype.lanes        = 1;
   t->dl_tensor.shape              = shape;
   t->dl_tensor.strides            = NULL;
   t->dl_tensor.byte_offset        = 0;
   t->manager_ctx                  = NULL;
   t->deleter                      = NULL;
 }

 static void fill_vector_tensor(DLManagedTensor* t,
                                void* data,
                                int64_t* shape,
                                DLDeviceType device_type,
                                uint8_t code,
                                uint8_t bits)
 {
   t->dl_tensor.data               = data;
   t->dl_tensor.device.device_type = device_type;
   t->dl_tensor.device.device_id   = 0;
   t->dl_tensor.ndim               = 1;
   t->dl_tensor.dtype.code         = code;
   t->dl_tensor.dtype.bits         = bits;
   t->dl_tensor.dtype.lanes        = 1;
   t->dl_tensor.shape              = shape;
   t->dl_tensor.strides            = NULL;
   t->dl_tensor.byte_offset        = 0;
   t->manager_ctx                  = NULL;
   t->deleter                      = NULL;
 }

 /**
  * Run KMeans fit + (optional) predict + cluster_cost using the C API.
  *
  * If `dataset_on_host` is true, `dataset_data` is a host pointer, otherwise it is a
  * device pointer. `centroids_data` and `labels_data` are always device pointers.
  *
  * `predict_inertia_out`/`labels_data`/`cluster_cost_out` are only used when
  * `dataset_on_host` is false (predict + cluster_cost require device data).
  */
 void run_kmeans(int64_t n_samples,
                    int64_t n_features,
                    int n_clusters,
                    int max_iter,
                    double tol,
                    cuvsKMeansInitMethod init,
                    bool dataset_on_host,
                    int64_t streaming_batch_size,
                    void* dataset_data,
                    float* centroids_data,
                    int32_t* labels_data,
                    double* inertia_out,
                    int* n_iter_out,
                    double* predict_inertia_out,
                    double* cluster_cost_out)
 {
   cuvsResources_t res;
   cuvsResourcesCreate(&res);

   cuvsKMeansParams_t params;
   cuvsKMeansParamsCreate(&params);
   params->n_clusters           = n_clusters;
   params->max_iter             = max_iter;
   params->tol                  = tol;
   params->init                 = init;
   params->streaming_batch_size = streaming_batch_size;

   DLManagedTensor dataset_tensor;
   int64_t dataset_shape[2] = {n_samples, n_features};
   fill_matrix_tensor(&dataset_tensor,
                      dataset_data,
                      dataset_shape,
                      dataset_on_host ? kDLCPU : kDLCUDA,
                      kDLFloat,
                      32);

   DLManagedTensor centroids_tensor;
   int64_t centroids_shape[2] = {n_clusters, n_features};
   fill_matrix_tensor(
     &centroids_tensor, centroids_data, centroids_shape, kDLCUDA, kDLFloat, 32);

   cuvsKMeansFit(
     res, params, &dataset_tensor, NULL, &centroids_tensor, inertia_out, n_iter_out);

   if (!dataset_on_host) {
     DLManagedTensor labels_tensor;
     int64_t labels_shape[1] = {n_samples};
     fill_vector_tensor(&labels_tensor, labels_data, labels_shape, kDLCUDA, kDLInt, 32);

     cuvsKMeansPredict(res,
                       params,
                       &dataset_tensor,
                       NULL,
                       &centroids_tensor,
                       &labels_tensor,
                       false,
                       predict_inertia_out);

     cuvsKMeansClusterCost(res, &dataset_tensor, &centroids_tensor, cluster_cost_out);
   }

   cuvsKMeansParamsDestroy(params);
   cuvsResourcesDestroy(res);
 }

/**
 * Run KMeans fit + (optional) predict + cluster_cost.
 *
 * TODO(cuVS 26.08): delete run_kmeans_v2 once the `_v2` entry points
 * (cuvsKMeansFit_v2 / cuvsKMeansPredict_v2 / cuvsKMeansParamsCreate_v2 /
 * cuvsKMeansParamsDestroy_v2) are promoted to the unsuffixed names in the
 * public header.
 */
void run_kmeans_v2(int64_t n_samples,
                    int64_t n_features,
                    int n_clusters,
                    int max_iter,
                    double tol,
                    cuvsKMeansInitMethod init,
                    bool dataset_on_host,
                    int64_t streaming_batch_size,
                    void* dataset_data,
                    float* centroids_data,
                    int32_t* labels_data,
                    double* inertia_out,
                    int* n_iter_out,
                    double* predict_inertia_out,
                    double* cluster_cost_out)
 {
   cuvsResources_t res;
   cuvsResourcesCreate(&res);

   cuvsKMeansParams_v2_t params;
   cuvsKMeansParamsCreate_v2(&params);
   params->n_clusters           = n_clusters;
   params->max_iter             = max_iter;
   params->tol                  = tol;
   params->init                 = init;
   params->streaming_batch_size = streaming_batch_size;

   DLManagedTensor dataset_tensor;
   int64_t dataset_shape[2] = {n_samples, n_features};
   fill_matrix_tensor(&dataset_tensor,
                      dataset_data,
                      dataset_shape,
                      dataset_on_host ? kDLCPU : kDLCUDA,
                      kDLFloat,
                      32);

   DLManagedTensor centroids_tensor;
   int64_t centroids_shape[2] = {n_clusters, n_features};
   fill_matrix_tensor(
     &centroids_tensor, centroids_data, centroids_shape, kDLCUDA, kDLFloat, 32);

   cuvsKMeansFit_v2(
     res, params, &dataset_tensor, NULL, &centroids_tensor, inertia_out, n_iter_out);

   if (!dataset_on_host) {
     DLManagedTensor labels_tensor;
     int64_t labels_shape[1] = {n_samples};
     fill_vector_tensor(&labels_tensor, labels_data, labels_shape, kDLCUDA, kDLInt, 32);

     cuvsKMeansPredict_v2(res,
                          params,
                          &dataset_tensor,
                          NULL,
                          &centroids_tensor,
                          &labels_tensor,
                          false,
                          predict_inertia_out);

     cuvsKMeansClusterCost(res, &dataset_tensor, &centroids_tensor, cluster_cost_out);
   }

   cuvsKMeansParamsDestroy_v2(params);
   cuvsResourcesDestroy(res);
 }

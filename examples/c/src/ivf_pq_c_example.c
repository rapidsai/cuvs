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

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/ivf_pq.h>
#include <cuvs/neighbors/refine.h>

#include <cuda_runtime.h>
#include "common.h"

void ivf_pq_build_search(cuvsResources_t *res, DLManagedTensor * dataset_tensor, DLManagedTensor * queries_tensor) {
    // Create default index params
    cuvsIvfPqIndexParams_t index_params;
    cuvsIvfPqIndexParamsCreate(&index_params);
    index_params->n_lists                  = 1024; // default value
    index_params->kmeans_trainset_fraction = 0.1;
    //index_params->metric default is L2Expanded
    index_params->pq_bits = 8;
    index_params->pq_dim = 2;

    // Create IVF-PQ index
    cuvsIvfPqIndex_t index;
    cuvsIvfPqIndexCreate(&index);

    printf("Building IVF-PQ index\n");

    // Build the IVF-PQ Index
    cuvsError_t build_status = cuvsIvfPqBuild(*res, index_params, dataset_tensor, index);
    if (build_status != CUVS_SUCCESS) {
        printf("%s.\n", cuvsGetLastErrorText());
        cuvsIvfPqIndexDestroy(index);
        cuvsIvfPqIndexParamsDestroy(index_params);
        return;
    }
    
    // Create output arrays.
    int64_t topk      = 10;
    int64_t n_queries = queries_tensor->dl_tensor.shape[0];

    //Allocate memory for `neighbors` and `distances` output
    int64_t *neighbors_d;
    float *distances_d;
    cuvsRMMAlloc(*res, (void**) &neighbors_d, sizeof(int64_t) * n_queries * topk);
    cuvsRMMAlloc(*res, (void**) &distances_d, sizeof(float) * n_queries * topk);

    DLManagedTensor neighbors_tensor;
    int64_t neighbors_shape[2] = {n_queries, topk};
    int_tensor_initialize(neighbors_d, neighbors_shape, &neighbors_tensor);

    DLManagedTensor distances_tensor;
    int64_t distances_shape[2] = {n_queries, topk};
    float_tensor_initialize(distances_d, distances_shape, &distances_tensor);

    // Create default search params
    cuvsIvfPqSearchParams_t search_params;
    cuvsIvfPqSearchParamsCreate(&search_params);
    search_params->n_probes = 50;
    search_params->internal_distance_dtype = CUDA_R_16F;
    search_params->lut_dtype = CUDA_R_16F;

    // Search the `index` built using `cuvsIvfPqBuild`
    cuvsError_t search_status = cuvsIvfPqSearch(*res, search_params, index,
     queries_tensor, &neighbors_tensor, &distances_tensor);
    if (build_status != CUVS_SUCCESS) {
        printf("%s.\n", cuvsGetLastErrorText());
    }

    int64_t *neighbors = (int64_t *)malloc(n_queries * topk * sizeof(int64_t));
    float *distances = (float *)malloc(n_queries * topk * sizeof(float));
    memset(neighbors, 0, n_queries * topk * sizeof(int64_t));
    memset(distances, 0, n_queries * topk * sizeof(float));

    cudaMemcpy(neighbors, neighbors_d, sizeof(int64_t) * n_queries * topk, cudaMemcpyDefault);
    cudaMemcpy(distances, distances_d, sizeof(float) * n_queries * topk, cudaMemcpyDefault);

    printf("\nOriginal results:\n");
    print_results(neighbors, distances, 2, topk);
    
    // Re-ranking operation: refine the initial search results by computing exact distances
    int64_t topk_refined = 7;
    int64_t *neighbors_refined_d;
    float *distances_refined_d;
    cuvsRMMAlloc(*res, (void**) &neighbors_refined_d, sizeof(int64_t) * n_queries * topk_refined);
    cuvsRMMAlloc(*res, (void**) &distances_refined_d, sizeof(float) * n_queries * topk_refined);

    DLManagedTensor neighbors_refined_tensor;
    int64_t neighbors_refined_shape[2] = {n_queries, topk_refined};
    int_tensor_initialize(neighbors_refined_d, neighbors_refined_shape, &neighbors_refined_tensor);
    
    DLManagedTensor distances_refined_tensor;
    int64_t distances_refined_shape[2] = {n_queries, topk_refined};
    float_tensor_initialize(distances_refined_d, distances_refined_shape, &distances_refined_tensor);
    
    // Note, refinement requires the original dataset and the queries.
    // Don't forget to specify the same distance metric as used by the index.
    cuvsError_t refine_status = cuvsRefine(*res, dataset_tensor, queries_tensor,
                       &neighbors_tensor, index_params->metric,
                       &neighbors_refined_tensor, &distances_refined_tensor);
    if (refine_status != CUVS_SUCCESS) {
        printf("%s.\n", cuvsGetLastErrorText());
    }

    int64_t *neighbors_refine = (int64_t *)malloc(n_queries * topk_refined * sizeof(int64_t));
    float *distances_refine = (float *)malloc(n_queries * topk_refined * sizeof(float));
    memset(neighbors_refine, 0, n_queries * topk_refined * sizeof(int64_t));
    memset(distances_refine, 0, n_queries * topk_refined * sizeof(float));

    cudaMemcpy(neighbors_refine, neighbors_refined_d, sizeof(int64_t) * n_queries * topk_refined, cudaMemcpyDefault);
    cudaMemcpy(distances_refine, distances_refined_d, sizeof(float) * n_queries * topk_refined, cudaMemcpyDefault);

    printf("\nRefined results:\n");
    print_results(neighbors, distances, 2, topk_refined);

    free(distances_refine);
    free(neighbors_refine);

    free(distances);
    free(neighbors);

    cuvsRMMFree(*res, neighbors_refined_d, sizeof(int64_t) * n_queries * topk_refined);
    cuvsRMMFree(*res, distances_refined_d, sizeof(float) * n_queries * topk_refined);

    cuvsRMMFree(*res, neighbors_d, sizeof(int64_t) * n_queries * topk);
    cuvsRMMFree(*res, distances_d, sizeof(float) * n_queries * topk);

    cuvsIvfPqSearchParamsDestroy(search_params);
    cuvsIvfPqIndexDestroy(index);
    cuvsIvfPqIndexParamsDestroy(index_params);  
}

int main() {
    // Create input arrays.
    int64_t n_samples = 10000;
    int64_t n_dim     = 3;
    int64_t n_queries = 10;
    float *dataset = (float *)malloc(n_samples * n_dim * sizeof(float));
    float *queries = (float *)malloc(n_queries * n_dim * sizeof(float));
    generate_dataset(dataset, n_samples, n_dim, -10.0, 10.0);
    generate_dataset(queries, n_queries, n_dim, -1.0, 1.0);
    
    // Create a cuvsResources_t object
    cuvsResources_t res;
    cuvsResourcesCreate(&res);

    // Allocate memory for `queries`
    float *dataset_d;
    cuvsRMMAlloc(res, (void**) &dataset_d, sizeof(float) * n_samples * n_dim);
    // Use DLPack to represent `dataset_d` as a tensor
    cudaMemcpy(dataset_d, dataset, sizeof(float) * n_samples * n_dim, cudaMemcpyDefault);

    DLManagedTensor dataset_tensor;
    int64_t dataset_shape[2] = {n_samples,n_dim};
    float_tensor_initialize(dataset_d, dataset_shape, &dataset_tensor);
    
    // Allocate memory for `queries`
    float *queries_d;
    cuvsRMMAlloc(res, (void**) &queries_d, sizeof(float) * n_queries * n_dim);

    // Use DLPack to represent `queries` as tensors
    cudaMemcpy(queries_d, queries, sizeof(float) * n_queries * n_dim, cudaMemcpyDefault);

    DLManagedTensor queries_tensor;
    int64_t queries_shape[2] = {n_queries, n_dim};
    float_tensor_initialize(queries_d, queries_shape, &queries_tensor);
    
    // Simple build and search example.
    ivf_pq_build_search(&res, &dataset_tensor, &queries_tensor);

    cuvsRMMFree(res, queries_d, sizeof(float) * n_queries * n_dim);
    cuvsRMMFree(res, dataset_d, sizeof(float) * n_samples * n_dim);
    cuvsResourcesDestroy(res);
    free(dataset);
    free(queries);
}

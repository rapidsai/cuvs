#include <cuvs/core/c_api.h>
#include <cuvs/distance/pairwise_distance.h>
#include <cuvs/neighbors/cagra.h>

#include <dlpack/dlpack.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM 4
#define N_ROWS 1

float PointA[N_ROWS][DIM] = {1.0,2.0,3.0,4.0};
float PointB[N_ROWS][DIM] = {2.0,3.0,4.0,5.0};

void outputVector(float * Vec) {
  printf("Vector is ");
  for (int i = 0; i < DIM; ++i){
    printf(" %f",Vec[i]);
  }
  printf("\n");
}

cuvsResources_t res;

int initResources(void){
  // Create a cuvsResources_t object
  cuvsResourcesCreate(&res);
  return 0;
}

int freeResources(void){
  cuvsResourcesDestroy(res);
  return 0;
}

/* Started by AICoder, pid:96cd0bcb6ad27b1149870b5f20ef16170f62a328 */
void initializeTensor(float* x_d, int64_t x_shape[2], DLManagedTensor* x_tensor) {
  x_tensor->dl_tensor.data = x_d;
  x_tensor->dl_tensor.device.device_type = kDLCUDA;
  x_tensor->dl_tensor.ndim = 2;
  x_tensor->dl_tensor.dtype.code = kDLFloat;
  x_tensor->dl_tensor.dtype.bits = 32;
  x_tensor->dl_tensor.dtype.lanes = 1;
  x_tensor->dl_tensor.shape = x_shape;
  x_tensor->dl_tensor.strides = NULL;
}
/* Ended by AICoder, pid:96cd0bcb6ad27b1149870b5f20ef16170f62a328 */

int calcL2Distance(int64_t n_cols,float x[], float y[], float *ret) {
  float *x_d, *y_d;
  float *distance_d;
  cuvsRMMAlloc(res, (void**) &x_d, sizeof(float) * N_ROWS * n_cols);
  cuvsRMMAlloc(res, (void**) &y_d, sizeof(float) * N_ROWS * n_cols);
  cuvsRMMAlloc(res, (void**) &distance_d, sizeof(float) * N_ROWS * N_ROWS);

  // Use DLPack to represent x[] and y[] as tensors
  cudaMemcpy(x_d, x, sizeof(float) * N_ROWS * n_cols, cudaMemcpyDefault);
  cudaMemcpy(y_d, y, sizeof(float) * N_ROWS * n_cols, cudaMemcpyDefault);

  DLManagedTensor x_tensor;
  int64_t x_shape[2] = {N_ROWS, n_cols};
  initializeTensor(x_d, x_shape, &x_tensor);

  DLManagedTensor y_tensor;
  int64_t y_shape[2] = {N_ROWS, n_cols};
  initializeTensor(y_d, y_shape, &y_tensor);
  
  DLManagedTensor dist_tensor;
  int64_t distances_shape[2] = {N_ROWS, N_ROWS};
  initializeTensor(distance_d, distances_shape, &dist_tensor);

  // metric_arg default value is 2.0,used for Minkowski distance
  cuvsPairwiseDistance(res, &x_tensor, &y_tensor, &dist_tensor, L2SqrtUnexpanded, 2.0);

  cudaMemcpy(ret, distance_d, sizeof(float) * N_ROWS * N_ROWS, cudaMemcpyDefault);
  
  cuvsRMMFree(res, distance_d, sizeof(float) * N_ROWS * N_ROWS);
  cuvsRMMFree(res, x_d, sizeof(float) * N_ROWS * n_cols);
  cuvsRMMFree(res, y_d, sizeof(float) * N_ROWS * n_cols);

  return 0;
}

int euclidean_distance_calculation_example() {
  float ret;

  outputVector((float *)PointA);
  outputVector((float *)PointB);

  initResources();
  
  calcL2Distance(DIM, (float *)PointA, (float *)PointB, &ret);
  printf("L2 distance is %f.\n", ret);
  
  freeResources();

  return 0;
}

int main() {
    euclidean_distance_calculation_example();
    return 0;
}

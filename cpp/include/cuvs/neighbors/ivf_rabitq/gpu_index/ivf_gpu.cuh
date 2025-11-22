/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 2/23/25.
//

#pragma once

#include <raft/core/resources.hpp>

#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuvs/neighbors/ivf_rabitq/defines.hpp>
#include <cuvs/neighbors/ivf_rabitq/gpu_index/initializer_gpu.cuh>
#include <cuvs/neighbors/ivf_rabitq/gpu_index/pool_gpu.cuh>
#include <cuvs/neighbors/ivf_rabitq/gpu_index/quantizer_gpu.cuh>
#include <cuvs/neighbors/ivf_rabitq/gpu_index/rotator_gpu.cuh>
#include <cuvs/neighbors/ivf_rabitq/utils/utils_cuda.cuh>
#include <vector>

namespace cuvs::neighbors::ivf_rabitq::detail {

// Structure for cluster-query pairs
struct ClusterQueryPair {
  int cluster_idx;
  int query_idx;
};

class IVFGPU {
 public:
  class GPUClusterMeta {
   public:
    size_t num;          // Number of vectors in this cluster.
    size_t iter;         // Number of iterations for FastScan.
    size_t remain;       // Number of leftover vectors after blocks.
    size_t start_index;  // Combined offset: index of first vector in the flattened arrays.

    // Constructor: computes iter and REMAIN based on FAST_SIZE.
    __host__ GPUClusterMeta(size_t num, size_t start_idx) : num(num), start_index(start_idx)
    {
      iter   = num / FAST_SIZE;
      remain = num - iter * FAST_SIZE;
    }

    // default constructor (useful for vector initialize)
    __host__ GPUClusterMeta() : num(0), iter(0), remain(0), start_index(0) {}

    // Copy constructor.
    __host__ GPUClusterMeta(const GPUClusterMeta& other) = default;

    // Move constructor.（Would there be a memory leak if using this?
    __host__ GPUClusterMeta(GPUClusterMeta&& other) noexcept
      : num(other.num), iter(other.iter), remain(other.remain), start_index(other.start_index)
    {
    }

    __host__ __device__ GPUClusterMeta& operator=(const GPUClusterMeta& other)
    {
      // self-assignment check is cheap and avoids UB if someone
      // ever writes “meta = meta;”
      if (this != &other) {
        num         = other.num;
        iter        = other.iter;
        remain      = other.remain;
        start_index = other.start_index;
      }
      return *this;
    }
    // Destructor.
    __host__ __device__ ~GPUClusterMeta() = default;

    /**
     * @brief Compute pointer to the first block of quantized short data. note that short code
     * length and factors are counted in 32bits (uint32_t and float)
     *
     * @param parent Pointer to the IVFGPU instance that holds the base pointer.
     * @return Pointer to the first block of this cluster’s short data.
     */
    // jamxia edit
    // __host__ __device__
    __host__ uint32_t* first_block(const IVFGPU& parent) const
    {
      return parent.d_short_data +
             start_index * (parent.DQ.short_code_length() + parent.DQ.num_short_factors());
    }

    // for batch data, factors are stored separately.
    uint32_t* first_block_batch(const IVFGPU& parent) const
    {
      return parent.d_short_data + start_index * (parent.DQ.short_code_length());
    }

    __device__ uint32_t* first_block_gpu(uint32_t* d_short_data,
                                         size_t short_code_length,
                                         size_t num_short_factors) const
    {
      return d_short_data + start_index * (short_code_length + num_short_factors);
    }

    // jamxia edit
    // __host__ __device__
    __host__ uint32_t* first_block_host(const IVFGPU& parent) const
    {
      return parent.h_short_data +
             start_index * (parent.DQ.short_code_length() + parent.DQ.num_short_factors());
    }

    float* short_factor_batch(const IVFGPU& parent, size_t i) const
    {
      return parent.d_short_factors_batch + (start_index + i) * 3;
    }

    /**
     * @brief Get the pointer to the long code for the i-th vector in this cluster.
     *
     * @param parent Pointer to the IVFGPU instance that holds the base pointer.
     * @param i Index within the cluster (0 ≤ i < Num).
     * @param long_code_length Fixed length (in bytes) for each vector’s long code.
     * @return Pointer to the long code for the i-th vector.
     */
    __host__ __device__ uint8_t* long_code(const IVFGPU& parent,
                                           size_t i,
                                           size_t long_code_length) const
    {
      return parent.d_long_code + (start_index + i) * long_code_length;
    }

    __host__ __device__ uint8_t* long_code_host(const IVFGPU& parent,
                                                size_t i,
                                                size_t long_code_length) const
    {
      return parent.h_long_code + (start_index + i) * long_code_length;
    }

    /**
     * @brief Get the pointer to the extra factor for the i-th vector in this cluster.
     *
     * @param parent Pointer to the IVFGPU instance that holds the base pointer.
     * @param i Index within the cluster.
     * @return Pointer to the extra factor for the i-th vector.
     */
    __host__ __device__ ExFactor* ex_factor(const IVFGPU& parent, size_t i) const
    {
      return parent.d_ex_factor + start_index + i;
    }

    // two ex_factors are stored
    ExFactor* ex_factor_batch(const IVFGPU& parent, size_t i) const
    {
      return parent.d_ex_factor + (start_index + i) * 2;
    }

    ExFactor* ex_factor_host(const IVFGPU& parent, size_t i) const
    {
      return parent.h_ex_factor + start_index + i;
    }

    /**
     * @brief Get the pointer to the vector IDs for this cluster.
     *
     * @param parent Pointer to the IVFGPU instance that holds the base pointer.
     * @return Pointer to the first ID in this cluster.
     */
    __host__ __device__ PID* ids(const IVFGPU& parent) const { return parent.d_ids + start_index; }

    __host__ __device__ PID* ids_host(const IVFGPU& parent) const
    {
      return parent.h_ids + start_index;
    }
  };

  /**
   * @brief initialize function (no memory allocated yet)
   *
   * @param n Num of data points.
   * @param dim Dimension of data points.
   * @param k Num of centroids.
   * @param bits_per_dim totalbits = EX_BITS+1
   */
  IVFGPU(raft::resources const& handle,
         size_t n,
         size_t dim,
         size_t k,
         size_t bits_per_dim,
         bool batch_flag);
  IVFGPU(raft::resources const& handle)
    : Rota(handle, 128),
      d_short_data(nullptr),
      d_long_code(nullptr),
      d_short_factors_batch(nullptr),
      d_ex_factor(nullptr),
      d_ids(nullptr),
      initializer(nullptr),
      d_cluster_meta(nullptr),
      batch_flag(false)
  {
  }

  ~IVFGPU();

  /**
   * @brief Build function
   *
   * @param host_data pointer to host data.
   * @param host_centroids pointer to centroids.
   * @param pids PIDs of vectors.
   */
  void construct(raft::resources const& handle,
                 const float* host_data,
                 const float* host_centroids,
                 const uint32_t* pids,
                 bool fast_quantize = false);

  /**
   * @brief ANN search
   *
   * @param host_query pointer to query vector (currently on host)
   * @param results pid results function
   * @param k number of nearest neighbors to retrieve.
   * @param nprobe number of nearest clusters to probe.
   */
  void search(raft::resources const& handle,
              const float* d_query,
              size_t k,
              size_t nprobe,
              PID* results,
              cudaStream_t single_stream = nullptr) const;
  //    void search(const float* host_query, float* results, size_t k, size_t nprobe) const;
  void search_with_time(raft::resources const& handle,
                        const float* d_query,
                        size_t k,
                        size_t nprobe,
                        PID* results,
                        std::vector<int>& probe_hist) const;
  // Device pointers for each data array.
  InitializerGPU*
    initializer;  // Initializer, indicates which initializer to use (currently only FlatIVF)
  uint32_t* d_short_data;          // RaBitQ code and factors.
  uint8_t* d_long_code;            // ExRaBitQ code.
  ExFactor* d_ex_factor;           // ExRaBitQ factor.
  PID* d_ids;                      // PID of vectors。
  GPUClusterMeta* d_cluster_meta;  // Device-side array of clusters.  Replace std::vector with a raw
                                   // pointer for clusters.

  // batch-data
  bool batch_flag;
  //    uint32_t* d_short_data_batch;   // rabitq codes
  float* d_short_factors_batch;  // N * 3 float rabitq factors
  // d_long_code is the same
  // exfactor use the same place as before

  // host-side copies
  std::vector<GPUClusterMeta> h_cluster_meta;  // Host-side copy of clusters
  //    float* d_centroids;      // Device centroids (if needed for search, now stored in
  //    initializer).

  uint32_t* h_short_data;  // TODO: CPU side, we need on factors from h_short_data, so no need to
                           // store all these codes
  uint8_t* h_long_code;    // ExRaBitQ code.
  ExFactor* h_ex_factor;   // ExRaBitQ factor.
  PID* h_ids;              // PID of vectors。

  // Index meta-data.
  size_t num_vectors;     // Total number of vectors.
  size_t num_dimensions;  // Dimensionality of the input vectors.
  size_t num_padded_dim;  // padded dimensions
  size_t num_centroids;   // Centroids && Clusters
  size_t max_cluster_length;
  size_t ex_bits;  // Extra bits parameter for quantization.

  DataQuantizerGPU DQ;  // Data quantizer.
  RotatorGPU Rota;      // Data Rotator.

  // save_batch_flag and load_batch_flag are add for compatity with the previous non-batch index,
  // load_transposed only applies for new batch index
  void save(const char* filename, bool save_batch_flag = false) const;

  void load(raft::resources const& handle, const char* filename, bool load_batch_flag = false);

  void load_transposed(raft::resources const& handle, const char* filename);

  size_t padded_dim() { return this->num_padded_dim; }

  RotatorGPU& rotator() { return this->Rota; }

  size_t num_clusters() const { return num_centroids; }

  void MemOptimizedSearch(
    const float* d_query, size_t k, size_t nprobe, PID* results, void* searcher) const;

  void CPUGPUCoSearch(
    const float* d_query, size_t k, size_t nprobe, PID* results, void* searcher1) const;

  void CPUGPUCoSearchV2(
    const float* d_query, size_t k, size_t nprobe, PID* results, void* searcher1) const;

  void MemOptimizedSearchV2(
    const float* d_query, size_t k, size_t nprobe, PID* results, void* searcher1) const;

  void MultiClusterSearch(const float* d_query,
                          size_t k,
                          size_t nprobe,
                          PID* results,
                          void* searcher1,
                          cudaStream_t single_stream,
                          DeviceResultPool** knn_array,
                          std::vector<Candidate>& centroid_candidates) const;

  void BatchClusterSearch(const float* d_query,
                          size_t k,
                          size_t nprobe,
                          void* searcher,
                          size_t batch_size,
                          float* d_topk_dists,
                          float* d_final_dists,
                          PID* d_topk_pids,
                          PID* d_final_pids,
                          cudaStream_t single_stream);

  void BatchClusterSearchPreComputeThresholds(const float* d_query,
                                              size_t k,
                                              size_t nprobe,
                                              void* searcher,
                                              size_t batch_size,
                                              float* d_topk_dists,
                                              float* d_final_dists,
                                              PID* d_topk_pids,
                                              PID* d_final_pids,
                                              cudaStream_t single_stream);

  void BatchClusterSearchLUT16(const float* d_query,
                               size_t k,
                               size_t nprobe,
                               void* searcher,
                               size_t batch_size,
                               float* d_topk_dists,
                               float* d_final_dists,
                               PID* d_topk_pids,
                               PID* d_final_pids,
                               cudaStream_t single_stream);

  void BatchClusterSearchQuantizeQuery(const float* d_query,
                                       size_t k,
                                       size_t nprobe,
                                       void* searcher,
                                       size_t batch_size,
                                       float* d_topk_dists,
                                       float* d_final_dists,
                                       PID* d_topk_pids,
                                       PID* d_final_pids,
                                       int query_bits,
                                       cudaStream_t single_stream);

 private:
  //    // Host copy of per-cluster metadata.
  //    std::vector<GPUClusterMeta> h_cluster_meta;

  /**
   * @brief function to allocate memory based on the cluster
   *
   * @param cluster_sizes
   */
  void AllocateDeviceMemory(size_t* cluster_sizes, size_t num_clusters);

  /**
   * @brief function to free all memory
   */
  void FreeDeviceMemory() const;

  // Following are inline functions to compute spaces for memory allocation

  // TODO: Check whether it is actually an inline function
  // now 1 block represent 1 binaried vector + factor
  //    size_t GetShortDataBytes(size_t* cluster_sizes, size_t num_clusters) const {
  //        assert(num_clusters == num_centroids);  // num of clusters
  //        size_t total_blocks = 0;
  //        for (auto s = 0; s < num_clusters; s++) {
  //            total_blocks += cluster_sizes[s];
  //        }
  //        return total_blocks * DQ.block_bytes();
  //    }

  size_t GetShortDataBytesSimple() const
  {
    //        assert(num_clusters == num_centroids);  // num of clusters
    //        size_t total_blocks = 0;
    //        for (auto s = 0; s < num_clusters; s++) {
    //            total_blocks += cluster_sizes[s];
    //        }
    return num_vectors * DQ.block_bytes();
  }

  size_t GetShortDataFactorBytesBatch() const
  {
    //        assert(num_clusters == num_centroids);  // num of clusters
    //        size_t total_blocks = 0;
    //        for (auto s = 0; s < num_clusters; s++) {
    //            total_blocks += cluster_sizes[s];
    //        }
    return num_vectors * 3 * sizeof(float);
  }

  size_t GetExFactorBytes() const
  {
    if (!batch_flag) {
      return sizeof(ExFactor) * num_vectors;
    } else {
      return 2 * sizeof(float) * num_vectors;  // only f_add_ex and f_rescale_ex
    }
  }

  size_t GetPIDsBytes() const { return sizeof(PID) * num_vectors; }

  size_t GetLongCodeBytes() const { return sizeof(uint8_t) * DQ.long_code_length() * num_vectors; }
  void init_clusters(const std::vector<size_t>& cluster_sizes);

  void quantize_cluster(raft::resources const& handle,
                        GPUClusterMeta& cp,
                        /*const std::vector<PID> &IDs,*/ const float* data,
                        const float* cur_centroid,
                        float* rotated_c) const;

  void AllocateHostMemory(size_t* cluster_sizes, size_t num_clusters);

  void FreeHostMemory() const;

  void BatchClusterSearchGather(const float* d_query,
                                size_t k,
                                size_t nprobe,
                                void* searcher,
                                size_t batch_size,
                                cudaStream_t single_stream);
};

}  // namespace cuvs::neighbors::ivf_rabitq::detail

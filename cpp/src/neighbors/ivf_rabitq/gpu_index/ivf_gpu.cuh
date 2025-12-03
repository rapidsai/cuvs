/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 2/23/25.
//

#pragma once

#include "../defines.hpp"
#include "../utils/utils_cuda.cuh"
#include "initializer_gpu.cuh"
#include "pool_gpu.cuh"
#include "quantizer_gpu.cuh"
#include "rotator_gpu.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
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
    GPUClusterMeta(size_t num, size_t start_idx) : num(num), start_index(start_idx)
    {
      iter   = num / FAST_SIZE;
      remain = num - iter * FAST_SIZE;
    }

    // default constructor (useful for vector initialize)
    GPUClusterMeta() : num(0), iter(0), remain(0), start_index(0) {}

    // Copy constructor.
    GPUClusterMeta(const GPUClusterMeta& other) = default;

    // Copy assignment.
    GPUClusterMeta& operator=(const GPUClusterMeta& other) = default;

    // Destructor.
    ~GPUClusterMeta() = default;

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
      return parent.get_short_data_device() +
             start_index *
               (parent.quantizer().short_code_length() + parent.quantizer().num_short_factors());
    }

    // for batch data, factors are stored separately.
    uint32_t* first_block_batch(const IVFGPU& parent) const
    {
      return parent.get_short_data_device() +
             start_index * (parent.quantizer().short_code_length());
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
      return parent.get_short_data_host() + start_index * (parent.quantizer().short_code_length() +
                                                           parent.quantizer().num_short_factors());
    }

    float* short_factor_batch(const IVFGPU& parent, size_t i) const
    {
      return parent.get_short_factors_batch_device() + (start_index + i) * 3;
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
      return parent.get_long_code_device() + (start_index + i) * long_code_length;
    }

    __host__ __device__ uint8_t* long_code_host(const IVFGPU& parent,
                                                size_t i,
                                                size_t long_code_length) const
    {
      return parent.get_long_code_host() + (start_index + i) * long_code_length;
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
      return parent.get_ex_factor_device() + start_index + i;
    }

    // two ex_factors are stored
    ExFactor* ex_factor_batch(const IVFGPU& parent, size_t i) const
    {
      return parent.get_ex_factor_device() + (start_index + i) * 2;
    }

    ExFactor* ex_factor_host(const IVFGPU& parent, size_t i) const
    {
      return parent.get_ex_factor_host() + start_index + i;
    }

    /**
     * @brief Get the pointer to the vector IDs for this cluster.
     *
     * @param parent Pointer to the IVFGPU instance that holds the base pointer.
     * @return Pointer to the first ID in this cluster.
     */
    __host__ __device__ PID* ids(const IVFGPU& parent) const
    {
      return parent.get_ids_device() + start_index;
    }

    __host__ __device__ PID* ids_host(const IVFGPU& parent) const
    {
      return parent.get_ids_host() + start_index;
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
    : handle_(handle),
      stream_(raft::resource::get_cuda_stream(handle_)),
      short_data_(raft::make_device_vector<uint32_t, int64_t>(handle_, 0)),
      long_code_(raft::make_device_vector<uint8_t, int64_t>(handle_, 0)),
      ex_factor_(raft::make_device_vector<ExFactor, int64_t>(handle_, 0)),
      ids_(raft::make_device_vector<PID, int64_t>(handle_, 0)),
      cluster_meta_(raft::make_device_vector<GPUClusterMeta, int64_t>(handle_, 0)),
      batch_flag(false),
      short_factors_batch_(raft::make_device_vector<float, int64_t>(handle_, 0)),
      short_data_host_(raft::make_host_vector<uint32_t, int64_t>(0)),
      long_code_host_(raft::make_host_vector<uint8_t, int64_t>(0)),
      ex_factor_host_(raft::make_host_vector<ExFactor, int64_t>(0)),
      ids_host_(raft::make_host_vector<PID, int64_t>(0)),
      cluster_meta_host_(raft::make_host_vector<GPUClusterMeta, int64_t>(0)),
      initializer(nullptr),
      Rota(std::make_unique<RotatorGPU>(handle_, 128))
  {
  }

  /**
   * @brief Build function
   *
   * @param host_data pointer to host data.
   * @param host_centroids pointer to centroids.
   * @param pids PIDs of vectors.
   */
  void construct(const float* host_data,
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
  void search(const float* d_query, size_t k, size_t nprobe, PID* results) const;
  //    void search(const float* host_query, float* results, size_t k, size_t nprobe) const;
  void search_with_time(const float* d_query,
                        size_t k,
                        size_t nprobe,
                        PID* results,
                        std::vector<int>& probe_hist) const;

  // save_batch_flag and load_batch_flag are add for compatity with the previous non-batch index,
  // load_transposed only applies for new batch index
  void save(const char* filename, bool save_batch_flag = false) const;

  void load(const char* filename, bool load_batch_flag = false);

  void load_transposed(const char* filename);

  // device data getters
  __host__ __device__ uint32_t* get_short_data_device() const noexcept
  {
    return const_cast<uint32_t*>(this->short_data_.data_handle());
  }
  __host__ __device__ uint8_t* get_long_code_device() const noexcept
  {
    return const_cast<uint8_t*>(this->long_code_.data_handle());
  }
  __host__ __device__ ExFactor* get_ex_factor_device() const noexcept
  {
    return const_cast<ExFactor*>(this->ex_factor_.data_handle());
  }
  __host__ __device__ PID* get_ids_device() const noexcept
  {
    return const_cast<PID*>(this->ids_.data_handle());
  }
  __host__ __device__ float* get_short_factors_batch_device() const noexcept
  {
    return const_cast<float*>(this->short_factors_batch_.data_handle());
  }

  // host data getters
  raft::host_vector<GPUClusterMeta, int64_t> const& get_cluster_meta_host()
  {
    return cluster_meta_host_;
  }
  __host__ __device__ uint32_t* get_short_data_host() const noexcept
  {
    return const_cast<uint32_t*>(short_data_host_.data_handle());
  }
  __host__ __device__ uint8_t* get_long_code_host() const noexcept
  {
    return const_cast<uint8_t*>(long_code_host_.data_handle());
  }
  __host__ __device__ ExFactor* get_ex_factor_host() const noexcept
  {
    return const_cast<ExFactor*>(ex_factor_host_.data_handle());
  }
  __host__ __device__ PID* get_ids_host() const noexcept
  {
    return const_cast<PID*>(ids_host_.data_handle());
  }

  // metadata getters
  size_t get_num_dimensions() const noexcept { return this->num_dimensions; }
  size_t get_num_padded_dim() const noexcept { return this->num_padded_dim; }
  size_t get_num_centroids() const { return num_centroids; }
  size_t get_max_cluster_length() const noexcept { return max_cluster_length; }
  size_t get_ex_bits() const noexcept { return ex_bits; }

  // member object getters
  DataQuantizerGPU& quantizer() const { return *(this->DQ); }
  RotatorGPU& rotator() const { return *(this->Rota); }

  // metadata setters
  void set_max_cluster_length(size_t new_max_cluster_length)
  {
    max_cluster_length = new_max_cluster_length;
  }

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
                          std::vector<DeviceResultPool>& knn_array,
                          std::vector<Candidate>& centroid_candidates) const;

  void BatchClusterSearch(const float* d_query,
                          size_t k,
                          size_t nprobe,
                          void* searcher,
                          size_t batch_size,
                          float* d_topk_dists,
                          float* d_final_dists,
                          PID* d_topk_pids,
                          PID* d_final_pids);

  void BatchClusterSearchPreComputeThresholds(const float* d_query,
                                              size_t k,
                                              size_t nprobe,
                                              void* searcher,
                                              size_t batch_size,
                                              float* d_topk_dists,
                                              float* d_final_dists,
                                              PID* d_topk_pids,
                                              PID* d_final_pids);

  void BatchClusterSearchLUT16(const float* d_query,
                               size_t k,
                               size_t nprobe,
                               void* searcher,
                               size_t batch_size,
                               float* d_topk_dists,
                               float* d_final_dists,
                               PID* d_topk_pids,
                               PID* d_final_pids);

  void BatchClusterSearchQuantizeQuery(const float* d_query,
                                       size_t k,
                                       size_t nprobe,
                                       void* searcher,
                                       size_t batch_size,
                                       float* d_topk_dists,
                                       float* d_final_dists,
                                       PID* d_topk_pids,
                                       PID* d_final_pids,
                                       int query_bits);

 private:
  /**
   * @brief function to allocate memory based on the cluster
   *
   */
  void AllocateDeviceMemory();

  // Following are inline functions to compute spaces for memory allocation

  // TODO: Check whether it is actually an inline function
  // now 1 block represent 1 binaried vector + factor
  //    size_t GetShortDataBytes(size_t* cluster_sizes, size_t num_clusters) const {
  //        assert(num_clusters == num_centroids);  // num of clusters
  //        size_t total_blocks = 0;
  //        for (auto s = 0; s < num_clusters; s++) {
  //            total_blocks += cluster_sizes[s];
  //        }
  //        return total_blocks * this->quantizer().block_bytes();
  //    }

  size_t GetShortDataBytesSimple() const
  {
    //        assert(num_clusters == num_centroids);  // num of clusters
    //        size_t total_blocks = 0;
    //        for (auto s = 0; s < num_clusters; s++) {
    //            total_blocks += cluster_sizes[s];
    //        }
    return num_vectors * this->quantizer().block_bytes();
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

  size_t GetLongCodeBytes() const
  {
    return sizeof(uint8_t) * quantizer().long_code_length() * num_vectors;
  }
  void init_clusters(const std::vector<size_t>& cluster_sizes);

  void quantize_cluster(GPUClusterMeta& cp,
                        /*const std::vector<PID> &IDs,*/ const float* data,
                        const float* cur_centroid,
                        float* rotated_c) const;

  void AllocateHostMemory();

  void BatchClusterSearchGather(const float* d_query,
                                size_t k,
                                size_t nprobe,
                                void* searcher,
                                size_t batch_size,
                                rmm::cuda_stream_view single_stream);

  raft::resources const& handle_;  // reusable resource handle
  rmm::cuda_stream_view stream_;   // CUDA stream obtained from handle_

  // Device pointers for each data array.
  raft::device_vector<uint32_t, int64_t> short_data_;          // RaBitQ code and factors.
  raft::device_vector<uint8_t, int64_t> long_code_;            // ExRaBitQ code.
  raft::device_vector<ExFactor, int64_t> ex_factor_;           // ExRaBitQ factor.
  raft::device_vector<PID, int64_t> ids_;                      // PID of vectors.
  raft::device_vector<GPUClusterMeta, int64_t> cluster_meta_;  // Device-side array of clusters.

  // batch-data
  bool batch_flag;
  //    uint32_t* d_short_data_batch;   // rabitq codes
  raft::device_vector<float, int64_t> short_factors_batch_;  // N * 3 float rabitq factors
  // long_code_ is the same
  // exfactor use the same place as before

  // host-side copies
  //    float* d_centroids;      // Device centroids (if needed for search, now stored in
  //    initializer).

  raft::host_vector<uint32_t, int64_t>
    short_data_host_;  // TODO: CPU side, we need on factors from short_data_host_, so no need to
                       // store all these codes
  raft::host_vector<uint8_t, int64_t> long_code_host_;            // ExRaBitQ code.
  raft::host_vector<ExFactor, int64_t> ex_factor_host_;           // ExRaBitQ factor.
  raft::host_vector<PID, int64_t> ids_host_;                      // PID of vectors.
  raft::host_vector<GPUClusterMeta, int64_t> cluster_meta_host_;  // Host-side copy of clusters

  // Index meta-data.
  size_t num_vectors;     // Total number of vectors.
  size_t num_dimensions;  // Dimensionality of the input vectors.
  size_t num_padded_dim;  // padded dimensions
  size_t num_centroids;   // Centroids && Clusters
  size_t max_cluster_length;
  size_t ex_bits;  // Extra bits parameter for quantization.

  std::unique_ptr<InitializerGPU>
    initializer;  // Initializer, indicates which initializer to use (currently only FlatIVF)
  std::unique_ptr<DataQuantizerGPU> DQ;  // Data quantizer.
  std::unique_ptr<RotatorGPU> Rota;      // Data Rotator.
};

}  // namespace cuvs::neighbors::ivf_rabitq::detail

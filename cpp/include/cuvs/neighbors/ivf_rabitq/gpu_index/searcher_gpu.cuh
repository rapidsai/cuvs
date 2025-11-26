/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 4/14/25.
//

#pragma once

#include <cuvs/neighbors/ivf_rabitq/gpu_index/ivf_gpu.cuh>
#include <cuvs/neighbors/ivf_rabitq/gpu_index/pool_gpu.cuh>
#include <cuvs/neighbors/ivf_rabitq/gpu_index/quantizer_gpu.cuh>

#include <raft/core/resources.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cuvs::neighbors::ivf_rabitq::detail {

struct Candidate3 {  // unchanged
  float dist;
  float ip;
  uint32_t pid;
  int idx;
};
// 4: For multiple cluster search
struct Candidate4 {  // unchanged
  float dist;
  float ip;
  uint32_t pid;
  int idx;  // idx inside clusters
  int nprobe;
};

struct SumNorm {
  float sum;
  float norm;
};

class SearcherGPU {
 public:
  size_t D;                        // number of dimension
  const float* query   = nullptr;  // rotated query
  int16_t* quant_query = nullptr;  // quantized query (to 2 bytes)
  float* unit_q        = nullptr;
  int shift;
  float delta = 0;  // quantized related
  float sumq;
  float one_over_sqrtD  = 0;  // 1/sqrt(D)
  int FAC_RESCALE       = 0;
  float* d_filter_distk = nullptr;
  float h_filter_distk  = INFINITY;
  int direct_num        = 0;
  int sort_num          = 0;
  bool rabitq_quantize_flag;
  std::string mode;
  //    float *d_filter_distk = 0x7F800000; // infinity

  // Additional preparation to avoid repeatedly malloc space among different probes
  // will be more if using multiple clusters method
  float* d_unit_q_gpu = nullptr;
  float* d_ip_results = nullptr;  // will be more if using multiple clusters method
  float* d_est_dis    = nullptr;
  float* d_top_ip     = nullptr;
  PID* d_top_pids     = nullptr;
  int* d_top_idx      = nullptr;
  float* d_ip2        = nullptr;
  Candidate3* d_buf   = nullptr;

  float* h_ip_results = nullptr;
  float* h_est_dis    = nullptr;

  // batch
  float* d_topk_threshold = nullptr;  // threshold to filter distance for each query
  float* d_centroid_distances =
    nullptr;  // stores distances between centroids and queries (l2 square)
  float* d_c_norms                 = nullptr;  // centroid norms
  float* d_q_norms                 = nullptr;  // query norms
  Candidate* d_centroid_candidates = nullptr;  // for candidate computation
  std::vector<DeviceResultPool*>
    topk_results;  // TODO: Change the vectors into a whole array on the GPU

  // quantization for queries
  float best_rescaling_factor = 0.0f;

  // additional for multiple cluster search！！！！！！

  // constexpr int  K      = 20000;  // total centroids in memory
  // constexpr int  ITER   = 20;     // timing iterations
  // constexpr int  BLOCK  = 256;    // baseline / warp256 block‑dim
  // constexpr float EPS_R = 1e-6f;  // verification tol

  //--------------------------------------------------------------

  SumNorm* d_sum_norm            = nullptr;
  Candidate4* d_candidate_buffer = nullptr;
  //    float* c_query = nullptr;
  //    int* d_starts = nullptr;

  explicit SearcherGPU(raft::resources const& handle,
                       const float* q,
                       size_t d,
                       size_t ex_bits,
                       std::string mode          = "",
                       bool rabitq_quantize_flag = true);

  ~SearcherGPU() { destroy(); }

  /**
   * @brief malloc temp search space in GPU memory
   *
   * @param cur_ivf related ivf for search
   * @param num_queries number of queries
   * @param k topk
   * @param max_nprobes maximum nprobes for search
   * @param s cuda stream
   * @return no returns
   */
  void AllocateSearcherSpace(const IVFGPU& cur_ivf,
                             size_t num_queries,
                             size_t k,
                             size_t max_nprobes,
                             size_t max_cluster_length);

  /**
   * @brief Search one of the IVF Cluster
   *
   * @param centroid_id the cid that indicates the cluster
   * @param sqr_y distance of (centroid <-> quantized query)
   * @param KNNs device pointer that points to the KNN result
   */
  void SearchCluster(const IVFGPU& cur_ivf,
                     const typename IVFGPU::GPUClusterMeta& cur_cluster,
                     float sqr_y,
                     DeviceResultPool* KNNs,
                     float* centroid_data);

  void SearchClustershowingTime(const IVFGPU& cur_ivf,
                                const IVFGPU::GPUClusterMeta& cur_cluster,
                                float sqr_y,
                                DeviceResultPool* KNNs,
                                float* centroid_data);

  void SearchClusterWithFilter(const IVFGPU& cur_ivf,
                               const IVFGPU::GPUClusterMeta& cur_cluster,
                               float sqr_y,
                               DeviceResultPool* KNNs,
                               float* centroid_data);

  void SearchClusterWithFilterMemOpt(const IVFGPU& cur_ivf,
                                     const IVFGPU::GPUClusterMeta& cur_cluster,
                                     float sqr_y,
                                     DeviceResultPool* KNNs,
                                     float* centroid_data);

  void SearchClusterWithFilterMemOptOffload(const IVFGPU& cur_ivf,
                                            const IVFGPU::GPUClusterMeta& cur_cluster,
                                            float sqr_y,
                                            BoundedKNN* KNNs,
                                            float* centroid_data);

  void SearchClusterWithFilterMemOptV2(const IVFGPU& cur_ivf,
                                       const IVFGPU::GPUClusterMeta& cur_cluster,
                                       float sqr_y,
                                       DeviceResultPool* KNNs,
                                       float* centroid_data);

  void SearchMultipleClusters(const IVFGPU& cur_ivf,
                              IVFGPU::GPUClusterMeta* d_cluster_meta,
                              Candidate* d_centroid_candidates,
                              DeviceResultPool* KNNs,
                              float* d_centroid,
                              const float* h_query,
                              size_t nprobe);

  void SearchClusterWithFilterMemOptOneforMulti(const IVFGPU& cur_ivf,
                                                const IVFGPU::GPUClusterMeta& cur_cluster,
                                                float sqr_y,
                                                DeviceResultPool* KNNs,
                                                float* centroid_data);

  static SearcherGPU* CreateNewSearcherforStream(size_t d,
                                                 size_t ex_bits,
                                                 size_t num_clusters,
                                                 size_t num_vectors,
                                                 size_t k,
                                                 rmm::cuda_stream_view stream);

  /**
   * @brief Function to Search <cluster_id, query_id> pairs
   * @param cur_ivf
   * @param d_cluster_meta
   * @param d_sorted_pairs
   * @param num_queries
   * @param d_query
   * @param d_G_k1xSumq
   * @param d_G_kbxSumq
   * @param all_topk_results
   * @param h_query
   * @param nprobe
   * @param d_centroid
   * @param topk
   * @param d_topk_dists
   * @param d_topk_pids
   * @param stream
   */
  void SearchClusterQueryPairs(const IVFGPU& cur_ivf,
                               IVFGPU::GPUClusterMeta* d_cluster_meta,
                               ClusterQueryPair* d_sorted_pairs,
                               size_t num_queries,
                               const float* d_query,
                               const float* d_G_k1xSumq,
                               const float* d_G_kbxSumq,
                               size_t nprobe,
                               size_t topk,
                               float* d_topk_dists,
                               PID* d_topk_pids,
                               float* d_final_dists,
                               PID* d_final_pids);

  void SearchClusterQueryPairsPreComputeThreshold(const IVFGPU& cur_ivf,
                                                  IVFGPU::GPUClusterMeta* d_cluster_meta,
                                                  ClusterQueryPair* d_nearest_sorted_pairs,
                                                  ClusterQueryPair* d_rest_sorted_pairs,
                                                  size_t num_queries,
                                                  const float* d_query,
                                                  const float* d_G_k1xSumq,
                                                  const float* d_G_kbxSumq,
                                                  size_t nprobe,
                                                  size_t topk,
                                                  float* d_topk_dists,
                                                  PID* d_topk_pids,
                                                  float* d_final_dists,
                                                  PID* d_final_pids);

  void SearchClusterQueryPairsSharedMemOpt(const IVFGPU& cur_ivf,
                                           IVFGPU::GPUClusterMeta* d_cluster_meta,
                                           ClusterQueryPair* d_sorted_pairs,
                                           size_t num_queries,
                                           const float* d_query,
                                           const float* d_G_k1xSumq,
                                           const float* d_G_kbxSumq,
                                           size_t nprobe,
                                           size_t topk,
                                           float* d_topk_dists,
                                           PID* d_topk_pids,
                                           float* d_final_dists,
                                           PID* d_final_pids);

  void SearchClusterQueryPairsQuantizeQuery(const IVFGPU& cur_ivf,
                                            IVFGPU::GPUClusterMeta* d_cluster_meta,
                                            ClusterQueryPair* d_sorted_pairs,
                                            size_t num_queries,
                                            const float* d_query,
                                            const float* d_G_k1xSumq,
                                            const float* d_G_kbxSumq,
                                            size_t nprobe,
                                            size_t topk,
                                            float* d_topk_dists,
                                            PID* d_topk_pids,
                                            float* d_final_dists,
                                            PID* d_final_pids,
                                            bool use_4bit = false);

 private:
  raft::resources const& handle_;  // reusable resource handle
  rmm::cuda_stream_view stream_;   // CUDA stream obtained from handle_
  //    float* uint_q = nullptr;
  void destroy() noexcept;

  /* ---------- tiny helpers ---------- */
  inline void safeCudaFreeAsync(void* p) noexcept
  {
    if (p) RAFT_CUDA_TRY_NO_THROW(cudaFreeAsync(p, stream_));  // ignore error in a noexcept dtor
  }
  template <typename T>
  static inline void safeHostFree(T*& p) noexcept
  {
    if (p) {
      /* replace with delete[] if you used new[] */
      std::free(p);
      p = nullptr;
    }
  }

  void SearchClusterWithFilterMemOptBoundedKNN(const IVFGPU& cur_ivf,
                                               const IVFGPU::GPUClusterMeta& cur_cluster,
                                               float sqr_y,
                                               BoundedKNN* KNNs,
                                               float* centroid_data);
};

// Launch this kernel with at least num_vector_cluster * 32 threads.
// Each warp of 32 threads handles one vector.
__global__ void compute_ip_kernel(const int16_t* quant_query_gpu,
                                  const uint32_t* rabitq_codes_and_factors,
                                  size_t num_dimensions,
                                  size_t num_short_factors,
                                  size_t num_vector_cluster,
                                  float* ip_results,
                                  float* est_dis,
                                  float delta,
                                  float sumq,
                                  float qnorm,
                                  float one_over_sqrtD);

}  // namespace cuvs::neighbors::ivf_rabitq::detail

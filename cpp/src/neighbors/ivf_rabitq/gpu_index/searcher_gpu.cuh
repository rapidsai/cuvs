/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 4/14/25.
//

#pragma once

#include "ivf_gpu.cuh"
#include "pool_gpu.cuh"
#include "quantizer_gpu.cuh"

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
  explicit SearcherGPU(raft::resources const& handle,
                       const float* q,
                       size_t d,
                       size_t ex_bits,
                       std::string mode                                             = "",
                       DataQuantizerGPU::FastQuantizeFactors* fast_quantize_factors = nullptr,
                       bool rabitq_quantize_flag                                    = true);

  SearcherGPU(SearcherGPU&& other) = default;

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

  // Getter methods
  int get_direct_num() { return direct_num_; }
  int get_sort_num() { return sort_num_; }
  std::string const& get_mode() { return mode_; }
  raft::device_vector<float, int64_t>& get_unit_q_gpu() { return unit_q_gpu_; }
  raft::device_vector<float, int64_t>& get_ip_results() { return ip_results_; }
  raft::device_vector<float, int64_t>& get_est_dis() { return est_dis_; }
  raft::device_vector<float, int64_t>& get_top_ip() { return top_ip_; }
  raft::device_vector<PID, int64_t>& get_top_pids() { return top_pids_; }
  raft::device_vector<int, int64_t>& get_top_idx() { return top_idx_; }
  raft::device_vector<float, int64_t>& get_ip2() { return ip2_; }
  raft::device_vector<Candidate3, int64_t>& get_buf() { return buf_; }
  float* get_centroid_distances() { return centroid_distances_.data_handle(); }
  float* get_c_norms() { return c_norms_.data_handle(); }
  float* get_q_norms() { return q_norms_.data_handle(); }
  raft::device_vector<SumNorm, int64_t>& get_sum_norm() { return sum_norm_; }
  raft::device_vector<Candidate4, int64_t>& get_candidate_buffer() { return candidate_buffer_; }

  // Setter methods
  void set_query(const float* query) { query_ = query; }
  void set_quant_query(int16_t* quant_query)
  {
    quant_query_ = std::unique_ptr<int16_t, decltype(std::free)*>(quant_query, std::free);
  }  // quant_query must be allocated with `malloc` or similar, as opposed to `new`
  void set_unit_q(float* unit_q)
  {
    unit_q_ = std::unique_ptr<float, decltype(std::free)*>(unit_q, std::free);
  }  // unit_q must be allocated with `malloc` or similar, as opposed to `new`
  void set_filter_distk(const float filter_distk) { filter_distk_ = filter_distk; }

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
                     DeviceResultPool& KNNs,
                     float* centroid_data);

  void SearchClustershowingTime(const IVFGPU& cur_ivf,
                                const IVFGPU::GPUClusterMeta& cur_cluster,
                                float sqr_y,
                                DeviceResultPool& KNNs,
                                float* centroid_data);

  void SearchClusterWithFilter(const IVFGPU& cur_ivf,
                               const IVFGPU::GPUClusterMeta& cur_cluster,
                               float sqr_y,
                               DeviceResultPool& KNNs,
                               float* centroid_data);

  void SearchClusterWithFilterMemOpt(const IVFGPU& cur_ivf,
                                     const IVFGPU::GPUClusterMeta& cur_cluster,
                                     float sqr_y,
                                     DeviceResultPool& KNNs,
                                     float* centroid_data);

  void SearchClusterWithFilterMemOptOffload(const IVFGPU& cur_ivf,
                                            const IVFGPU::GPUClusterMeta& cur_cluster,
                                            float sqr_y,
                                            BoundedKNN* KNNs,
                                            float* centroid_data);

  void SearchClusterWithFilterMemOptV2(const IVFGPU& cur_ivf,
                                       const IVFGPU::GPUClusterMeta& cur_cluster,
                                       float sqr_y,
                                       DeviceResultPool& KNNs,
                                       float* centroid_data);

  void SearchMultipleClusters(const IVFGPU& cur_ivf,
                              IVFGPU::GPUClusterMeta* d_cluster_meta,
                              Candidate* d_centroid_candidates,
                              DeviceResultPool& KNNs,
                              float* d_centroid,
                              const float* h_query,
                              size_t nprobe);

  void SearchClusterWithFilterMemOptOneforMulti(const IVFGPU& cur_ivf,
                                                const IVFGPU::GPUClusterMeta& cur_cluster,
                                                float sqr_y,
                                                DeviceResultPool& KNNs,
                                                float* centroid_data);

  static SearcherGPU* CreateNewSearcherforStream(raft::resources const& handle,
                                                 size_t d,
                                                 size_t ex_bits,
                                                 size_t num_clusters,
                                                 size_t num_vectors,
                                                 size_t k);

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
  size_t D;                        // number of dimension
  const float* query_ = nullptr;   // rotated query (non-owning)
  std::unique_ptr<int16_t, decltype(std::free)*> quant_query_ = {
    nullptr, std::free};  // quantized query (to 2 bytes)
  std::unique_ptr<float, decltype(std::free)*> unit_q_ = {nullptr, std::free};
  int shift_;
  float delta_ = 0;  // quantized related
  float sumq_;
  float one_over_sqrtD_ = 0;  // 1/sqrt(D)
  int FAC_RESCALE_      = 0;
  float filter_distk_   = INFINITY;
  int direct_num_       = 0;
  int sort_num_         = 0;
  bool rabitq_quantize_flag_;
  std::string mode_;

  // Additional preparation to avoid repeatedly malloc space among different probes
  // will be more if using multiple clusters method
  raft::device_vector<float, int64_t> unit_q_gpu_ =
    raft::make_device_vector<float, int64_t>(handle_, 0);
  raft::device_vector<float, int64_t> ip_results_ = raft::make_device_vector<float, int64_t>(
    handle_, 0);  // will be more if using multiple clusters method
  raft::device_vector<float, int64_t> est_dis_ =
    raft::make_device_vector<float, int64_t>(handle_, 0);
  raft::device_vector<float, int64_t> top_ip_ =
    raft::make_device_vector<float, int64_t>(handle_, 0);
  raft::device_vector<PID, int64_t> top_pids_ = raft::make_device_vector<PID, int64_t>(handle_, 0);
  raft::device_vector<int, int64_t> top_idx_  = raft::make_device_vector<int, int64_t>(handle_, 0);
  raft::device_vector<float, int64_t> ip2_ = raft::make_device_vector<float, int64_t>(handle_, 0);
  raft::device_vector<Candidate3, int64_t> buf_ =
    raft::make_device_vector<Candidate3, int64_t>(handle_, 0);

  raft::host_vector<float, int64_t> ip_results_host_ = raft::make_host_vector<float, int64_t>(0);
  raft::host_vector<float, int64_t> est_dis_host_    = raft::make_host_vector<float, int64_t>(0);

  // batch
  raft::device_vector<float, int64_t> centroid_distances_ =
    raft::make_device_vector<float, int64_t>(
      handle_, 0);  // stores distances between centroids and queries (l2 square)
  raft::device_vector<float, int64_t> c_norms_ =
    raft::make_device_vector<float, int64_t>(handle_, 0);  // centroid norms
  raft::device_vector<float, int64_t> q_norms_ =
    raft::make_device_vector<float, int64_t>(handle_, 0);  // query norms
  std::vector<DeviceResultPool>
    topk_results;  // TODO: Change the vectors into a whole array on the GPU

  // quantization for queries
  float best_rescaling_factor = 0.0f;

  //--------------------------------------------------------------

  raft::device_vector<SumNorm, int64_t> sum_norm_ =
    raft::make_device_vector<SumNorm, int64_t>(handle_, 0);
  raft::device_vector<Candidate4, int64_t> candidate_buffer_ =
    raft::make_device_vector<Candidate4, int64_t>(handle_, 0);

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

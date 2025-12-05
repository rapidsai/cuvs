/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 4/14/25.
//

#pragma once

#include "ivf_gpu.cuh"
#include "quantizer_gpu.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cuvs::neighbors::ivf_rabitq::detail {

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
  std::string const& get_mode() { return mode_; }
  float* get_centroid_distances() { return centroid_distances_.data_handle(); }
  float* get_c_norms() { return c_norms_.data_handle(); }
  float* get_q_norms() { return q_norms_.data_handle(); }

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
  rmm::cuda_stream_view stream_ =
    raft::resource::get_cuda_stream(handle_);  // CUDA stream obtained from handle_
  size_t D;                                    // number of dimension
  const float* query_ = nullptr;               // rotated query (non-owning)
  std::unique_ptr<int16_t, decltype(std::free)*> quant_query_ = {
    nullptr, std::free};  // quantized query (to 2 bytes)
  std::unique_ptr<float, decltype(std::free)*> unit_q_ = {nullptr, std::free};
  bool rabitq_quantize_flag_;
  std::string mode_;

  // batch
  raft::device_vector<float, int64_t> centroid_distances_ =
    raft::make_device_vector<float, int64_t>(
      handle_, 0);  // stores distances between centroids and queries (l2 square)
  raft::device_vector<float, int64_t> c_norms_ =
    raft::make_device_vector<float, int64_t>(handle_, 0);  // centroid norms
  raft::device_vector<float, int64_t> q_norms_ =
    raft::make_device_vector<float, int64_t>(handle_, 0);  // query norms

  // quantization for queries
  float best_rescaling_factor = 0.0f;
};

}  // namespace cuvs::neighbors::ivf_rabitq::detail

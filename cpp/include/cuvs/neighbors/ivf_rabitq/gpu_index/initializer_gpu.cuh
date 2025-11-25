/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 3/3/25.
//

#pragma once

#include <cuvs/neighbors/ivf_rabitq/defines.hpp>
#include <cuvs/neighbors/ivf_rabitq/utils/utils_cuda.cuh>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace cuvs::neighbors::ivf_rabitq::detail {

class InitializerGPU {
 public:
  /**
   * @brief Constructor.
   *
   * @param d dimension of vectors
   * @param k number of centroids
   */
  explicit InitializerGPU(raft::resources const& handle, size_t d, size_t k)
    : D(d), K(k), stream_(raft::resource::get_cuda_stream(handle))
  {
  }

  virtual ~InitializerGPU() = default;

  /**
   * @brief Get a getCentroidbyId's vector data. Use ID as index.
   *
   * @param id Index.
   * @return
   */
  virtual __host__ __device__ float* GetCentroid(PID id) const = 0;

  /**
   * @brief Copies centroids from a host pointer to the device memory assuming that 'cent' points to
   * a host array of size K * D.
   *
   * @param cent pointer to centroids
   */
  virtual void AddVectors(raft::resources const& handle, const float* cent) = 0;

  virtual void AddVectorsD2D(raft::resources const& handle, const float* cent) = 0;

  /**
   * @brief Computes the distances from the query vector to each getCentroidbyId.
   *
   * @param query
   * @param nprobe How many of the closest centroids should be returned.
   * @param candidates Array of candidates.
   * @param num_candidates
   */
  virtual void ComputeCentroidsDistances(raft::resources const& handle,
                                         const float* query,
                                         size_t nprobe,
                                         Candidate* candidates,
                                         size_t num_candidates) const = 0;

  /**
   * @brief LoadCentroids centroids' information from files.
   *
   * @param input
   * @param filename
   */
  virtual void LoadCentroids(raft::resources const& handle,
                             std::ifstream& input,
                             const char* filename) = 0;

  /**
   * @brief SaveCentroids centroids' information from files.
   *
   * @param save
   * @param filename
   */
  virtual void SaveCentroids(raft::resources const& handle,
                             std::ofstream& output,
                             const char* filename) const = 0;

 protected:
  size_t D;              // Dimension
  size_t K;              // Num of Centroids
  cudaStream_t stream_;  // CUDA stream
};

class FlatInitializerGPU : public InitializerGPU {
 public:
  explicit FlatInitializerGPU(raft::resources const& handle, size_t d, size_t k);

  ~FlatInitializerGPU() override;

  [[nodiscard]] __host__ __device__ float* GetCentroid(PID id) const override;

  void AddVectors(raft::resources const& handle, const float* cent) override;

  void ComputeCentroidsDistances(raft::resources const& handle,
                                 const float* query,
                                 size_t nprobe,
                                 Candidate* candidates,
                                 size_t num_candidates) const override;

  void LoadCentroids(raft::resources const& handle,
                     std::ifstream& input,
                     const char* filename) override;

  void SaveCentroids(raft::resources const& handle,
                     std::ofstream& output,
                     const char* filename) const override;

  __host__ __device__ float* GetCentroidTranspose(PID id) const;

  void AddVectorsTranspose(raft::resources const& handle, const float* cent);

  void AddVectorsD2D(raft::resources const& handle, const float* cent);

  void ComputeCentroidsDistancesTranspose(raft::resources const& handle,
                                          const float* query,
                                          size_t nprobe,
                                          Candidate* candidates,
                                          size_t num_candidates) const;

  void LoadCentroidsTranspose(raft::resources const& handle,
                              std::ifstream& input,
                              const char* filename);

  void SaveCentroidsTranspose(raft::resources const& handle,
                              std::ofstream& output,
                              const char* filename) const;

 private:
  // D, K are inherited from parent

  float* Centroids;  // Stored in GPU device memory. Points to the parent centroids' array (do we
                     // really need a pointer?)
  // For simplicity, we use a single distance function.

  float (*dist_func)(const float* __restrict__, const float* __restrict__, size_t);

  [[nodiscard]] size_t data_bytes() const { return sizeof(float) * K * D; }
};

}  // namespace cuvs::neighbors::ivf_rabitq::detail

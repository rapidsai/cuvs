/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 4/1/25.
//

#include <cuvs/neighbors/ivf_rabitq/utils/tools_gpu.cuh>

#include <raft/util/cuda_rt_essentials.hpp>

#include <iostream>

namespace cuvs::neighbors::ivf_rabitq::detail {

/**
 * @brief Creates a specified number of CUDA streams.
 * * @param num_streams The number of CUDA streams to create.
 * @return A vector containing the handles to the created CUDA streams.
 */
std::vector<cudaStream_t> create_cuda_streams(size_t num_streams)
{
  std::cout << "Attempting to create " << num_streams << " CUDA streams..." << std::endl;
  std::vector<cudaStream_t> streams(num_streams);

  for (size_t i = 0; i < num_streams; ++i) {
    // cudaStreamCreate allocates a new stream and returns its handle.
    // The handle is stored in the vector at the current index.
    RAFT_CUDA_TRY(cudaStreamCreate(&streams[i]));

    // Always check for errors after a CUDA API call.
    //        CUDA_CHECK(err);

    //        std::cout << "  Successfully created CUDA stream " << i << "." << std::endl;
  }
  std::cout << "All CUDA streams created successfully." << std::endl;
  return streams;
}

/**
 * @brief Destroys a vector of CUDA streams.
 * * @param streams A reference to a vector of cudaStream_t handles.
 * The vector will be cleared after cleanup.
 */
void delete_cuda_streams(std::vector<cudaStream_t>& streams)
{
  std::cout << "\nDestroying " << streams.size() << " CUDA streams..." << std::endl;

  // Iterate over the vector of stream handles
  for (cudaStream_t stream : streams) {
    // cudaStreamDestroy frees the resources associated with a stream.
    RAFT_CUDA_TRY(cudaStreamDestroy(stream));

    // It's good practice to check for errors even during cleanup.
    //        CUDA_CHECK(err);
  }
  std::cout << "Successfully destroyed all CUDA streams." << std::endl;

  // Clear the vector to remove the now-invalidated handles.
  streams.clear();
  //    std::cout << "Streams vector has been cleared." << std::endl;
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail

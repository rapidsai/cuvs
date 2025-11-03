/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

//
// Created by Stardust on 4/1/25.
//

#include <cuvs/neighbors/ivf_rabitq/utils/tools_gpu.cuh>
#include <iostream>

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
    cudaStreamCreate(&streams[i]);

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
    cudaStreamDestroy(stream);

    // It's good practice to check for errors even during cleanup.
    //        CUDA_CHECK(err);
  }
  std::cout << "Successfully destroyed all CUDA streams." << std::endl;

  // Clear the vector to remove the now-invalidated handles.
  streams.clear();
  //    std::cout << "Streams vector has been cleared." << std::endl;
}

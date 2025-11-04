/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "topk_core.cuh"

namespace cuvs::neighbors::cagra::detail {

//
size_t _cuann_find_topk_bufferSize(uint32_t topK,
                                   uint32_t sizeBatch,
                                   uint32_t numElements,
                                   cudaDataType_t sampleDtype)
{
  constexpr int numThreads  = NUM_THREADS;
  constexpr int stateBitLen = STATE_BIT_LENGTH;
  assert(stateBitLen == 0 || stateBitLen == 8);

  size_t workspaceSize = 1;
  // state
  if (stateBitLen == 8) {
    workspaceSize = _cuann_aligned(
      sizeof(uint8_t) * get_state_size<stateBitLen, numThreads>(numElements) * sizeBatch);
  }

  return workspaceSize;
}

template <class ValT>
void _cuann_find_topk(uint32_t topK,
                      uint32_t sizeBatch,
                      uint32_t numElements,
                      const float* inputKeys,  // [sizeBatch, ldIK,]
                      uint32_t ldIK,           // (*) ldIK >= numElements
                      const ValT* inputVals,   // [sizeBatch, ldIV,]
                      uint32_t ldIV,           // (*) ldIV >= numElements
                      float* outputKeys,       // [sizeBatch, ldOK,]
                      uint32_t ldOK,           // (*) ldOK >= topK
                      ValT* outputVals,        // [sizeBatch, ldOV,]
                      uint32_t ldOV,           // (*) ldOV >= topK
                      void* workspace,
                      bool sort,
                      uint32_t* hints,
                      cudaStream_t stream)
{
  assert(ldIK >= numElements);
  assert(ldIV >= numElements);
  assert(ldOK >= topK);
  assert(ldOV >= topK);

  constexpr int numThreads  = NUM_THREADS;
  constexpr int stateBitLen = STATE_BIT_LENGTH;
  assert(stateBitLen == 0 || stateBitLen == 8);

  uint8_t* state = NULL;
  if (stateBitLen == 8) { state = (uint8_t*)workspace; }

  dim3 threads(numThreads, 1, 1);
  dim3 blocks(sizeBatch, 1, 1);

  void (*cta_kernel)(uint32_t,
                     uint32_t,
                     uint32_t,
                     uint32_t,
                     uint32_t,
                     const uint32_t*,
                     uint32_t,
                     const ValT*,
                     uint32_t,
                     uint32_t*,
                     uint32_t,
                     ValT*,
                     uint32_t,
                     uint8_t*,
                     uint32_t*,
                     bool) = nullptr;

  int _vecLen                    = _get_vecLen(ldIK, 2);
  constexpr int maxTopkPerThread = 4;

  if (_vecLen == 2) {
    cta_kernel = kern_topk_cta_11<stateBitLen, 2, maxTopkPerThread, ValT>;
  } else if (_vecLen == 1) {
    cta_kernel = kern_topk_cta_11<stateBitLen, 1, maxTopkPerThread, ValT>;
  }

  int max_topk{};
  int num_sort_threads{};
  if (topK <= 32) {
    max_topk         = 32;
    num_sort_threads = 32;
  } else if (topK <= 64) {
    max_topk         = 64;
    num_sort_threads = 32;
  } else if (topK <= 96) {
    max_topk         = 96;
    num_sort_threads = 32;
  } else if (topK <= 128) {
    max_topk         = 128;
    num_sort_threads = 32;
  } else if (topK <= 192) {
    max_topk         = 192;
    num_sort_threads = 64;
  } else if (topK <= 256) {
    max_topk         = 256;
    num_sort_threads = 64;
  } else if (topK <= 384) {
    max_topk         = 384;
    num_sort_threads = 128;
  } else if (topK <= 512) {
    max_topk         = 512;
    num_sort_threads = 128;
  } else if (topK <= 768) {
    max_topk         = 768;
    num_sort_threads = 256;
  } else if (topK <= 1024) {
    max_topk         = 1024;
    num_sort_threads = 256;
  } else {
    RAFT_FAIL("topK must be lower than or equal to 1024");
  }

  assert(max_topk % num_sort_threads == 0);
  assert(max_topk / num_sort_threads <= maxTopkPerThread);

  const size_t smem_len = 2 * max_topk + 2048 + 8;
  assert(max_topk * (1 + utils::size_of<ValT>() / utils::size_of<uint32_t>()) <= smem_len);
  const size_t smem_size = smem_len * sizeof(uint32_t);
  cta_kernel<<<blocks, threads, smem_size, stream>>>(max_topk,
                                                     num_sort_threads,
                                                     topK,
                                                     sizeBatch,
                                                     numElements,
                                                     (const uint32_t*)inputKeys,
                                                     ldIK,
                                                     inputVals,
                                                     ldIV,
                                                     (uint32_t*)outputKeys,
                                                     ldOK,
                                                     outputVals,
                                                     ldOV,
                                                     state,
                                                     hints,
                                                     sort);

  return;
}

template void _cuann_find_topk<uint32_t>(uint32_t topK,
                                         uint32_t sizeBatch,
                                         uint32_t numElements,
                                         const float* inputKeys,     // [sizeBatch, ldIK,]
                                         uint32_t ldIK,              // (*) ldIK >= numElements
                                         const uint32_t* inputVals,  // [sizeBatch, ldIV,]
                                         uint32_t ldIV,              // (*) ldIV >= numElements
                                         float* outputKeys,          // [sizeBatch, ldOK,]
                                         uint32_t ldOK,              // (*) ldOK >= topK
                                         uint32_t* outputVals,       // [sizeBatch, ldOV,]
                                         uint32_t ldOV,              // (*) ldOV >= topK
                                         void* workspace,
                                         bool sort,
                                         uint32_t* hint,
                                         cudaStream_t stream);

}  // namespace cuvs::neighbors::cagra::detail

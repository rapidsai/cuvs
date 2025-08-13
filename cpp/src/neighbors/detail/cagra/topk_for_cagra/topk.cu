/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

  // V:vecLen, K:maxTopk, T:numSortThreads
#define SET_KERNEL_VKT(V, K, T, ValT)                          \
  do {                                                         \
    assert(numThreads >= T);                                   \
    assert((K % T) == 0);                                      \
    assert((K / T) <= 4);                                      \
    cta_kernel = kern_topk_cta_11<stateBitLen, V, K, T, ValT>; \
  } while (0)

  // V: vecLen
#define SET_KERNEL_V(V, ValT)                                \
  do {                                                       \
    if (topK <= 32) {                                        \
      SET_KERNEL_VKT(V, 32, 32, ValT);                       \
    } else if (topK <= 64) {                                 \
      SET_KERNEL_VKT(V, 64, 32, ValT);                       \
    } else if (topK <= 96) {                                 \
      SET_KERNEL_VKT(V, 96, 32, ValT);                       \
    } else if (topK <= 128) {                                \
      SET_KERNEL_VKT(V, 128, 32, ValT);                      \
    } else if (topK <= 192) {                                \
      SET_KERNEL_VKT(V, 192, 64, ValT);                      \
    } else if (topK <= 256) {                                \
      SET_KERNEL_VKT(V, 256, 64, ValT);                      \
    } else if (topK <= 384) {                                \
      SET_KERNEL_VKT(V, 384, 128, ValT);                     \
    } else if (topK <= 512) {                                \
      SET_KERNEL_VKT(V, 512, 128, ValT);                     \
    } else if (topK <= 768) {                                \
      SET_KERNEL_VKT(V, 768, 256, ValT);                     \
    } else if (topK <= 1024) {                               \
      SET_KERNEL_VKT(V, 1024, 256, ValT);                    \
    } \
        /* else if (topK <= 1536) { SET_KERNEL_VKT(V, 1536, 512); } */ \
        /* else if (topK <= 2048) { SET_KERNEL_VKT(V, 2048, 512); } */ \
        /* else if (topK <= 3072) { SET_KERNEL_VKT(V, 3072, 1024); } */ \
        /* else if (topK <= 4096) { SET_KERNEL_VKT(V, 4096, 1024); } */ \
        else {                                                      \
      RAFT_FAIL("topk must be lower than or equal to 1024"); \
    }                                                        \
  } while (0)

  int _vecLen = _get_vecLen(ldIK, 2);
  if (_vecLen == 2) {
    SET_KERNEL_V(2, ValT);
  } else if (_vecLen == 1) {
    SET_KERNEL_V(1, ValT);
  }

  cta_kernel<<<blocks, threads, 0, stream>>>(topK,
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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "topk_core.cuh"

namespace cuvs::neighbors::cagra::detail {

//
auto _cuann_find_topk_bufferSize(  // NOLINT(readability-identifier-naming)
  uint32_t topK,
  uint32_t sizeBatch,
  uint32_t numElements,
  cudaDataType_t sampleDtype) -> size_t
{
  constexpr int numThreads   = kNumThreads;      // NOLINT(readability-identifier-naming)
  constexpr int kStateBitLen = kStateBitLength;  // NOLINT(readability-identifier-naming)
  assert(kStateBitLen == 0 || kStateBitLen == 8);

  size_t workspace_size = 1;
  // state
  if (kStateBitLen == 8) {
    workspace_size = _cuann_aligned(
      sizeof(uint8_t) * get_state_size<kStateBitLen, numThreads>(numElements) * sizeBatch);
  }

  return workspace_size;
}

template <class ValT>
void cuann_find_topk(uint32_t topK,  // NOLINT(readability-identifier-naming)
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

  constexpr int numThreads   = kNumThreads;  // NOLINT(readability-identifier-naming)
  constexpr int kStateBitLen = kStateBitLength;
  assert(kStateBitLen == 0 || kStateBitLen == 8);

  uint8_t* state = nullptr;
  if (kStateBitLen == 8) { state = (uint8_t*)workspace; }  // NOLINT(google-readability-casting)

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
#define SET_KERNEL_VKT(V, K, T, ValT)                           \
  do {                                                          \
    assert(numThreads >= T);                                    \
    assert((K % T) == 0);                                       \
    assert((K / T) <= 4);                                       \
    cta_kernel = kern_topk_cta_11<kStateBitLen, V, K, T, ValT>; \
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

  int _vecLen = get_vec_len(ldIK, 2);  // NOLINT(readability-identifier-naming)
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

template void cuann_find_topk<uint32_t>(uint32_t topK,
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

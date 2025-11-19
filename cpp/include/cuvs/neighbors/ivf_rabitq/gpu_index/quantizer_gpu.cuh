/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 3/10/25.
//

#ifndef EXRABITQ_QUANTIZER_GPU_CUH
#define EXRABITQ_QUANTIZER_GPU_CUH

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
// #include <math_functions.h>
#include <cuvs/neighbors/ivf_rabitq/defines.hpp>
// Assume RotatorGPU is defined in your GPU version of Rotator.
#include <cuvs/neighbors/ivf_rabitq/gpu_index/rotator_gpu.cuh>
// Include space_cuda.cuh if needed for CUDA‐related memory routines.
#include <cuvs/neighbors/ivf_rabitq/utils/space_cuda.cuh>
#include <cuvs/neighbors/ivf_rabitq/utils/tools_gpu.cuh>

#include <raft/core/resources.hpp>

// Define PID and ExFactor as in your CPU version.
typedef uint32_t PID;
// typedef float ExFactor; // example

/*!
 * @brief GPU version of DataQuantizer for ExRaBitQ.
 *
 * This class quantizes data into two sets of codes.
 * Its interface closely follows the CPU version. Later, you would replace
 * stub functions with CUDA kernels and device‐side implementations.
 */
class DataQuantizerGPU {
 private:
  size_t DIM;                // Original data dimension.
  size_t D;                  // Padded dimension (multiple of 64).
  size_t EX_BITS;            // Number of bits for ExRaBitQ.
  size_t SHORT_CODE_LENGTH;  // Number of uint32_t to store 1-bit code for a vector.
  size_t LONG_CODE_LENGTH;   // Number of uint8_t to store EX_BITS code for a vector.
  double FAC_NORM;
  double FAC_ERR;
  bool batch_flag_dq;
  float const_scaling_factor;
#if defined(HIGH_ACC_FAST_SCAN)
  static constexpr size_t NUM_SHORT_FACTORS = 1;
#else
  static constexpr size_t NUM_SHORT_FACTORS = 4;
#endif

  // Private helper functions (to be implemented with GPU kernels eventually):
  //    void pack_binary(const int* /*int matrix*/, uint64_t* out, size_t index) const;
  //    void rabitq_factor(const float* data, const float* centroid,
  //                       const std::vector<PID>& pids,
  //                       const int* intMat, const float* floatMat,
  //                       float* out1, float* out2, float* out3, float* out4) const;
  void rabitq_factor(const float* d_data,
                     const float* d_centroid,
                     const PID* d_IDs,
                     const int* d_bin_XP,
                     const float* d_XP_norm,
                     float* fac_x2,
                     float* fac_ip,
                     float* fac_sumxb,
                     float* fac_err,
                     size_t num_points) const;
  //    void fast_quantize(const float* data, uint8_t* code, float& factor) const;
  void rabitq_codes(const int* d_bin_XP, uint32_t* d_packed_code, size_t num_points) const;
  //    void exrabitq_codes(const int* intMat, const float* floatMat,
  //                         uint8_t* code, ExFactor* exFactor, const float* extra) const;
  void exrabitq_codes(const int* d_bin_XP,
                      const float* d_XP_norm,
                      uint8_t* d_long_code,
                      float* d_ex_factor,
                      const float* d_fac_x2,
                      size_t num_points) const;

  void store_compacted_code(uint8_t* dest, uint8_t* src) const;

  //     rd_up_to_multiple_of = [](uint32_t dim, uint32_t mult) -> size_t {
  //        return ((dim + mult - 1) / mult) * mult;
  //    };
 public:
  static float get_const_scaling_factors(size_t dim, size_t ex_bits);
  // Constructor: initialize from dimension and bit count.
  explicit DataQuantizerGPU(size_t dim, size_t b, bool batch_flag_dq = false)
    : DIM(dim),
      D(rd_up_to_multiple_of_new(dim, 64)),
      EX_BITS(b),
      SHORT_CODE_LENGTH((D + 31) / 32),
      LONG_CODE_LENGTH(D * EX_BITS / 8)  // Simplified for now.
      ,
      FAC_NORM(1 / std::sqrt((double)D)),
      FAC_ERR(2.0 / std::sqrt((double)(D - 1))),
      batch_flag_dq(batch_flag_dq),
      fast_quantize_flag(false)
  {
    const_scaling_factor = get_const_scaling_factors(dim, b);
  }

  explicit DataQuantizerGPU() {}

  // Assignment operator.
  DataQuantizerGPU& operator=(const DataQuantizerGPU& other)
  {
    this->DIM                  = other.DIM;
    this->D                    = other.D;
    this->EX_BITS              = other.EX_BITS;
    this->SHORT_CODE_LENGTH    = other.SHORT_CODE_LENGTH;
    this->LONG_CODE_LENGTH     = other.LONG_CODE_LENGTH;
    this->FAC_NORM             = other.FAC_NORM;
    this->FAC_ERR              = other.FAC_ERR;
    this->batch_flag_dq        = other.batch_flag_dq;
    this->fast_quantize_flag   = other.fast_quantize_flag;
    this->const_scaling_factor = other.const_scaling_factor;
    return *this;
  }

  // Accessor functions.
  size_t short_code_length() const { return SHORT_CODE_LENGTH; }
  size_t long_code_length() const { return LONG_CODE_LENGTH; }
  // Now block_types represents a single factor
  size_t block_bytes() const
  {
    if (!batch_flag_dq) {
      return SHORT_CODE_LENGTH * sizeof(uint32_t) + sizeof(float) * NUM_SHORT_FACTORS;
    } else {
      // 3 factors for batch: f_add, f_rescale and f_error are stored separately
      return SHORT_CODE_LENGTH * sizeof(uint32_t);
    }
  }  // May be useless
  size_t num_blocks(size_t num) const { return div_rd_up_new(num, FAST_SIZE); }
  static constexpr size_t num_short_factors() { return NUM_SHORT_FACTORS; }

  /*!
   * @brief Quantize the input data.
   *
   * @param data Pointer to the input data (host pointer or device pointer as needed).
   * @param centroid Pointer to the corresponding centroid.
   * @param pids Vector of vector IDs.
   * @param rotator A RotatorGPU instance.
   * @param outShort Output buffer for short codes.
   * @param outLong Output buffer for long codes.
   * @param outExFactor Output buffer for extra factors.
   * @param outTemp Temporary buffer.
   */
  //    void quantize(const float* data, const float* centroid,
  //                  const std::vector<PID>& pids,
  //                  const RotatorGPU& rotator,
  //                  uint8_t* outShort, uint8_t* outLong, ExFactor* outExFactor, float* outTemp)
  //                  const;
  void quantize(raft::resources const& handle,
                const float* d_data,
                const float* d_centroid,
                const PID* d_IDs,
                size_t num_points,
                const RotatorGPU& rotator,
                uint32_t* d_short_data,
                uint8_t* d_long_code,
                float* d_ex_factor,
                float* d_rotated_c) const;

  /*!
   * @brief Quantize the input data for batch layout.
   *
   * @param data Pointer to the input data (host pointer or device pointer as needed).
   * @param centroid Pointer to the corresponding centroid.
   * @param pids Vector of vector IDs.
   * @param rotator A RotatorGPU instance.
   * @param outShort Output buffer for short codes.
   * @param outLong Output buffer for long codes.
   * @param outExFactor Output buffer for extra factors.
   * @param outTemp Temporary buffer.
   */

  void quantize_batch(raft::resources const& handle,
                      const float* d_data,
                      const float* d_centroid,
                      const PID* d_IDs,
                      size_t num_points,
                      const RotatorGPU& rotator,
                      uint32_t* d_short_data,
                      float* short_data_factors,
                      uint8_t* d_long_code,
                      float* d_ex_factor,
                      float* d_rotated_c) const;

  void quantize_batch_opt(raft::resources const& handle,
                          const float* d_data,
                          const float* d_centroid,
                          const PID* d_IDs,
                          size_t num_points,
                          const RotatorGPU& rotator,
                          uint32_t* d_short_data,
                          float* short_data_factors,
                          uint8_t* d_long_code,
                          float* d_ex_factor,
                          float* d_rotated_c) const;

  /*!
   * @brief Get pointer of factors for the current block.
   *
   * Each block (stored in uint32_t*) contains the 1st bit of quantization code
   * and corresponding factors.
   */
  float* block_factor(uint32_t* block) const
  {
    return reinterpret_cast<float*>(&block[SHORT_CODE_LENGTH]);  // same as CPU version.
  }

  // Assume these helper inline functions are defined (they mirror the CPU versions):
  // block_factor: returns pointer to the factor section in a short code block.
  __host__ inline float* block_factor(uint32_t* block, size_t D) const
  {
    return reinterpret_cast<float*>(
      &block[SHORT_CODE_LENGTH]);  // Example: adjust offset appropriately.
  }
  __host__ inline float* factor_x2(uint32_t* block_fac) const
  {
    return reinterpret_cast<float*>(block_fac);  // In our layout, x2 is at the beginning.
  }
  __host__ inline float* factor_ip(uint32_t* block_fac, size_t FAST_SIZE) const
  {
    return reinterpret_cast<float*>(block_fac) + 1;  // adjust as needed.
  }
  __host__ inline float* factor_sumxb(uint32_t* block_fac, size_t FAST_SIZE) const
  {
    return reinterpret_cast<float*>(block_fac) + 2;
  }
  __host__ inline float* factor_err(uint32_t* block_fac, size_t FAST_SIZE) const
  {
    return reinterpret_cast<float*>(block_fac) + 3;
  }
  // next_block returns pointer to next block in the short_data buffer.
  // For this example, assume each block occupies: code_len + factor bytes.
  // TODO: remove unused num_factors
  __host__ inline uint32_t* next_block(uint32_t* block,
                                       size_t code_len,
                                       size_t FAST_SIZE,
                                       size_t num_factors = 1) const
  {
    return block + code_len + num_factors;
  }

  //    void data_transformation(const float* data, const float* centroid,
  //                             const std::vector<PID>& pids,
  //                             const RotatorGPU& rotator,
  //                             float* out, float* floatMat, int* intMat) const;
  void data_transformation(raft::resources const& handle,
                           const float* d_data,
                           const float* d_centroid,
                           const PID* d_IDs,
                           size_t num_points,
                           const RotatorGPU& rotator,
                           float* d_rotated_c,
                           float* d_XP_norm,
                           int* d_bin_XP) const;

  void data_transformation_batch(raft::resources const& handle,
                                 const float* d_data,
                                 const float* d_centroid,
                                 const PID* d_IDs,
                                 size_t num_points,
                                 const RotatorGPU& rotator,
                                 float* d_rotated_c,
                                 float* d_XP_norm,
                                 int* d_bin_XP,
                                 float* d_XP) const;

  void data_transformation_batch_opt(raft::resources const& handle,
                                     const float* d_data,
                                     const float* d_centroid,
                                     const PID* d_IDs,
                                     size_t num_points,
                                     const RotatorGPU& rotator,
                                     float* d_rotated_c,
                                     float* d_XP_norm,
                                     int* d_bin_XP,
                                     float* d_XP) const;

  void exrabitq_codes_hybrid_advanced(const int* d_bin_XP,
                                      const float* d_XP_norm,
                                      uint8_t* d_long_code,
                                      float* d_ex_factor,
                                      const float* d_fac_x2,
                                      size_t num_points) const;

  void exrabitq_codes_batch(const int* d_bin_XP,
                            const float* d_XP_norm,
                            float* d_XP,
                            uint8_t* d_long_code,
                            float* d_ex_factor,
                            const float* d_fac_x2,
                            size_t num_points) const;

  void exrabitq_codes_and_factors_fused(const int* d_bin_XP,
                                        const float* d_XP_norm,
                                        float* d_XP,
                                        uint8_t* d_long_code,
                                        float* d_ex_factor,
                                        const float* d_fac_x2,
                                        size_t num_points) const;

  void rabitq_codes_and_factors_fused(const float* d_rotated_c,
                                      const int* d_bin_XP,
                                      const float* d_XP,
                                      uint32_t* d_short_data,
                                      float* d_short_data_factors,
                                      size_t num_points) const;

  bool fast_quantize_flag;

  void exrabitq_codes_and_factors_fused_ori(const int* d_bin_XP,
                                            const float* d_XP_norm,
                                            float* d_XP,
                                            uint8_t* d_long_code,
                                            float* d_ex_factor,
                                            const float* d_centroid,
                                            size_t num_points) const;
};

#endif  // EXRABITQ_QUANTIZER_GPU_CUH

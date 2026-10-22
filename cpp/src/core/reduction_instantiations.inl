/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Included for extern declarations and owning definitions.
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    coalescedReduction<double, double, int, raft::identity_op, raft::min_op, raft::identity_op>(
      bool,
      double*,
      double const*,
      int,
      int,
      double,
      CUstream_st*,
      bool,
      raft::identity_op,
      raft::min_op,
      raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    coalescedReduction<double, double, long, raft::identity_op, raft::min_op, raft::identity_op>(
      bool,
      double*,
      double const*,
      long,
      long,
      double,
      CUstream_st*,
      bool,
      raft::identity_op,
      raft::min_op,
      raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    coalescedReduction<float, float, int, raft::identity_op, raft::min_op, raft::identity_op>(
      bool,
      float*,
      float const*,
      int,
      int,
      float,
      CUstream_st*,
      bool,
      raft::identity_op,
      raft::min_op,
      raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    coalescedReduction<float, float, long, raft::identity_op, raft::min_op, raft::identity_op>(
      bool,
      float*,
      float const*,
      long,
      long,
      float,
      CUstream_st*,
      bool,
      raft::identity_op,
      raft::min_op,
      raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(void raft::linalg::detail::coalescedReductionThin<
                             raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
                             double,
                             double,
                             int,
                             raft::identity_op,
                             raft::add_op,
                             raft::identity_op>(double*,
                                                double const*,
                                                int,
                                                int,
                                                double,
                                                cudaStream_t,
                                                bool,
                                                raft::identity_op,
                                                raft::add_op,
                                                raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(void raft::linalg::detail::coalescedReductionThin<
                             raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
                             double,
                             double,
                             int,
                             raft::identity_op,
                             raft::add_op,
                             raft::sqrt_op>(double*,
                                            double const*,
                                            int,
                                            int,
                                            double,
                                            cudaStream_t,
                                            bool,
                                            raft::identity_op,
                                            raft::add_op,
                                            raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(void raft::linalg::detail::coalescedReductionThin<
                             raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
                             double,
                             double,
                             int,
                             raft::identity_op,
                             raft::min_op,
                             raft::identity_op>(double*,
                                                double const*,
                                                int,
                                                int,
                                                double,
                                                cudaStream_t,
                                                bool,
                                                raft::identity_op,
                                                raft::min_op,
                                                raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(void raft::linalg::detail::coalescedReductionThin<
                             raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
                             double,
                             double,
                             long,
                             raft::identity_op,
                             raft::add_op,
                             raft::identity_op>(double*,
                                                double const*,
                                                long,
                                                long,
                                                double,
                                                cudaStream_t,
                                                bool,
                                                raft::identity_op,
                                                raft::add_op,
                                                raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(void raft::linalg::detail::coalescedReductionThin<
                             raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
                             double,
                             double,
                             long,
                             raft::identity_op,
                             raft::add_op,
                             raft::sqrt_op>(double*,
                                            double const*,
                                            long,
                                            long,
                                            double,
                                            cudaStream_t,
                                            bool,
                                            raft::identity_op,
                                            raft::add_op,
                                            raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(void raft::linalg::detail::coalescedReductionThin<
                             raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
                             double,
                             double,
                             long,
                             raft::identity_op,
                             raft::min_op,
                             raft::identity_op>(double*,
                                                double const*,
                                                long,
                                                long,
                                                double,
                                                cudaStream_t,
                                                bool,
                                                raft::identity_op,
                                                raft::min_op,
                                                raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(void raft::linalg::detail::coalescedReductionThin<
                             raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
                             double,
                             double,
                             unsigned int,
                             raft::identity_op,
                             raft::add_op,
                             raft::sqrt_op>(double*,
                                            double const*,
                                            unsigned int,
                                            unsigned int,
                                            double,
                                            cudaStream_t,
                                            bool,
                                            raft::identity_op,
                                            raft::add_op,
                                            raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(void raft::linalg::detail::coalescedReductionThin<
                             raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
                             float,
                             float,
                             int,
                             raft::identity_op,
                             raft::add_op,
                             raft::identity_op>(float*,
                                                float const*,
                                                int,
                                                int,
                                                float,
                                                cudaStream_t,
                                                bool,
                                                raft::identity_op,
                                                raft::add_op,
                                                raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(void raft::linalg::detail::coalescedReductionThin<
                             raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
                             float,
                             float,
                             int,
                             raft::identity_op,
                             raft::add_op,
                             raft::sqrt_op>(float*,
                                            float const*,
                                            int,
                                            int,
                                            float,
                                            cudaStream_t,
                                            bool,
                                            raft::identity_op,
                                            raft::add_op,
                                            raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(void raft::linalg::detail::coalescedReductionThin<
                             raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
                             float,
                             float,
                             int,
                             raft::identity_op,
                             raft::min_op,
                             raft::identity_op>(float*,
                                                float const*,
                                                int,
                                                int,
                                                float,
                                                cudaStream_t,
                                                bool,
                                                raft::identity_op,
                                                raft::min_op,
                                                raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(void raft::linalg::detail::coalescedReductionThin<
                             raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
                             float,
                             float,
                             long,
                             raft::identity_op,
                             raft::add_op,
                             raft::identity_op>(float*,
                                                float const*,
                                                long,
                                                long,
                                                float,
                                                cudaStream_t,
                                                bool,
                                                raft::identity_op,
                                                raft::add_op,
                                                raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::coalescedReductionThin<
    raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
    float,
    float,
    long,
    raft::identity_op,
    raft::add_op,
    raft::plug_const_op<float, raft::mul_op>>(float*,
                                              float const*,
                                              long,
                                              long,
                                              float,
                                              cudaStream_t,
                                              bool,
                                              raft::identity_op,
                                              raft::add_op,
                                              raft::plug_const_op<float, raft::mul_op>))
CUVS_REDUCTION_INSTANTIATION(void raft::linalg::detail::coalescedReductionThin<
                             raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
                             float,
                             float,
                             long,
                             raft::identity_op,
                             raft::add_op,
                             raft::sqrt_op>(float*,
                                            float const*,
                                            long,
                                            long,
                                            float,
                                            cudaStream_t,
                                            bool,
                                            raft::identity_op,
                                            raft::add_op,
                                            raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(void raft::linalg::detail::coalescedReductionThin<
                             raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
                             float,
                             float,
                             long,
                             raft::identity_op,
                             raft::min_op,
                             raft::identity_op>(float*,
                                                float const*,
                                                long,
                                                long,
                                                float,
                                                cudaStream_t,
                                                bool,
                                                raft::identity_op,
                                                raft::min_op,
                                                raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(void raft::linalg::detail::coalescedReductionThin<
                             raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
                             float,
                             float,
                             unsigned int,
                             raft::identity_op,
                             raft::add_op,
                             raft::identity_op>(float*,
                                                float const*,
                                                unsigned int,
                                                unsigned int,
                                                float,
                                                cudaStream_t,
                                                bool,
                                                raft::identity_op,
                                                raft::add_op,
                                                raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(void raft::linalg::detail::coalescedReductionThin<
                             raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
                             float,
                             float,
                             unsigned int,
                             raft::identity_op,
                             raft::add_op,
                             raft::sqrt_op>(float*,
                                            float const*,
                                            unsigned int,
                                            unsigned int,
                                            float,
                                            cudaStream_t,
                                            bool,
                                            raft::identity_op,
                                            raft::add_op,
                                            raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(void raft::linalg::detail::coalescedReductionThin<
                             raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
                             long,
                             long,
                             long,
                             raft::identity_op,
                             raft::add_op,
                             raft::identity_op>(long*,
                                                long const*,
                                                long,
                                                long,
                                                long,
                                                cudaStream_t,
                                                bool,
                                                raft::identity_op,
                                                raft::add_op,
                                                raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(void raft::linalg::detail::coalescedReductionThin<
                             raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
                             raft::KeyValuePair<int, double>,
                             raft::KeyValuePair<int, double>,
                             int,
                             raft::identity_op,
                             raft::argmin_op,
                             raft::identity_op>(raft::KeyValuePair<int, double>*,
                                                raft::KeyValuePair<int, double> const*,
                                                int,
                                                int,
                                                raft::KeyValuePair<int, double>,
                                                cudaStream_t,
                                                bool,
                                                raft::identity_op,
                                                raft::argmin_op,
                                                raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(void raft::linalg::detail::coalescedReductionThin<
                             raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
                             raft::KeyValuePair<int, float>,
                             raft::KeyValuePair<int, float>,
                             int,
                             raft::identity_op,
                             raft::argmin_op,
                             raft::identity_op>(raft::KeyValuePair<int, float>*,
                                                raft::KeyValuePair<int, float> const*,
                                                int,
                                                int,
                                                raft::KeyValuePair<int, float>,
                                                cudaStream_t,
                                                bool,
                                                raft::identity_op,
                                                raft::argmin_op,
                                                raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(void raft::linalg::detail::coalescedReductionThin<
                             raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
                             raft::KeyValuePair<long, double>,
                             raft::KeyValuePair<long, double>,
                             long,
                             raft::identity_op,
                             raft::argmin_op,
                             raft::identity_op>(raft::KeyValuePair<long, double>*,
                                                raft::KeyValuePair<long, double> const*,
                                                long,
                                                long,
                                                raft::KeyValuePair<long, double>,
                                                cudaStream_t,
                                                bool,
                                                raft::identity_op,
                                                raft::argmin_op,
                                                raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(void raft::linalg::detail::coalescedReductionThin<
                             raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
                             raft::KeyValuePair<long, float>,
                             raft::KeyValuePair<long, float>,
                             long,
                             raft::identity_op,
                             raft::argmin_op,
                             raft::identity_op>(raft::KeyValuePair<long, float>*,
                                                raft::KeyValuePair<long, float> const*,
                                                long,
                                                long,
                                                raft::KeyValuePair<long, float>,
                                                cudaStream_t,
                                                bool,
                                                raft::identity_op,
                                                raft::argmin_op,
                                                raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(void raft::linalg::detail::coalescedReductionThin<
                             raft::linalg::detail::ReductionThinPolicy<32, 128, 1, false>,
                             unsigned int,
                             unsigned int,
                             unsigned int,
                             raft::identity_op,
                             raft::add_op,
                             raft::identity_op>(unsigned int*,
                                                unsigned int const*,
                                                unsigned int,
                                                unsigned int,
                                                unsigned int,
                                                cudaStream_t,
                                                bool,
                                                raft::identity_op,
                                                raft::add_op,
                                                raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::reduce<
    true,
    true,
    __half,
    float,
    long,
    raft::compose_op<raft::sq_op, raft::plug_const_op<float, raft::div_op>, raft::cast_op<float>>,
    raft::add_op,
    raft::sqrt_op>(
    bool,
    float*,
    __half const*,
    long,
    long,
    float,
    CUstream_st*,
    bool,
    raft::compose_op<raft::sq_op, raft::plug_const_op<float, raft::div_op>, raft::cast_op<float>>,
    raft::add_op,
    raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::reduce<
    true,
    true,
    float,
    float,
    long,
    raft::compose_op<raft::sq_op, raft::plug_const_op<float, raft::div_op>, raft::cast_op<float>>,
    raft::add_op,
    raft::sqrt_op>(
    bool,
    float*,
    float const*,
    long,
    long,
    float,
    CUstream_st*,
    bool,
    raft::compose_op<raft::sq_op, raft::plug_const_op<float, raft::div_op>, raft::cast_op<float>>,
    raft::add_op,
    raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::reduce<
    true,
    true,
    signed char,
    float,
    long,
    raft::compose_op<raft::sq_op, raft::plug_const_op<float, raft::div_op>, raft::cast_op<float>>,
    raft::add_op,
    raft::sqrt_op>(
    bool,
    float*,
    signed char const*,
    long,
    long,
    float,
    CUstream_st*,
    bool,
    raft::compose_op<raft::sq_op, raft::plug_const_op<float, raft::div_op>, raft::cast_op<float>>,
    raft::add_op,
    raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::reduce<
    true,
    true,
    unsigned char,
    float,
    long,
    raft::compose_op<raft::sq_op, raft::plug_const_op<float, raft::div_op>, raft::cast_op<float>>,
    raft::add_op,
    raft::sqrt_op>(
    bool,
    float*,
    unsigned char const*,
    long,
    long,
    float,
    CUstream_st*,
    bool,
    raft::compose_op<raft::sq_op, raft::plug_const_op<float, raft::div_op>, raft::cast_op<float>>,
    raft::add_op,
    raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    reduce<false, true, __half, float, int, raft::identity_op, raft::add_op, raft::identity_op>(
      bool,
      float*,
      __half const*,
      int,
      int,
      float,
      CUstream_st*,
      bool,
      raft::identity_op,
      raft::add_op,
      raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    reduce<false, true, double, double, int, raft::identity_op, raft::add_op, raft::identity_op>(
      bool,
      double*,
      double const*,
      int,
      int,
      double,
      CUstream_st*,
      bool,
      raft::identity_op,
      raft::add_op,
      raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    reduce<false, true, float, float, int, raft::identity_op, raft::add_op, raft::identity_op>(
      bool,
      float*,
      float const*,
      int,
      int,
      float,
      CUstream_st*,
      bool,
      raft::identity_op,
      raft::add_op,
      raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    reduce<true, false, double, double, int, raft::identity_op, raft::add_op, raft::identity_op>(
      bool,
      double*,
      double const*,
      int,
      int,
      double,
      CUstream_st*,
      bool,
      raft::identity_op,
      raft::add_op,
      raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    reduce<true, false, float, float, int, raft::identity_op, raft::add_op, raft::identity_op>(
      bool,
      float*,
      float const*,
      int,
      int,
      float,
      CUstream_st*,
      bool,
      raft::identity_op,
      raft::add_op,
      raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    reduce<true, false, float, float, long, raft::identity_op, raft::add_op, raft::identity_op>(
      bool,
      float*,
      float const*,
      long,
      long,
      float,
      CUstream_st*,
      bool,
      raft::identity_op,
      raft::add_op,
      raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    reduce<true, true, __half, float, int, raft::identity_op, raft::add_op, raft::identity_op>(
      bool,
      float*,
      __half const*,
      int,
      int,
      float,
      CUstream_st*,
      bool,
      raft::identity_op,
      raft::add_op,
      raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    reduce<true, true, double, double, int, raft::identity_op, raft::add_op, raft::identity_op>(
      bool,
      double*,
      double const*,
      int,
      int,
      double,
      CUstream_st*,
      bool,
      raft::identity_op,
      raft::add_op,
      raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    reduce<true, true, double, double, int, raft::identity_op, raft::min_op, raft::identity_op>(
      bool,
      double*,
      double const*,
      int,
      int,
      double,
      CUstream_st*,
      bool,
      raft::identity_op,
      raft::min_op,
      raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    reduce<true, true, double, double, long, raft::identity_op, raft::add_op, raft::identity_op>(
      bool,
      double*,
      double const*,
      long,
      long,
      double,
      CUstream_st*,
      bool,
      raft::identity_op,
      raft::add_op,
      raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    reduce<true, true, double, double, long, raft::identity_op, raft::min_op, raft::identity_op>(
      bool,
      double*,
      double const*,
      long,
      long,
      double,
      CUstream_st*,
      bool,
      raft::identity_op,
      raft::min_op,
      raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    reduce<true, true, float, float, int, raft::identity_op, raft::add_op, raft::identity_op>(
      bool,
      float*,
      float const*,
      int,
      int,
      float,
      CUstream_st*,
      bool,
      raft::identity_op,
      raft::add_op,
      raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    reduce<true, true, float, float, int, raft::identity_op, raft::min_op, raft::identity_op>(
      bool,
      float*,
      float const*,
      int,
      int,
      float,
      CUstream_st*,
      bool,
      raft::identity_op,
      raft::min_op,
      raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    reduce<true, true, float, float, long, raft::identity_op, raft::add_op, raft::identity_op>(
      bool,
      float*,
      float const*,
      long,
      long,
      float,
      CUstream_st*,
      bool,
      raft::identity_op,
      raft::add_op,
      raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    reduce<true, true, float, float, long, raft::identity_op, raft::min_op, raft::identity_op>(
      bool,
      float*,
      float const*,
      long,
      long,
      float,
      CUstream_st*,
      bool,
      raft::identity_op,
      raft::min_op,
      raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    reduce<true, true, float, float, long, raft::sq_op, raft::add_op, raft::identity_op>(
      bool,
      float*,
      float const*,
      long,
      long,
      float,
      CUstream_st*,
      bool,
      raft::sq_op,
      raft::add_op,
      raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, false, __half, int, raft::identity_op, float>(
      bool, float*, __half const*, int, int, CUstream_st*, raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, false, __half, int, raft::sqrt_op, float>(
      bool, float*, __half const*, int, int, CUstream_st*, raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, false, __half, long, raft::identity_op, float>(
      bool, float*, __half const*, long, long, CUstream_st*, raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, false, __half, long, raft::sqrt_op, float>(
      bool, float*, __half const*, long, long, CUstream_st*, raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, false, double, int, raft::identity_op, double>(
      bool, double*, double const*, int, int, CUstream_st*, raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, false, double, int, raft::sqrt_op, double>(
      bool, double*, double const*, int, int, CUstream_st*, raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, false, double, long, raft::identity_op, double>(
      bool, double*, double const*, long, long, CUstream_st*, raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, false, float, int, raft::identity_op, float>(
      bool, float*, float const*, int, int, CUstream_st*, raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, false, float, int, raft::sqrt_op, float>(
      bool, float*, float const*, int, int, CUstream_st*, raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, false, float, long, raft::identity_op, float>(
      bool, float*, float const*, long, long, CUstream_st*, raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, false, float, long, raft::sqrt_op, float>(
      bool, float*, float const*, long, long, CUstream_st*, raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, true, __half, int, raft::identity_op, float>(
      bool, float*, __half const*, int, int, CUstream_st*, raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, true, __half, int, raft::sqrt_op, float>(
      bool, float*, __half const*, int, int, CUstream_st*, raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, true, __half, long, raft::identity_op, float>(
      bool, float*, __half const*, long, long, CUstream_st*, raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, true, __half, long, raft::sqrt_op, float>(
      bool, float*, __half const*, long, long, CUstream_st*, raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, true, double, int, raft::identity_op, double>(
      bool, double*, double const*, int, int, CUstream_st*, raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, true, double, int, raft::sqrt_op, double>(
      bool, double*, double const*, int, int, CUstream_st*, raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, true, double, long, raft::identity_op, double>(
      bool, double*, double const*, long, long, CUstream_st*, raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, true, double, long, raft::sqrt_op, double>(
      bool, double*, double const*, long, long, CUstream_st*, raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, true, float, int, raft::identity_op, float>(
      bool, float*, float const*, int, int, CUstream_st*, raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, true, float, int, raft::sqrt_op, float>(
      bool, float*, float const*, int, int, CUstream_st*, raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, true, float, long, raft::identity_op, float>(
      bool, float*, float const*, long, long, CUstream_st*, raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, true, float, long, raft::sqrt_op, float>(
      bool, float*, float const*, long, long, CUstream_st*, raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, true, float, unsigned int, raft::identity_op, float>(
      bool, float*, float const*, unsigned int, unsigned int, CUstream_st*, raft::identity_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::
    rowNormCaller<(raft::linalg::NormType)2, true, float, unsigned int, raft::sqrt_op, float>(
      bool, float*, float const*, unsigned int, unsigned int, CUstream_st*, raft::sqrt_op))
CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::stridedReduction<float,
                                              float,
                                              int64_t,
                                              raft::identity_op,
                                              raft::add_op,
                                              raft::mul_const_op<float>>(float*,
                                                                         const float*,
                                                                         int64_t,
                                                                         int64_t,
                                                                         float,
                                                                         cudaStream_t,
                                                                         bool,
                                                                         raft::identity_op,
                                                                         raft::add_op,
                                                                         raft::mul_const_op<float>))

CUVS_REDUCTION_INSTANTIATION(
  void raft::linalg::detail::coalescedReduction<float,
                                                float,
                                                long,
                                                raft::identity_op,
                                                raft::add_op,
                                                raft::mul_const_op<float>>(
    bool,
    float*,
    float const*,
    long,
    long,
    float,
    cudaStream_t,
    bool,
    raft::identity_op,
    raft::add_op,
    raft::mul_const_op<float>))

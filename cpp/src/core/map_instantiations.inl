/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Included once for extern declarations and once for owning definitions.
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<false, __half, int, raft::sq_op, __half>(
  cuda::stream_ref, __half*, int, raft::sq_op, __half const*))
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<false, __half, int, raft::sqrt_op, __half>(
  cuda::stream_ref, __half*, int, raft::sqrt_op, __half const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, __half, long, raft::cast_op<__half>, float>(
    cuda::stream_ref, __half*, long, raft::cast_op<__half>, float const*))
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<false, __half, long, raft::const_op<__half>>(
  cuda::stream_ref, __half*, long, raft::const_op<__half>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::
    map<false, __half, unsigned int, cuvs::spatial::knn::detail::utils::mapping<__half>, float>(
      cuda::stream_ref,
      __half*,
      unsigned int,
      cuvs::spatial::knn::detail::utils::mapping<__half>,
      float const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, __half, unsigned long, raft::cast_op<__half>, float>(
    cuda::stream_ref, __half*, unsigned long, raft::cast_op<__half>, float const*))
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<false, bool, long, raft::const_op<bool>>(
  cuda::stream_ref, bool*, long, raft::const_op<bool>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, double, int, raft::add_op, double, double>(
    cuda::stream_ref, double*, int, raft::add_op, double const*, double const*))
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<false, double, int, raft::const_op<double>>(
  cuda::stream_ref, double*, int, raft::const_op<double>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, double, int, raft::const_op<double>, double>(
    cuda::stream_ref, double*, int, raft::const_op<double>, double const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, double, int, raft::identity_op, double>(
    cuda::stream_ref, double*, int, raft::identity_op, double const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, double, int, raft::min_op, double, double>(
    cuda::stream_ref, double*, int, raft::min_op, double const*, double const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, double, int, raft::mul_op, double, double>(
    cuda::stream_ref, double*, int, raft::mul_op, double const*, double const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::
    map<false, double, int, raft::sparse::linalg::detail::zero_to_one_functor, double>(
      cuda::stream_ref,
      double*,
      int,
      raft::sparse::linalg::detail::zero_to_one_functor,
      double const*))
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<false, double, int, raft::sq_op, double>(
  cuda::stream_ref, double*, int, raft::sq_op, double const*))
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<false, double, int, raft::sqrt_op, double>(
  cuda::stream_ref, double*, int, raft::sqrt_op, double const*))
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<false, double, long, raft::const_op<double>>(
  cuda::stream_ref, double*, long, raft::const_op<double>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, double, long, raft::const_op<double>, double>(
    cuda::stream_ref, double*, long, raft::const_op<double>, double const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, double, long, raft::identity_op, double>(
    cuda::stream_ref, double*, long, raft::identity_op, double const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, double, long, raft::min_op, double, double>(
    cuda::stream_ref, double*, long, raft::min_op, double const*, double const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, double, long, raft::mul_op, double, double>(
    cuda::stream_ref, double*, long, raft::mul_op, double const*, double const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::
    map<false, double, long, raft::plug_const_op<double, raft::mul_op>, double>(
      cuda::stream_ref, double*, long, raft::plug_const_op<double, raft::mul_op>, double const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::
    map<false, double, long, raft::plug_const_op<double, raft::pow_op>, double>(
      cuda::stream_ref, double*, long, raft::plug_const_op<double, raft::pow_op>, double const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, double, unsigned int, raft::add_op, double, double>(
    cuda::stream_ref, double*, unsigned int, raft::add_op, double const*, double const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, double, unsigned int, raft::const_op<double>>(
    cuda::stream_ref, double*, unsigned int, raft::const_op<double>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, double, unsigned long, raft::abs_op, double>(
    cuda::stream_ref, double*, unsigned long, raft::abs_op, double const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, float, int, raft::add_op, float, float>(
    cuda::stream_ref, float*, int, raft::add_op, float const*, float const*))
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<false, float, int, raft::const_op<float>>(
  cuda::stream_ref, float*, int, raft::const_op<float>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, float, int, raft::const_op<float>, float>(
    cuda::stream_ref, float*, int, raft::const_op<float>, float const*))
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<false, float, int, raft::identity_op, float>(
  cuda::stream_ref, float*, int, raft::identity_op, float const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, float, int, raft::min_op, float, float>(
    cuda::stream_ref, float*, int, raft::min_op, float const*, float const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, float, int, raft::mul_op, float, float>(
    cuda::stream_ref, float*, int, raft::mul_op, float const*, float const*))
CUVS_MAP_INSTANTIATION(
  void
    raft::linalg::detail::map<false, float, int, raft::plug_const_op<float, raft::mul_op>, float>(
      cuda::stream_ref, float*, int, raft::plug_const_op<float, raft::mul_op>, float const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::
    map<false, float, int, raft::sparse::linalg::detail::zero_to_one_functor, float>(
      cuda::stream_ref,
      float*,
      int,
      raft::sparse::linalg::detail::zero_to_one_functor,
      float const*))
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<false, float, int, raft::sq_op, float>(
  cuda::stream_ref, float*, int, raft::sq_op, float const*))
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<false, float, int, raft::sqrt_op, float>(
  cuda::stream_ref, float*, int, raft::sqrt_op, float const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::
    map<false, float, long, cuvs::spatial::knn::detail::utils::mapping<float>, __half>(
      cuda::stream_ref,
      float*,
      long,
      cuvs::spatial::knn::detail::utils::mapping<float>,
      __half const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::
    map<false, float, long, cuvs::spatial::knn::detail::utils::mapping<float>, float>(
      cuda::stream_ref,
      float*,
      long,
      cuvs::spatial::knn::detail::utils::mapping<float>,
      float const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::
    map<false, float, long, cuvs::spatial::knn::detail::utils::mapping<float>, signed char>(
      cuda::stream_ref,
      float*,
      long,
      cuvs::spatial::knn::detail::utils::mapping<float>,
      signed char const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::
    map<false, float, long, cuvs::spatial::knn::detail::utils::mapping<float>, unsigned char>(
      cuda::stream_ref,
      float*,
      long,
      cuvs::spatial::knn::detail::utils::mapping<float>,
      unsigned char const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, float, long, raft::cast_op<float>, __half>(
    cuda::stream_ref, float*, long, raft::cast_op<float>, __half const*))
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<false, float, long, raft::const_op<float>>(
  cuda::stream_ref, float*, long, raft::const_op<float>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, float, long, raft::const_op<float>, float>(
    cuda::stream_ref, float*, long, raft::const_op<float>, float const*))
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<false, float, long, raft::identity_op, float>(
  cuda::stream_ref, float*, long, raft::identity_op, float const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, float, long, raft::min_op, float, float>(
    cuda::stream_ref, float*, long, raft::min_op, float const*, float const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, float, long, raft::mul_op, float, float>(
    cuda::stream_ref, float*, long, raft::mul_op, float const*, float const*))
CUVS_MAP_INSTANTIATION(
  void
    raft::linalg::detail::map<false, float, long, raft::plug_const_op<float, raft::mul_op>, float>(
      cuda::stream_ref, float*, long, raft::plug_const_op<float, raft::mul_op>, float const*))
CUVS_MAP_INSTANTIATION(
  void
    raft::linalg::detail::map<false, float, long, raft::plug_const_op<float, raft::pow_op>, float>(
      cuda::stream_ref, float*, long, raft::plug_const_op<float, raft::pow_op>, float const*))
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<false, float, long, raft::sq_op, float>(
  cuda::stream_ref, float*, long, raft::sq_op, float const*))
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<false, float, long, raft::sqrt_op, float>(
  cuda::stream_ref, float*, long, raft::sqrt_op, float const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::
    map<false, float, unsigned int, cuvs::spatial::knn::detail::utils::mapping<float>, __half>(
      cuda::stream_ref,
      float*,
      unsigned int,
      cuvs::spatial::knn::detail::utils::mapping<float>,
      __half const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::
    map<false, float, unsigned int, cuvs::spatial::knn::detail::utils::mapping<float>, signed char>(
      cuda::stream_ref,
      float*,
      unsigned int,
      cuvs::spatial::knn::detail::utils::mapping<float>,
      signed char const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false,
                                 float,
                                 unsigned int,
                                 cuvs::spatial::knn::detail::utils::mapping<float>,
                                 unsigned char>(cuda::stream_ref,
                                                float*,
                                                unsigned int,
                                                cuvs::spatial::knn::detail::utils::mapping<float>,
                                                unsigned char const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, float, unsigned int, raft::add_op, float, float>(
    cuda::stream_ref, float*, unsigned int, raft::add_op, float const*, float const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, float, unsigned int, raft::const_op<float>>(
    cuda::stream_ref, float*, unsigned int, raft::const_op<float>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, float, unsigned long, raft::cast_op<float>, __half>(
    cuda::stream_ref, float*, unsigned long, raft::cast_op<float>, __half const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, float, unsigned long, raft::cast_op<float>, float>(
    cuda::stream_ref, float*, unsigned long, raft::cast_op<float>, float const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, float, unsigned long, raft::cast_op<float>, signed char>(
    cuda::stream_ref, float*, unsigned long, raft::cast_op<float>, signed char const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, float, unsigned long, raft::cast_op<float>, unsigned char>(
    cuda::stream_ref, float*, unsigned long, raft::cast_op<float>, unsigned char const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<
    false,
    float,
    unsigned long,
    raft::compose_op<raft::plug_const_op<float, raft::mul_op>, raft::cast_op<float>>,
    __half>(cuda::stream_ref,
            float*,
            unsigned long,
            raft::compose_op<raft::plug_const_op<float, raft::mul_op>, raft::cast_op<float>>,
            __half const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<
    false,
    float,
    unsigned long,
    raft::compose_op<raft::plug_const_op<float, raft::mul_op>, raft::cast_op<float>>,
    float>(cuda::stream_ref,
           float*,
           unsigned long,
           raft::compose_op<raft::plug_const_op<float, raft::mul_op>, raft::cast_op<float>>,
           float const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<
    false,
    float,
    unsigned long,
    raft::compose_op<raft::plug_const_op<float, raft::mul_op>, raft::sqrt_op, raft::cast_op<float>>,
    __half>(
    cuda::stream_ref,
    float*,
    unsigned long,
    raft::compose_op<raft::plug_const_op<float, raft::mul_op>, raft::sqrt_op, raft::cast_op<float>>,
    __half const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<
    false,
    float,
    unsigned long,
    raft::compose_op<raft::plug_const_op<float, raft::mul_op>, raft::sqrt_op, raft::cast_op<float>>,
    float>(
    cuda::stream_ref,
    float*,
    unsigned long,
    raft::compose_op<raft::plug_const_op<float, raft::mul_op>, raft::sqrt_op, raft::cast_op<float>>,
    float const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::
    map<false, float, unsigned long, raft::compose_op<raft::sqrt_op, raft::cast_op<float>>, __half>(
      cuda::stream_ref,
      float*,
      unsigned long,
      raft::compose_op<raft::sqrt_op, raft::cast_op<float>>,
      __half const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::
    map<false, float, unsigned long, raft::compose_op<raft::sqrt_op, raft::cast_op<float>>, float>(
      cuda::stream_ref,
      float*,
      unsigned long,
      raft::compose_op<raft::sqrt_op, raft::cast_op<float>>,
      float const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, float, unsigned long, raft::const_op<float>>(
    cuda::stream_ref, float*, unsigned long, raft::const_op<float>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, float, unsigned long, raft::sqrt_op, __half>(
    cuda::stream_ref, float*, unsigned long, raft::sqrt_op, __half const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, float, unsigned long, raft::sqrt_op, float>(
    cuda::stream_ref, float*, unsigned long, raft::sqrt_op, float const*))
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<false, int, int, raft::const_op<int>>(
  cuda::stream_ref, int*, int, raft::const_op<int>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, int, int, raft::plug_const_op<int, raft::sub_op>, int>(
    cuda::stream_ref, int*, int, raft::plug_const_op<int, raft::sub_op>, int const*))
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<false, int, long, raft::const_op<int>>(
  cuda::stream_ref, int*, long, raft::const_op<int>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, int, unsigned long, raft::cast_op<int>, long>(
    cuda::stream_ref, int*, unsigned long, raft::cast_op<int>, long const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, int, unsigned long, raft::const_op<int>>(
    cuda::stream_ref, int*, unsigned long, raft::const_op<int>))
#if CUDART_VERSION >= 13000
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<false, long, int, raft::cast_op<long>, int>(
  cuda::stream_ref, long*, int, raft::cast_op<long>, int const*))
#endif
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, long, long, raft::cast_op<long>, unsigned int>(
    cuda::stream_ref, long*, long, raft::cast_op<long>, unsigned int const*))
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<false, long, long, raft::const_op<long>>(
  cuda::stream_ref, long*, long, raft::const_op<long>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, long, unsigned int, raft::const_op<long>>(
    cuda::stream_ref, long*, unsigned int, raft::const_op<long>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, long, unsigned long, raft::cast_op<long>, long>(
    cuda::stream_ref, long*, unsigned long, raft::cast_op<long>, long const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, long, unsigned long, raft::const_op<long>>(
    cuda::stream_ref, long*, unsigned long, raft::const_op<long>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::
    map<false, long, unsigned long, raft::plug_const_op<long, raft::add_op>, long>(
      cuda::stream_ref, long*, unsigned long, raft::plug_const_op<long, raft::add_op>, long const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false,
                                 raft::KeyValuePair<int, double>,
                                 int,
                                 raft::const_op<raft::KeyValuePair<int, double>>>(
    cuda::stream_ref,
    raft::KeyValuePair<int, double>*,
    int,
    raft::const_op<raft::KeyValuePair<int, double>>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::
    map<false, raft::KeyValuePair<int, float>, int, raft::const_op<raft::KeyValuePair<int, float>>>(
      cuda::stream_ref,
      raft::KeyValuePair<int, float>*,
      int,
      raft::const_op<raft::KeyValuePair<int, float>>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false,
                                 raft::KeyValuePair<long, double>,
                                 long,
                                 raft::const_op<raft::KeyValuePair<long, double>>>(
    cuda::stream_ref,
    raft::KeyValuePair<long, double>*,
    long,
    raft::const_op<raft::KeyValuePair<long, double>>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false,
                                 raft::KeyValuePair<long, float>,
                                 long,
                                 raft::const_op<raft::KeyValuePair<long, float>>>(
    cuda::stream_ref,
    raft::KeyValuePair<long, float>*,
    long,
    raft::const_op<raft::KeyValuePair<long, float>>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, signed char, long, raft::const_op<signed char>>(
    cuda::stream_ref, signed char*, long, raft::const_op<signed char>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false,
                                 signed char,
                                 unsigned int,
                                 cuvs::spatial::knn::detail::utils::mapping<signed char>,
                                 float>(cuda::stream_ref,
                                        signed char*,
                                        unsigned int,
                                        cuvs::spatial::knn::detail::utils::mapping<signed char>,
                                        float const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false,
                                 signed char,
                                 unsigned long,
                                 cuvs::spatial::knn::detail::utils::mapping<signed char>,
                                 float>(cuda::stream_ref,
                                        signed char*,
                                        unsigned long,
                                        cuvs::spatial::knn::detail::utils::mapping<signed char>,
                                        float const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, unsigned char, long, raft::const_op<unsigned char>>(
    cuda::stream_ref, unsigned char*, long, raft::const_op<unsigned char>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false,
                                 unsigned char,
                                 unsigned int,
                                 cuvs::spatial::knn::detail::utils::mapping<unsigned char>,
                                 float>(cuda::stream_ref,
                                        unsigned char*,
                                        unsigned int,
                                        cuvs::spatial::knn::detail::utils::mapping<unsigned char>,
                                        float const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, unsigned char, unsigned int, raft::const_op<unsigned char>>(
    cuda::stream_ref, unsigned char*, unsigned int, raft::const_op<unsigned char>))
CUVS_MAP_INSTANTIATION(
  void
    raft::linalg::detail::map<false, unsigned int, long, raft::add_op, unsigned int, unsigned int>(
      cuda::stream_ref,
      unsigned int*,
      long,
      raft::add_op,
      unsigned int const*,
      unsigned int const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, unsigned int, long, raft::const_op<unsigned int>>(
    cuda::stream_ref, unsigned int*, long, raft::const_op<unsigned int>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::
    map<false, unsigned int, long, raft::plug_const_op<unsigned int, raft::div_op>, unsigned int>(
      cuda::stream_ref,
      unsigned int*,
      long,
      raft::plug_const_op<unsigned int, raft::div_op>,
      unsigned int const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::
    map<false, unsigned int, unsigned int, raft::add_op, unsigned int, unsigned int>(
      cuda::stream_ref,
      unsigned int*,
      unsigned int,
      raft::add_op,
      unsigned int const*,
      unsigned int const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, unsigned int, unsigned int, raft::const_op<unsigned int>>(
    cuda::stream_ref, unsigned int*, unsigned int, raft::const_op<unsigned int>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, unsigned int, unsigned long, raft::const_op<unsigned int>>(
    cuda::stream_ref, unsigned int*, unsigned long, raft::const_op<unsigned int>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false,
                                 unsigned int,
                                 unsigned long,
                                 raft::plug_const_op<unsigned int, raft::add_op>,
                                 unsigned int>(cuda::stream_ref,
                                               unsigned int*,
                                               unsigned long,
                                               raft::plug_const_op<unsigned int, raft::add_op>,
                                               unsigned int const*))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, unsigned long, int, raft::const_op<unsigned long>>(
    cuda::stream_ref, unsigned long*, int, raft::const_op<unsigned long>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<false, unsigned long, long, raft::const_op<unsigned long>>(
    cuda::stream_ref, unsigned long*, long, raft::const_op<unsigned long>))
CUVS_MAP_INSTANTIATION(
  void
    raft::linalg::detail::map<false, unsigned long, unsigned long, raft::const_op<unsigned long>>(
      cuda::stream_ref, unsigned long*, unsigned long, raft::const_op<unsigned long>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::
    map<false, unsigned long long, long, raft::add_op, unsigned long long, unsigned long long>(
      cuda::stream_ref,
      unsigned long long*,
      long,
      raft::add_op,
      unsigned long long const*,
      unsigned long long const*))
CUVS_MAP_INSTANTIATION(
  void
    raft::linalg::detail::map<false, unsigned short, unsigned int, raft::const_op<unsigned short>>(
      cuda::stream_ref, unsigned short*, unsigned int, raft::const_op<unsigned short>))
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<true, int, int, raft::identity_op>(
  cuda::stream_ref, int*, int, raft::identity_op))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<true, int, int, raft::plug_const_op<int, raft::mul_op>>(
    cuda::stream_ref, int*, int, raft::plug_const_op<int, raft::mul_op>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<true, int, unsigned long, raft::plug_const_op<int, raft::mod_op>>(
    cuda::stream_ref, int*, unsigned long, raft::plug_const_op<int, raft::mod_op>))
CUVS_MAP_INSTANTIATION(void raft::linalg::detail::map<true, long, long, raft::identity_op>(
  cuda::stream_ref, long*, long, raft::identity_op))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<true, long, long, raft::plug_const_op<long, raft::div_op>>(
    cuda::stream_ref, long*, long, raft::plug_const_op<long, raft::div_op>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<true, long, long, raft::plug_const_op<long, raft::mul_op>>(
    cuda::stream_ref, long*, long, raft::plug_const_op<long, raft::mul_op>))
CUVS_MAP_INSTANTIATION(
  void
    raft::linalg::detail::map<true, long, unsigned long, raft::plug_const_op<long, raft::mod_op>>(
      cuda::stream_ref, long*, unsigned long, raft::plug_const_op<long, raft::mod_op>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<true, unsigned int, long, raft::cast_op<unsigned int>>(
    cuda::stream_ref, unsigned int*, long, raft::cast_op<unsigned int>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<
    true,
    unsigned int,
    long,
    raft::compose_op<raft::cast_op<unsigned int>, raft::plug_const_op<long, raft::div_op>>>(
    cuda::stream_ref,
    unsigned int*,
    long,
    raft::compose_op<raft::cast_op<unsigned int>, raft::plug_const_op<long, raft::div_op>>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<
    true,
    unsigned int,
    long,
    raft::compose_op<raft::cast_op<unsigned int>, raft::plug_const_op<long, raft::mod_op>>>(
    cuda::stream_ref,
    unsigned int*,
    long,
    raft::compose_op<raft::cast_op<unsigned int>, raft::plug_const_op<long, raft::mod_op>>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::map<true, unsigned int, unsigned int, raft::identity_op>(
    cuda::stream_ref, unsigned int*, unsigned int, raft::identity_op))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::
    map<true, unsigned int, unsigned int, raft::plug_const_op<unsigned int, raft::mul_op>>(
      cuda::stream_ref,
      unsigned int*,
      unsigned int,
      raft::plug_const_op<unsigned int, raft::mul_op>))
CUVS_MAP_INSTANTIATION(
  void raft::linalg::detail::
    map<true, unsigned int, unsigned long, raft::plug_const_op<unsigned int, raft::mod_op>>(
      cuda::stream_ref,
      unsigned int*,
      unsigned long,
      raft::plug_const_op<unsigned int, raft::mod_op>))

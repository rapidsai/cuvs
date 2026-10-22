/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Included for extern declarations and owning definitions.
CUVS_SELECTION_INSTANTIATION(
  void raft::matrix::detail::select_k<__half, uint32_t>(const raft::resources&,
                                                        const __half*,
                                                        const uint32_t*,
                                                        size_t,
                                                        size_t,
                                                        int,
                                                        __half*,
                                                        uint32_t*,
                                                        bool,
                                                        bool,
                                                        raft::matrix::SelectAlgo,
                                                        const uint32_t*))
CUVS_SELECTION_INSTANTIATION(
  void raft::matrix::detail::select_k<float, int64_t>(const raft::resources&,
                                                      const float*,
                                                      const int64_t*,
                                                      size_t,
                                                      size_t,
                                                      int,
                                                      float*,
                                                      int64_t*,
                                                      bool,
                                                      bool,
                                                      raft::matrix::SelectAlgo,
                                                      const int64_t*))
CUVS_SELECTION_INSTANTIATION(
  void raft::matrix::detail::select_k<float, int>(const raft::resources&,
                                                  const float*,
                                                  const int*,
                                                  size_t,
                                                  size_t,
                                                  int,
                                                  float*,
                                                  int*,
                                                  bool,
                                                  bool,
                                                  raft::matrix::SelectAlgo,
                                                  const int*))
CUVS_SELECTION_INSTANTIATION(
  void raft::matrix::detail::select_k<float, uint32_t>(const raft::resources&,
                                                       const float*,
                                                       const uint32_t*,
                                                       size_t,
                                                       size_t,
                                                       int,
                                                       float*,
                                                       uint32_t*,
                                                       bool,
                                                       bool,
                                                       raft::matrix::SelectAlgo,
                                                       const uint32_t*))
CUVS_SELECTION_INSTANTIATION(
  void raft::matrix::detail::select_k<int, uint32_t>(const raft::resources&,
                                                     const int*,
                                                     const uint32_t*,
                                                     size_t,
                                                     size_t,
                                                     int,
                                                     int*,
                                                     uint32_t*,
                                                     bool,
                                                     bool,
                                                     raft::matrix::SelectAlgo,
                                                     const uint32_t*))

CUVS_SELECTION_INSTANTIATION(void raft::sparse::matrix::detail::select_k<float, int64_t>(
  const raft::resources&,
  raft::device_csr_matrix_view<const float, int64_t, int64_t, int64_t>,
  std::optional<raft::device_vector_view<const int64_t, int64_t>>,
  raft::device_matrix_view<float, int64_t, raft::row_major>,
  raft::device_matrix_view<int64_t, int64_t, raft::row_major>,
  bool,
  bool,
  raft::matrix::SelectAlgo))

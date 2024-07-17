/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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
#pragma once

#include "detail/kernels/rbf_fin_op.cuh"  // rbf_fin_op
#include <cuvs/distance/distance.hpp>     // cuvs::distance::DistanceType
#include <raft/core/device_mdspan.hpp>    // raft::device_matrix_view
#include <raft/core/operators.hpp>        // raft::identity_op
#include <raft/core/resources.hpp>        // raft::resources
#include <raft/util/raft_explicit.hpp>    // RAFT_EXPLICIT

#include <rmm/device_uvector.hpp>  // rmm::device_uvector

#ifdef CUVS_EXPLICIT_INSTANTIATE_ONLY

namespace cuvs {
namespace distance {

template <cuvs::distance::DistanceType DistT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename FinalLambda,
          typename IdxT = int>
void distance(raft::resources const& handle,
              const DataT* x,
              const DataT* y,
              OutT* dist,
              IdxT m,
              IdxT n,
              IdxT k,
              void* workspace,
              size_t worksize,
              FinalLambda fin_op,
              bool isRowMajor  = true,
              DataT metric_arg = 2.0f) RAFT_EXPLICIT;

template <cuvs::distance::DistanceType DistT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT = int>
void distance(raft::resources const& handle,
              const DataT* x,
              const DataT* y,
              OutT* dist,
              IdxT m,
              IdxT n,
              IdxT k,
              void* workspace,
              size_t worksize,
              bool isRowMajor  = true,
              DataT metric_arg = 2.0f) RAFT_EXPLICIT;

template <cuvs::distance::DistanceType DistT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT = int>
size_t getWorkspaceSize(const DataT* x, const DataT* y, IdxT m, IdxT n, IdxT k) RAFT_EXPLICIT;

template <cuvs::distance::DistanceType DistT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT = int,
          typename layout>
size_t getWorkspaceSize(raft::device_matrix_view<DataT, IdxT, layout> const& x,
                        raft::device_matrix_view<DataT, IdxT, layout> const& y) RAFT_EXPLICIT;

template <cuvs::distance::DistanceType DistT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT = int>
void distance(raft::resources const& handle,
              const DataT* x,
              const DataT* y,
              OutT* dist,
              IdxT m,
              IdxT n,
              IdxT k,
              bool isRowMajor  = true,
              DataT metric_arg = 2.0f) RAFT_EXPLICIT;

template <typename Type, typename IdxT = int, typename DistT = Type>
void pairwise_distance(raft::resources const& handle,
                       const Type* x,
                       const Type* y,
                       DistT* dist,
                       IdxT m,
                       IdxT n,
                       IdxT k,
                       rmm::device_uvector<char>& workspace,
                       cuvs::distance::DistanceType metric,
                       bool isRowMajor  = true,
                       DistT metric_arg = DistT(2.0f)) RAFT_EXPLICIT;

template <typename Type, typename IdxT = int, typename DistT = Type>
void pairwise_distance(raft::resources const& handle,
                       const Type* x,
                       const Type* y,
                       DistT* dist,
                       IdxT m,
                       IdxT n,
                       IdxT k,
                       cuvs::distance::DistanceType metric,
                       bool isRowMajor  = true,
                       DistT metric_arg = DistT(2.0f)) RAFT_EXPLICIT;

template <cuvs::distance::DistanceType DistT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename layout = raft::layout_c_contiguous,
          typename IdxT   = int>
void distance(raft::resources const& handle,
              raft::device_matrix_view<DataT, IdxT, layout> const x,
              raft::device_matrix_view<DataT, IdxT, layout> const y,
              raft::device_matrix_view<OutT, IdxT, layout> dist,
              DataT metric_arg = 2.0f) RAFT_EXPLICIT;

template <typename Type,
          typename layout = layout_c_contiguous,
          typename IdxT   = int,
          typename DistT  = Type>
void pairwise_distance(raft::resources const& handle,
                       device_matrix_view<Type, IdxT, layout> const x,
                       device_matrix_view<Type, IdxT, layout> const y,
                       device_matrix_view<DistT, IdxT, layout> dist,
                       cuvs::distance::DistanceType metric,
                       DistT metric_arg = DistT(2.0f)) RAFT_EXPLICIT;

};  // namespace distance
};  // namespace cuvs

#endif  // RAFT_EXPLICIT_INSTANTIATE_ONLY

/*
 * Hierarchy of instantiations:
 *
 * This file defines the extern template instantiations for the public API of
 * cuvs::distance. To improve compile times, the extern template instantiation
 * of the distance kernels is handled in
 * distance/detail/pairwise_matrix/dispatch-ext.cuh.
 *
 * After adding an instance here, make sure to also add the instance to
 * dispatch-ext.cuh and the corresponding .cu files.
 */

#define instantiate_raft_distance_distance(DT, DataT, AccT, OutT, FinalLambda, IdxT)       \
  extern template void cuvs::distance::distance<DT, DataT, AccT, OutT, FinalLambda, IdxT>( \
    raft::resources const& handle,                                                         \
    const DataT* x,                                                                        \
    const DataT* y,                                                                        \
    OutT* dist,                                                                            \
    IdxT m,                                                                                \
    IdxT n,                                                                                \
    IdxT k,                                                                                \
    void* workspace,                                                                       \
    size_t worksize,                                                                       \
    FinalLambda fin_op,                                                                    \
    bool isRowMajor,                                                                       \
    DataT metric_arg)

// The following two instances are used in test/distance/gram.cu. Note the use
// of int64_t for the index type.
instantiate_raft_distance_distance(cuvs::distance::DistanceType::L2Unexpanded,
                                   float,
                                   float,
                                   float,
                                   cuvs::distance::kernels::detail::rbf_fin_op<float>,
                                   int64_t);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::L2Unexpanded,
                                   double,
                                   double,
                                   double,
                                   cuvs::distance::kernels::detail::rbf_fin_op<double>,
                                   int64_t);

instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::Canberra, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::Canberra, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::CorrelationExpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::CorrelationExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::identity_op,
                                   int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::CosineExpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::CosineExpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::HammingUnexpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::HammingUnexpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::HellingerExpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::HellingerExpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::InnerProduct, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::InnerProduct, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::JensenShannon, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::JensenShannon, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::KLDivergence, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::KLDivergence, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L1, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L1, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2Expanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2Expanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2SqrtExpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2SqrtExpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2SqrtUnexpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2SqrtUnexpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2Unexpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2Unexpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::Linf, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::Linf, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::LpUnexpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::LpUnexpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::RusselRaoExpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::RusselRaoExpanded, double, double, double, raft::identity_op, int);

#undef instantiate_raft_distance_distance

// Same, but without raft::identity_op
#define instantiate_raft_distance_distance(DT, DataT, AccT, OutT, IdxT)       \
  extern template void cuvs::distance::distance<DT, DataT, AccT, OutT, IdxT>( \
    raft::resources const& handle,                                            \
    const DataT* x,                                                           \
    const DataT* y,                                                           \
    OutT* dist,                                                               \
    IdxT m,                                                                   \
    IdxT n,                                                                   \
    IdxT k,                                                                   \
    void* workspace,                                                          \
    size_t worksize,                                                          \
    bool isRowMajor,                                                          \
    DataT metric_arg)

instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::Canberra, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::Canberra, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::CorrelationExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::CorrelationExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::CosineExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::CosineExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::HammingUnexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::HammingUnexpanded, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::HellingerExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::HellingerExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::InnerProduct, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::InnerProduct, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::JensenShannon, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::JensenShannon, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::KLDivergence, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::KLDivergence, double, double, double, int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::L1, float, float, float, int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::L1, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2Expanded, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2Expanded, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2SqrtExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2SqrtExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2SqrtUnexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2SqrtUnexpanded, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2Unexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2Unexpanded, double, double, double, int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::Linf, float, float, float, int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::Linf, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::LpUnexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::LpUnexpanded, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::RusselRaoExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::RusselRaoExpanded, double, double, double, int);

#undef instantiate_raft_distance_distance

// Same, but without workspace
#define instantiate_raft_distance_distance(DT, DataT, AccT, OutT, IdxT)       \
  extern template void cuvs::distance::distance<DT, DataT, AccT, OutT, IdxT>( \
    raft::resources const& handle,                                            \
    const DataT* x,                                                           \
    const DataT* y,                                                           \
    OutT* dist,                                                               \
    IdxT m,                                                                   \
    IdxT n,                                                                   \
    IdxT k,                                                                   \
    bool isRowMajor,                                                          \
    DataT metric_arg)

instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::Canberra, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::Canberra, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::CorrelationExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::CorrelationExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::CosineExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::CosineExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::HammingUnexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::HammingUnexpanded, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::HellingerExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::HellingerExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::InnerProduct, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::InnerProduct, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::JensenShannon, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::JensenShannon, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::KLDivergence, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::KLDivergence, double, double, double, int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::L1, float, float, float, int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::L1, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2Expanded, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2Expanded, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2SqrtExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2SqrtExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2SqrtUnexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2SqrtUnexpanded, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2Unexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2Unexpanded, double, double, double, int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::Linf, float, float, float, int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::Linf, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::LpUnexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::LpUnexpanded, double, double, double, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::RusselRaoExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::RusselRaoExpanded, double, double, double, int);

#undef instantiate_raft_distance_distance

#define instantiate_raft_distance_getWorkspaceSize(DistT, DataT, AccT, OutT, IdxT)         \
  extern template size_t cuvs::distance::getWorkspaceSize<DistT, DataT, AccT, OutT, IdxT>( \
    const DataT* x, const DataT* y, IdxT m, IdxT n, IdxT k)

instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::Canberra, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::Canberra, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::CorrelationExpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::CorrelationExpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::CosineExpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::CosineExpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::HammingUnexpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::HammingUnexpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::HellingerExpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::HellingerExpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::InnerProduct, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::InnerProduct, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::JensenShannon, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::JensenShannon, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::KLDivergence, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::KLDivergence, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::L1, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::L1, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::L2Expanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::L2Expanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::L2SqrtExpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::L2SqrtExpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::L2SqrtUnexpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::L2SqrtUnexpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::L2Unexpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::L2Unexpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::Linf, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::Linf, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::LpUnexpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::LpUnexpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::RusselRaoExpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::RusselRaoExpanded, double, double, double, int);

#undef instantiate_raft_distance_getWorkspaceSize

#define instantiate_raft_distance_getWorkspaceSize(DistT, DataT, AccT, OutT, IdxT, layout)         \
  extern template size_t cuvs::distance::getWorkspaceSize<DistT, DataT, AccT, OutT, IdxT, layout>( \
    raft::device_matrix_view<DataT, IdxT, layout> const& x,                                        \
    raft::device_matrix_view<DataT, IdxT, layout> const& y)

// We could consider not taking template parameters for this function. The
// number of instantiations seems a bit excessive..
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::Canberra, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::Canberra, double, double, double, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::Canberra, float, float, float, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::Canberra, double, double, double, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::CorrelationExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::CorrelationExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::CorrelationExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::CorrelationExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::CosineExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::CosineExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::CosineExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::CosineExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::HammingUnexpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::HammingUnexpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::HammingUnexpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::HammingUnexpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::HellingerExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::HellingerExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::HellingerExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::HellingerExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::InnerProduct, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::InnerProduct,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::InnerProduct, float, float, float, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::InnerProduct,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::JensenShannon, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::JensenShannon,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::JensenShannon, float, float, float, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::JensenShannon,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::KLDivergence, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::KLDivergence,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::KLDivergence, float, float, float, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::KLDivergence,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::L1, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::L1, double, double, double, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::L1, float, float, float, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::L1, double, double, double, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::L2Expanded, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::L2Expanded, double, double, double, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::L2Expanded, float, float, float, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::L2Expanded, double, double, double, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::L2SqrtExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::L2SqrtExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::L2SqrtExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::L2SqrtExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::L2SqrtUnexpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::L2SqrtUnexpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::L2SqrtUnexpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::L2SqrtUnexpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::L2Unexpanded, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(cuvs::distance::DistanceType::L2Unexpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  cuvs::distance::DistanceType::L2Unexpanded, float, float, float, int, raft::layout_f_contiguous);

#undef instantiate_raft_distance_getWorkspaceSize

#define instantiate_raft_distance_pairwise_distance(DataT, IdxT, DistT)                        \
  extern template void cuvs::distance::pairwise_distance(raft::resources const& handle,        \
                                                         const DataT* x,                       \
                                                         const DataT* y,                       \
                                                         DistT* dist,                          \
                                                         IdxT m,                               \
                                                         IdxT n,                               \
                                                         IdxT k,                               \
                                                         rmm::device_uvector<char>& workspace, \
                                                         cuvs::distance::DistanceType metric,  \
                                                         bool isRowMajor,                      \
                                                         DistT metric_arg)

instantiate_raft_distance_pairwise_distance(float, int, float);
instantiate_raft_distance_pairwise_distance(double, int, double);
instantiate_raft_distance_pairwise_distance(half, int, float);

#undef instantiate_raft_distance_pairwise_distance

// Same, but without workspace
#define instantiate_raft_distance_pairwise_distance(DataT, IdxT, DistT)                       \
  extern template void cuvs::distance::pairwise_distance(raft::resources const& handle,       \
                                                         const DataT* x,                      \
                                                         const DataT* y,                      \
                                                         DistT* dist,                         \
                                                         IdxT m,                              \
                                                         IdxT n,                              \
                                                         IdxT k,                              \
                                                         cuvs::distance::DistanceType metric, \
                                                         bool isRowMajor,                     \
                                                         DistT metric_arg)

instantiate_raft_distance_pairwise_distance(float, int, float);
instantiate_raft_distance_pairwise_distance(double, int, double);
instantiate_raft_distance_pairwise_distance(half, int, float);

#undef instantiate_raft_distance_pairwise_distance

// Version with mdspan
#define instantiate_raft_distance_distance(DistT, DataT, AccT, OutT, layout, IdxT)       \
  extern template void cuvs::distance::distance<DistT, DataT, AccT, OutT, layout, IdxT>( \
    raft::resources const& handle,                                                       \
    raft::device_matrix_view<const DataT, IdxT, layout> const x,                         \
    raft::device_matrix_view<const DataT, IdxT, layout> const y,                         \
    raft::device_matrix_view<OutT, IdxT, layout> dist,                                   \
    DataT metric_arg)

// Again, we might want to consider reigning in the number of instantiations...
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::Canberra, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::Canberra, double, double, double, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::Canberra, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::Canberra, double, double, double, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::CorrelationExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::CorrelationExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::CorrelationExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::CorrelationExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::CosineExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::CosineExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::CosineExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::CosineExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::HammingUnexpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::HammingUnexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::HammingUnexpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::HammingUnexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::HellingerExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::HellingerExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::HellingerExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::HellingerExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::InnerProduct, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::InnerProduct,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::InnerProduct, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::InnerProduct,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::JensenShannon, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::JensenShannon,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::JensenShannon, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::JensenShannon,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::KLDivergence, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::KLDivergence,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::KLDivergence, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::KLDivergence,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L1, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L1, double, double, double, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L1, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L1, double, double, double, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2Expanded, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2Expanded, double, double, double, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2Expanded, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2Expanded, double, double, double, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::L2SqrtExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::L2SqrtExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::L2SqrtExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::L2SqrtExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::L2SqrtUnexpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::L2SqrtUnexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::L2SqrtUnexpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::L2SqrtUnexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2Unexpanded, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::L2Unexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::L2Unexpanded, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::L2Unexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::Linf, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::Linf, double, double, double, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::Linf, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::Linf, double, double, double, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::LpUnexpanded, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::LpUnexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(
  cuvs::distance::DistanceType::LpUnexpanded, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::LpUnexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::RusselRaoExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::RusselRaoExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::RusselRaoExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(cuvs::distance::DistanceType::RusselRaoExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);

#undef instantiate_raft_distance_distance

#define instantiate_raft_distance_pairwise_distance(DataT, layout, IdxT, DistT) \
  extern template void cuvs::distance::pairwise_distance(                       \
    raft::resources const& handle,                                              \
    raft::device_matrix_view<const DataT, IdxT, layout> const x,                \
    raft::device_matrix_view<const DataT, IdxT, layout> const y,                \
    raft::device_matrix_view<DistT, IdxT, layout> dist,                         \
    cuvs::distance::DistanceType metric,                                        \
    DistT metric_arg)

instantiate_raft_distance_pairwise_distance(float, raft::layout_c_contiguous, int, float);
instantiate_raft_distance_pairwise_distance(float, raft::layout_f_contiguous, int, float);
instantiate_raft_distance_pairwise_distance(double, raft::layout_c_contiguous, int, double);
instantiate_raft_distance_pairwise_distance(double, raft::layout_f_contiguous, int, double);
instantiate_raft_distance_pairwise_distance(half, raft::layout_c_contiguous, int, float);
instantiate_raft_distance_pairwise_distance(half, raft::layout_f_contiguous, int, float);

#undef instantiate_raft_distance_pairwise_distance

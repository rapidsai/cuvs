/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/linalg/add.cuh>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/init.cuh>

namespace cuvs::spatial::knn::detail::utils {
template <typename T>
struct mapping;
}

namespace cuvs::detail {

// Named operations share launchers; caller-specific device lambdas retain their dispatch.
template <typename Op>
struct shared_map_op : std::false_type {};
template <>
struct shared_map_op<raft::identity_op> : std::true_type {};
template <>
struct shared_map_op<raft::sqrt_op> : std::true_type {};
template <>
struct shared_map_op<raft::abs_op> : std::true_type {};
template <>
struct shared_map_op<raft::sq_op> : std::true_type {};
template <>
struct shared_map_op<raft::add_op> : std::true_type {};
template <>
struct shared_map_op<raft::sub_op> : std::true_type {};
template <>
struct shared_map_op<raft::mul_op> : std::true_type {};
template <>
struct shared_map_op<raft::div_op> : std::true_type {};
template <>
struct shared_map_op<raft::div_checkzero_op> : std::true_type {};
template <>
struct shared_map_op<raft::pow_op> : std::true_type {};
template <>
struct shared_map_op<raft::mod_op> : std::true_type {};
template <>
struct shared_map_op<raft::min_op> : std::true_type {};
template <>
struct shared_map_op<raft::max_op> : std::true_type {};
template <typename T>
struct shared_map_op<raft::cast_op<T>> : std::true_type {};
template <typename T>
struct shared_map_op<raft::const_op<T>> : std::true_type {};
template <typename T>
struct shared_map_op<cuvs::spatial::knn::detail::utils::mapping<T>> : std::true_type {};
template <typename T, typename Op>
struct shared_map_op<raft::plug_const_op<T, Op>> : shared_map_op<Op> {};
template <typename... Ops>
struct shared_map_op<raft::compose_op<Ops...>>
  : std::bool_constant<(shared_map_op<Ops>::value && ...)> {};

template <bool PassOffset, typename OutT, typename IdxT, typename Func, typename... InTs>
void map_impl(cuda::stream_ref stream, OutT* out_ptr, IdxT len, Func f, const InTs*... in_ptrs);

template <bool PassOffset, typename OutT, typename IdxT, typename Func, typename... InTs>
void dispatch_map(cuda::stream_ref stream, OutT* out_ptr, IdxT len, Func f, const InTs*... in_ptrs)
{
  if constexpr (shared_map_op<Func>::value) {
    map_impl<PassOffset>(stream, out_ptr, len, f, in_ptrs...);
  } else {
    raft::linalg::detail::map<PassOffset>(stream, out_ptr, len, f, in_ptrs...);
  }
}

template <bool PassOffset,
          typename OutType,
          typename Func,
          typename... InTypes,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InTypes...>>
void dispatch_map(const raft::resources& res, OutType out, Func f, InTypes... ins)
{
  if (raft::resource::get_dry_run_flag(res)) { return; }
  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");
  (raft::linalg::detail::map_check_shape(out, ins), ...);

  dispatch_map<PassOffset,
               typename OutType::value_type,
               typename OutType::index_type,
               Func,
               typename InTypes::value_type...>(
    raft::resource::get_cuda_stream(res),
    out.data_handle(),
    static_cast<typename OutType::index_type>(out.size()),
    f,
    ins.data_handle()...);
}

template <typename OutType,
          typename Func,
          typename... InTypes,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InTypes...>>
void map(const raft::resources& res, OutType out, Func f, InTypes... ins)
{
  return dispatch_map<false>(res, out, f, ins...);
}

template <typename InType1,
          typename OutType,
          typename Func,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InType1>>
void map(const raft::resources& res, InType1 in1, OutType out, Func f)
{
  return dispatch_map<false>(res, out, f, in1);
}

template <typename InType1,
          typename InType2,
          typename OutType,
          typename Func,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InType1, InType2>>
void map(const raft::resources& res, InType1 in1, InType2 in2, OutType out, Func f)
{
  return dispatch_map<false>(res, out, f, in1, in2);
}

template <typename InType1,
          typename InType2,
          typename InType3,
          typename OutType,
          typename Func,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InType1, InType2, InType3>>
void map(const raft::resources& res, InType1 in1, InType2 in2, InType3 in3, OutType out, Func f)
{
  return dispatch_map<false>(res, out, f, in1, in2, in3);
}

template <typename OutType,
          typename Func,
          typename... InTypes,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InTypes...>>
void map_offset(const raft::resources& res, OutType out, Func f, InTypes... ins)
{
  return dispatch_map<true>(res, out, f, ins...);
}

template <typename InType1,
          typename OutType,
          typename Func,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InType1>>
void map_offset(const raft::resources& res, InType1 in1, OutType out, Func f)
{
  return dispatch_map<true>(res, out, f, in1);
}

template <typename InType1,
          typename InType2,
          typename OutType,
          typename Func,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InType1, InType2>>
void map_offset(const raft::resources& res, InType1 in1, InType2 in2, OutType out, Func f)
{
  return dispatch_map<true>(res, out, f, in1, in2);
}

template <typename InType1,
          typename InType2,
          typename InType3,
          typename OutType,
          typename Func,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InType1, InType2, InType3>>
void map_offset(
  const raft::resources& res, InType1 in1, InType2 in2, InType3 in3, OutType out, Func f)
{
  return dispatch_map<true>(res, out, f, in1, in2, in3);
}

template <typename InType,
          typename Lambda,
          typename IdxType = int,
          typename OutType = InType,
          int TPB          = 256>
void unaryOp(OutType* out, const InType* in, IdxType len, Lambda op, cudaStream_t stream)
{
  return dispatch_map<false>(stream, out, len, op, in);
}

template <typename InType,
          typename Lambda,
          typename OutType = InType,
          typename IdxType = int,
          int TPB          = 256>
void binaryOp(
  OutType* out, const InType* in1, const InType* in2, IdxType len, Lambda op, cudaStream_t stream)
{
  return dispatch_map<false>(stream, out, len, op, in1, in2);
}

template <typename math_t, typename extents, typename layout>
void fill(raft::resources const& handle,
          raft::device_mdspan<const math_t, extents, layout> in,
          raft::device_mdspan<math_t, extents, layout> out,
          raft::host_scalar_view<math_t> scalar)
{
  if (raft::resource::get_dry_run_flag(handle)) { return; }
  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Data layout not supported");
  RAFT_EXPECTS(in.size() == out.size(), "Input and output matrices must be the same size.");
  RAFT_EXPECTS(scalar.data_handle() != nullptr, "Empty scalar");
  raft::matrix::detail::setValue(out.data_handle(),
                                 in.data_handle(),
                                 *(scalar.data_handle()),
                                 in.size(),
                                 raft::resource::get_cuda_stream(handle).get());
}

template <typename math_t, typename extents, typename layout>
void fill(raft::resources const& handle,
          raft::device_mdspan<math_t, extents, layout> inout,
          math_t scalar)
{
  if (raft::resource::get_dry_run_flag(handle)) { return; }
  cuvs::detail::map(handle, inout, raft::const_op{scalar});
}

template <typename InT, typename OutT = InT, typename IdxType = int>
void dispatch_addScalar(OutT* out, const InT* in, InT scalar, IdxType len, cudaStream_t stream)
{
  cuvs::detail::unaryOp(out, in, len, raft::add_const_op<InT>(scalar), stream);
}

template <typename InT, typename OutT = InT, typename IdxType = int>
void dispatch_add(OutT* out, const InT* in1, const InT* in2, IdxType len, cudaStream_t stream)
{
  cuvs::detail::binaryOp(out, in1, in2, len, raft::add_op(), stream);
}

template <typename InT, typename OutT = InT, typename IdxType = int>
void addScalar(OutT* out, const InT* in, const InT scalar, IdxType len, cudaStream_t stream)
{
  dispatch_addScalar(out, in, scalar, len, stream);
}

template <typename InT, typename OutT = InT, typename IdxType = int>
void add(OutT* out, const InT* in1, const InT* in2, IdxType len, cudaStream_t stream)
{
  dispatch_add(out, in1, in2, len, stream);
}

template <typename InType,
          typename OutType,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void add(raft::resources const& handle, InType in1, InType in2, OutType out)
{
  if (raft::resource::get_dry_run_flag(handle)) { return; }
  using in_value_t  = typename InType::value_type;
  using out_value_t = typename OutType::value_type;

  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(in1), "Input 1 must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(in2), "Input 2 must be contiguous");
  RAFT_EXPECTS(out.size() == in1.size() && in1.size() == in2.size(),
               "Size mismatch between Output and Inputs");

  add<in_value_t, out_value_t, typename OutType::index_type>(
    out.data_handle(),
    in1.data_handle(),
    in2.data_handle(),
    static_cast<typename OutType::index_type>(out.size()),
    raft::resource::get_cuda_stream(handle).get());
}

template <typename InT, typename OutT = InT, typename IdxType = int>
void addDevScalar(
  OutT* outDev, const InT* inDev, const InT* singleScalarDev, IdxType len, cudaStream_t stream)
{
  raft::linalg::detail::addDevScalar(outDev, inDev, singleScalarDev, len, stream);
}

template <typename InType,
          typename OutType,
          typename ScalarIdxType,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void add_scalar(raft::resources const& handle,
                InType in,
                OutType out,
                raft::device_scalar_view<const typename InType::value_type, ScalarIdxType> scalar)
{
  if (raft::resource::get_dry_run_flag(handle)) { return; }
  using in_value_t  = typename InType::value_type;
  using out_value_t = typename OutType::value_type;

  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(in), "Input must be contiguous");
  RAFT_EXPECTS(out.size() == in.size(), "Size mismatch between Output and Input");

  addDevScalar<in_value_t, out_value_t, typename OutType::index_type>(
    out.data_handle(),
    in.data_handle(),
    scalar.data_handle(),
    static_cast<typename OutType::index_type>(out.size()),
    raft::resource::get_cuda_stream(handle));
}

template <typename InType,
          typename OutType,
          typename ScalarIdxType,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void add_scalar(raft::resources const& handle,
                const InType in,
                OutType out,
                raft::host_scalar_view<const typename InType::value_type, ScalarIdxType> scalar)
{
  if (raft::resource::get_dry_run_flag(handle)) { return; }
  using in_value_t  = typename InType::value_type;
  using out_value_t = typename OutType::value_type;

  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(in), "Input must be contiguous");
  RAFT_EXPECTS(out.size() == in.size(), "Size mismatch between Output and Input");

  addScalar<in_value_t, out_value_t, typename OutType::index_type>(
    out.data_handle(),
    in.data_handle(),
    *scalar.data_handle(),
    static_cast<typename OutType::index_type>(out.size()),
    raft::resource::get_cuda_stream(handle));
}

}  // namespace cuvs::detail

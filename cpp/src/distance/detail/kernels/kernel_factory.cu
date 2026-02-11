/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/distance/grammian.hpp>

namespace cuvs::distance::kernels {

template <typename MathT>
auto KernelFactory<MathT>::create(KernelParams params) -> GramMatrixBase<MathT>*
{
  GramMatrixBase<MathT>* res;
  // KernelParams is not templated, we convert the parameters to MathT here:
  MathT coef0 = params.coef0;
  MathT gamma = params.gamma;
  switch (params.kernel) {
    case LINEAR: res = new GramMatrixBase<MathT>(); break;
    case POLYNOMIAL: res = new PolynomialKernel<MathT, int>(params.degree, gamma, coef0); break;
    case TANH: res = new TanhKernel<MathT>(gamma, coef0); break;
    case RBF: res = new RBFKernel<MathT>(gamma); break;
    default: throw raft::exception("Kernel not implemented");
  }
  return res;
}

template <typename MathT>
[[deprecated]] auto KernelFactory<MathT>::create(KernelParams params, cublasHandle_t handle)
  -> GramMatrixBase<MathT>*
{
  GramMatrixBase<MathT>* res;
  // KernelParams is not templated, we convert the parameters to MathT here:
  MathT coef0 = params.coef0;
  MathT gamma = params.gamma;
  switch (params.kernel) {
    case LINEAR: res = new GramMatrixBase<MathT>(handle); break;
    case POLYNOMIAL:
      res = new PolynomialKernel<MathT, int>(params.degree, gamma, coef0, handle);
      break;
    case TANH: res = new TanhKernel<MathT>(gamma, coef0, handle); break;
    case RBF: res = new RBFKernel<MathT>(gamma, handle); break;
    default: throw raft::exception("Kernel not implemented");
  }
  return res;
}

template class KernelFactory<float>;
template class KernelFactory<double>;

};  // end namespace cuvs::distance::kernels

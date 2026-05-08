---
slug: api-reference/cpp-api-distance-grammian
---

# Grammian

_Source header: `cuvs/distance/grammian.hpp`_

## Types

<a id="cuvs-distance-kernels-grammatrixbase"></a>
### cuvs::distance::kernels::GramMatrixBase

Base class for general Gram matrices

A Gram matrix is the Hermitian matrix of inner probucts G_ik = &lt;x_i, x_k&gt; Here, the  inner product is evaluated for all elements from vectors sets X1, and X2.

To be more precise, on exit the output buffer will store:

- if is_row_major == true: out[j+k*n1] = &lt;x1_j, x2_k&gt;,
- if is_row_major == false: out[j*n2 + k] = &lt;x1_j, x2_k&gt;, where x1_j is the j-th vector from the x1 set and x2_k is the k-th vector from the x2 set.

```cpp
template <typename math_t>
class GramMatrixBase { ... };
```

<a id="cuvs-distance-kernels-polynomialkernel"></a>
### cuvs::distance::kernels::PolynomialKernel

Create a kernel matrix using polynomial kernel function.

```cpp
template <typename math_t, typename exp_t>
class PolynomialKernel : public GramMatrixBase<math_t> { ... };
```

<a id="cuvs-distance-kernels-tanhkernel"></a>
### cuvs::distance::kernels::TanhKernel

Create a kernel matrix using tanh kernel function.

```cpp
template <typename math_t>
class TanhKernel : public GramMatrixBase<math_t> { ... };
```

<a id="cuvs-distance-kernels-rbfkernel"></a>
### cuvs::distance::kernels::RBFKernel

Create a kernel matrix using RBF kernel function.

```cpp
template <typename math_t>
class RBFKernel : public GramMatrixBase<math_t> { ... };
```

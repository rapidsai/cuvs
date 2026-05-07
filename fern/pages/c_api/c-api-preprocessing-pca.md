---
slug: api-reference/c-api-preprocessing-pca
---

# PCA

_Source header: `c/include/cuvs/preprocessing/pca.h`_

## C API for PCA (Principal Component Analysis)

_Doxygen group: `preprocessing_c_pca`_

### cuvsPcaSolver

Solver algorithm for PCA eigen decomposition.

```c
enum cuvsPcaSolver { ... } ;
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_PCA_COV_EIG_DQ` | `0` |
| `CUVS_PCA_COV_EIG_JACOBI` | `1` |

_Source: `c/include/cuvs/preprocessing/pca.h:25`_

### cuvsPcaParams

Parameters for PCA decomposition.

```c
struct cuvsPcaParams { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_components` | `int` | Number of principal components to keep. |
| `copy` | `bool` | If false, data passed to fit are overwritten and running fit(X).transform(X) will |
| `whiten` | `bool` | When true the component vectors are multiplied by the square root of n_samples and then |
| `algorithm` | `enum cuvsPcaSolver` | Solver algorithm to use. |
| `tol` | `float` | Tolerance for singular values (used by Jacobi solver). |
| `n_iterations` | `int` | Number of iterations for the power method (Jacobi solver). |

_Source: `c/include/cuvs/preprocessing/pca.h:35`_

### cuvsPcaParamsCreate

Allocate PCA params and populate with default values.

```c
cuvsError_t cuvsPcaParamsCreate(cuvsPcaParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | out | `cuvsPcaParams_t*` | cuvsPcaParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/preprocessing/pca.h:70`_

### cuvsPcaParamsDestroy

De-allocate PCA params.

```c
cuvsError_t cuvsPcaParamsDestroy(cuvsPcaParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsPcaParams_t` | cuvsPcaParams_t to de-allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/preprocessing/pca.h:78`_

### cuvsPcaFit

Perform PCA fit operation.

```c
cuvsError_t cuvsPcaFit(cuvsResources_t res,
cuvsPcaParams_t params,
DLManagedTensor* input,
DLManagedTensor* components,
DLManagedTensor* explained_var,
DLManagedTensor* explained_var_ratio,
DLManagedTensor* singular_vals,
DLManagedTensor* mu,
DLManagedTensor* noise_vars,
bool flip_signs_based_on_U);
```

Computes the principal components, explained variances, singular values, and column means from the input data.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `params` | in | `cuvsPcaParams_t` | PCA parameters |
| `input` | inout | `DLManagedTensor*` | input data [n_rows x n_cols] (col-major, float32, device) |
| `components` | out | `DLManagedTensor*` | principal components [n_components x n_cols] (col-major, float32, device) |
| `explained_var` | out | `DLManagedTensor*` | explained variances [n_components] (float32, device) |
| `explained_var_ratio` | out | `DLManagedTensor*` | explained variance ratios [n_components] (float32, device) |
| `singular_vals` | out | `DLManagedTensor*` | singular values [n_components] (float32, device) |
| `mu` | out | `DLManagedTensor*` | column means [n_cols] (float32, device) |
| `noise_vars` | out | `DLManagedTensor*` | noise variance [1] (float32, device) |
| `flip_signs_based_on_U` | in | `bool` | whether to determine signs by U (true) or V.T (false) |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/preprocessing/pca.h:128`_

### cuvsPcaFitTransform

Perform PCA fit and transform in a single operation.

```c
cuvsError_t cuvsPcaFitTransform(cuvsResources_t res,
cuvsPcaParams_t params,
DLManagedTensor* input,
DLManagedTensor* trans_input,
DLManagedTensor* components,
DLManagedTensor* explained_var,
DLManagedTensor* explained_var_ratio,
DLManagedTensor* singular_vals,
DLManagedTensor* mu,
DLManagedTensor* noise_vars,
bool flip_signs_based_on_U);
```

Computes the principal components and transforms the input data into the eigenspace.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `params` | in | `cuvsPcaParams_t` | PCA parameters |
| `input` | inout | `DLManagedTensor*` | input data [n_rows x n_cols] (col-major, float32, device) |
| `trans_input` | out | `DLManagedTensor*` | transformed data [n_rows x n_components] (col-major, float32, device) |
| `components` | out | `DLManagedTensor*` | principal components [n_components x n_cols] (col-major, float32, device) |
| `explained_var` | out | `DLManagedTensor*` | explained variances [n_components] (float32, device) |
| `explained_var_ratio` | out | `DLManagedTensor*` | explained variance ratios [n_components] (float32, device) |
| `singular_vals` | out | `DLManagedTensor*` | singular values [n_components] (float32, device) |
| `mu` | out | `DLManagedTensor*` | column means [n_cols] (float32, device) |
| `noise_vars` | out | `DLManagedTensor*` | noise variance [1] (float32, device) |
| `flip_signs_based_on_U` | in | `bool` | whether to determine signs by U (true) or V.T (false) |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/preprocessing/pca.h:157`_

### cuvsPcaTransform

Perform PCA transform operation.

```c
cuvsError_t cuvsPcaTransform(cuvsResources_t res,
cuvsPcaParams_t params,
DLManagedTensor* input,
DLManagedTensor* components,
DLManagedTensor* singular_vals,
DLManagedTensor* mu,
DLManagedTensor* trans_input);
```

Transforms the input data into the eigenspace using previously computed principal components.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `params` | in | `cuvsPcaParams_t` | PCA parameters |
| `input` | inout | `DLManagedTensor*` | data to transform [n_rows x n_cols] (col-major, float32, device) |
| `components` | in | `DLManagedTensor*` | principal components [n_components x n_cols] (col-major, float32, device) |
| `singular_vals` | in | `DLManagedTensor*` | singular values [n_components] (float32, device) |
| `mu` | in | `DLManagedTensor*` | column means [n_cols] (float32, device) |
| `trans_input` | out | `DLManagedTensor*` | transformed data [n_rows x n_components] (col-major, float32, device) |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/preprocessing/pca.h:183`_

### cuvsPcaInverseTransform

Perform PCA inverse transform operation.

```c
cuvsError_t cuvsPcaInverseTransform(cuvsResources_t res,
cuvsPcaParams_t params,
DLManagedTensor* trans_input,
DLManagedTensor* components,
DLManagedTensor* singular_vals,
DLManagedTensor* mu,
DLManagedTensor* output);
```

Transforms data from the eigenspace back to the original space.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `params` | in | `cuvsPcaParams_t` | PCA parameters |
| `trans_input` | in | `DLManagedTensor*` | transformed data [n_rows x n_components] (col-major, float32, device) |
| `components` | in | `DLManagedTensor*` | principal components [n_components x n_cols] (col-major, float32, device) |
| `singular_vals` | in | `DLManagedTensor*` | singular values [n_components] (float32, device) |
| `mu` | in | `DLManagedTensor*` | column means [n_cols] (float32, device) |
| `output` | out | `DLManagedTensor*` | reconstructed data [n_rows x n_cols] (col-major, float32, device) |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/preprocessing/pca.h:205`_

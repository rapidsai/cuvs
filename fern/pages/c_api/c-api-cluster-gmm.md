---
slug: api-reference/c-api-cluster-gmm
---

# Gmm

_Source header: `cuvs/cluster/gmm.h`_

## Gaussian mixture hyperparameters

<a id="cuvsgmmcovariancetype"></a>
### cuvsGMMCovarianceType

Covariance parameterization of the mixture components.

```c
typedef enum {
  CUVS_GMM_COVARIANCE_FULL = 0,
  CUVS_GMM_COVARIANCE_TIED = 1,
  CUVS_GMM_COVARIANCE_DIAG = 2,
  CUVS_GMM_COVARIANCE_SPHERICAL = 3
} cuvsGMMCovarianceType;
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_GMM_COVARIANCE_FULL` | `0` |
| `CUVS_GMM_COVARIANCE_TIED` | `1` |
| `CUVS_GMM_COVARIANCE_DIAG` | `2` |
| `CUVS_GMM_COVARIANCE_SPHERICAL` | `3` |

<a id="cuvsgmminitmethod"></a>
### cuvsGMMInitMethod

Strategy used to initialize the responsibilities before EM.

```c
typedef enum {
  CUVS_GMM_INIT_KMEANS = 0,
  CUVS_GMM_INIT_KMEANS_PLUS_PLUS = 1,
  CUVS_GMM_INIT_RANDOM = 2,
  CUVS_GMM_INIT_RANDOM_FROM_DATA = 3
} cuvsGMMInitMethod;
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_GMM_INIT_KMEANS` | `0` |
| `CUVS_GMM_INIT_KMEANS_PLUS_PLUS` | `1` |
| `CUVS_GMM_INIT_RANDOM` | `2` |
| `CUVS_GMM_INIT_RANDOM_FROM_DATA` | `3` |

<a id="cuvsgmmparams"></a>
### cuvsGMMParams

Hyper-parameters for the Gaussian mixture EM solver

```c
struct cuvsGMMParams {
  int n_components;
  cuvsGMMCovarianceType covariance_type;
  double tol;
  double reg_covar;
  int max_iter;
  int n_init;
  cuvsGMMInitMethod init;
  uint64_t seed;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_components` | `int` | The number of mixture components. Default: 1. |
| `covariance_type` | [`cuvsGMMCovarianceType`](/api-reference/c-api-cluster-gmm#cuvsgmmcovariancetype) | Covariance parameterization of the mixture components. Default: FULL. |
| `tol` | `double` | Convergence threshold on the change of the per-sample average log-likelihood (lower bound). Default: 1e-3. |
| `reg_covar` | `double` | Non-negative regularization added to the diagonal of covariance.<br />Default: 1e-6. |
| `max_iter` | `int` | Maximum number of EM iterations for a single run. Default: 100. |
| `n_init` | `int` | Number of initializations to perform; the best result is kept. Default: 1. |
| `init` | [`cuvsGMMInitMethod`](/api-reference/c-api-cluster-gmm#cuvsgmminitmethod) | Strategy used to initialize the responsibilities before EM.<br />Default: KMEANS. |
| `seed` | `uint64_t` | Seed to the random number generator. Default: 0. |

<a id="cuvsgmmparamscreate"></a>
### cuvsGMMParamsCreate

Allocate GMM params, and populate with default values

```c
cuvsError_t cuvsGMMParamsCreate(cuvsGMMParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsGMMParams_t*`](/api-reference/c-api-cluster-gmm#cuvsgmmparams) | cuvsGMMParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsgmmparamsdestroy"></a>
### cuvsGMMParamsDestroy

De-allocate GMM params

```c
cuvsError_t cuvsGMMParamsDestroy(cuvsGMMParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsGMMParams_t`](/api-reference/c-api-cluster-gmm#cuvsgmmparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Gaussian mixture model APIs

<a id="cuvsgmmfit"></a>
### cuvsGMMFit

Fit a Gaussian mixture with the EM algorithm.

```c
cuvsError_t cuvsGMMFit(cuvsResources_t res,
cuvsGMMParams_t params,
DLManagedTensor* X,
DLManagedTensor* weights,
DLManagedTensor* means,
DLManagedTensor* covariances,
DLManagedTensor* precisions_chol,
DLManagedTensor* precisions,
DLManagedTensor* labels,
double* lower_bound,
int* n_iter,
bool* converged,
bool warm_start);
```

Runs ``params-&gt;n_init`` random restarts (unless ``warm_start`` is true) and keeps the parameters with the largest lower bound.

All tensors must reside on device memory and be row-major. ``X``, ``weights``, ``means``, ``covariances``, ``precisions_chol`` and ``precisions`` must share one dtype (float32 or float64); ``labels`` is int32.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | opaque C handle |
| `params` | in | [`cuvsGMMParams_t`](/api-reference/c-api-cluster-gmm#cuvsgmmparams) | Parameters for the GMM model. |
| `X` | in | `DLManagedTensor*` | Training data. [dim = n_samples x n_features] |
| `weights` | inout | `DLManagedTensor*` | Mixture weights. [len = n_components] |
| `means` | inout | `DLManagedTensor*` | Component means. [dim = n_components x n_features] |
| `covariances` | inout | `DLManagedTensor*` | Component covariances, flat. Length by covariance_type (K=n_components, d=n_features): FULL K*d*d, TIED d*d, DIAG K*d, SPHERICAL K. |
| `precisions_chol` | out | `DLManagedTensor*` | Precision Cholesky factors, same flat layout as covariances (FULL/TIED: upper-triangular factor U with precision = U @ Uᵀ; DIAG/SPHERICAL: reciprocal standard deviations). |
| `precisions` | out | `DLManagedTensor*` | Precision matrices, same flat layout as covariances. |
| `labels` | out | `DLManagedTensor*` | Hard component assignment per sample. [len = n_samples] |
| `lower_bound` | out | `double*` | Per-sample average log-likelihood of the best fit. |
| `n_iter` | out | `int*` | Number of EM iterations of the best fit. |
| `converged` | out | `bool*` | Whether the best fit converged within tol. |
| `warm_start` | in | `bool` | Use the incoming weights/means/covariances as the single initialization. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsgmmpredict"></a>
### cuvsGMMPredict

Hard component labels (argmax responsibility) for new data.

```c
cuvsError_t cuvsGMMPredict(cuvsResources_t res,
cuvsGMMParams_t params,
DLManagedTensor* X,
DLManagedTensor* weights,
DLManagedTensor* means,
DLManagedTensor* precisions_chol,
DLManagedTensor* labels);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | opaque C handle |
| `params` | in | [`cuvsGMMParams_t`](/api-reference/c-api-cluster-gmm#cuvsgmmparams) | Parameters used to fit the GMM model. |
| `X` | in | `DLManagedTensor*` | Data to assign. [dim = n_samples x n_features] |
| `weights` | in | `DLManagedTensor*` | Fitted mixture weights. [len = n_components] |
| `means` | in | `DLManagedTensor*` | Fitted component means. [dim = n_components x n_features] |
| `precisions_chol` | in | `DLManagedTensor*` | Fitted precision Cholesky factors, flat. Length by covariance_type (K=n_components, d=n_features): FULL K*d*d, TIED d*d, DIAG K*d, SPHERICAL K. |
| `labels` | out | `DLManagedTensor*` | Hard component assignment per sample (int32). [len = n_samples] |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsgmmpredictproba"></a>
### cuvsGMMPredictProba

Posterior responsibilities for new data.

```c
cuvsError_t cuvsGMMPredictProba(cuvsResources_t res,
cuvsGMMParams_t params,
DLManagedTensor* X,
DLManagedTensor* weights,
DLManagedTensor* means,
DLManagedTensor* precisions_chol,
DLManagedTensor* resp);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | opaque C handle |
| `params` | in | [`cuvsGMMParams_t`](/api-reference/c-api-cluster-gmm#cuvsgmmparams) | Parameters used to fit the GMM model. |
| `X` | in | `DLManagedTensor*` | Data to evaluate. [dim = n_samples x n_features] |
| `weights` | in | `DLManagedTensor*` | Fitted mixture weights. [len = n_components] |
| `means` | in | `DLManagedTensor*` | Fitted component means. [dim = n_components x n_features] |
| `precisions_chol` | in | `DLManagedTensor*` | Fitted precision Cholesky factors, flat. Length by covariance_type (K=n_components, d=n_features): FULL K*d*d, TIED d*d, DIAG K*d, SPHERICAL K. |
| `resp` | out | `DLManagedTensor*` | Posterior probability of each component for each sample. [dim = n_samples x n_components] |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsgmmscoresamples"></a>
### cuvsGMMScoreSamples

Per-sample log-likelihood log p(x_i) for new data.

```c
cuvsError_t cuvsGMMScoreSamples(cuvsResources_t res,
cuvsGMMParams_t params,
DLManagedTensor* X,
DLManagedTensor* weights,
DLManagedTensor* means,
DLManagedTensor* precisions_chol,
DLManagedTensor* log_prob_norm);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | opaque C handle |
| `params` | in | [`cuvsGMMParams_t`](/api-reference/c-api-cluster-gmm#cuvsgmmparams) | Parameters used to fit the GMM model. |
| `X` | in | `DLManagedTensor*` | Data to evaluate. [dim = n_samples x n_features] |
| `weights` | in | `DLManagedTensor*` | Fitted mixture weights. [len = n_components] |
| `means` | in | `DLManagedTensor*` | Fitted component means. [dim = n_components x n_features] |
| `precisions_chol` | in | `DLManagedTensor*` | Fitted precision Cholesky factors, flat. Length by covariance_type (K=n_components, d=n_features): FULL K*d*d, TIED d*d, DIAG K*d, SPHERICAL K. |
| `log_prob_norm` | out | `DLManagedTensor*` | Log-likelihood of each sample under the model. [len = n_samples] |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

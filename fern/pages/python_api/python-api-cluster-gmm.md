---
slug: api-reference/python-api-cluster-gmm
---

# Gmm

_Python module: `cuvs.cluster.gmm`_

## GMMParams

```python
cdef class GMMParams
```

Hyper-parameters for the Gaussian mixture EM solver

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `n_components` | `int` | The number of mixture components. |
| `covariance_type` | `str` | Covariance parameterization, one of "full", "tied", "diag", "spherical". Matches scikit-learn's ``GaussianMixture``. |
| `tol` | `float` | Convergence threshold on the change of the per-sample average log-likelihood (lower bound). |
| `reg_covar` | `float` | Non-negative regularization added to the diagonal of covariance. |
| `max_iter` | `int` | Maximum number of EM iterations for a single run. |
| `n_init` | `int` | Number of initializations to perform; the best result is kept. |
| `init_method` | `str` | Strategy used to initialize the responsibilities before EM. One of: "kmeans" : run k-means and use the hard labels "k-means++" : use the k-means++ seeding labels "random" : random responsibilities, normalized per sample "random_from_data" : pick n_components samples as one-hot responsibilities |
| `seed` | `int` | Seed to the random number generator. |

**Constructor**

```python
def __init__(self, *, n_components=None, covariance_type=None, tol=None, reg_covar=None, max_iter=None, n_init=None, init_method=None, seed=None)
```

**Members**

| Name | Kind |
| --- | --- |
| `n_components` | property |
| `covariance_type` | property |
| `tol` | property |
| `reg_covar` | property |
| `max_iter` | property |
| `n_init` | property |
| `init_method` | property |
| `seed` | property |

### n_components

```python
def n_components(self)
```

### covariance_type

```python
def covariance_type(self)
```

### tol

```python
def tol(self)
```

### reg_covar

```python
def reg_covar(self)
```

### max_iter

```python
def max_iter(self)
```

### n_init

```python
def n_init(self)
```

### init_method

```python
def init_method(self)
```

### seed

```python
def seed(self)
```

## fit

`@auto_sync_resources`
`@auto_convert_output`

```python
def fit(GMMParams params, X, weights=None, means=None, covariances=None, warm_start=False, resources=None)
```

Fit a Gaussian mixture model with the EM algorithm

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `params` | `GMMParams` | Parameters of the EM solver. |
| `X` | `Input CUDA array interface compliant matrix shape (m, k)` |  |
| `weights` | `Optional writable CUDA array interface vector, shape (n_components,). Holds the initial mixture weights when` | ``warm_start`` is True and receives the fitted weights. |
| `means` | `Optional writable CUDA array interface matrix, shape (n_components, k). Holds the initial means when` | ``warm_start`` is True and receives the fitted means. |
| `covariances` | `Optional writable CUDA array interface array whose shape` | depends on ``params.covariance_type`` ("full": (n_components, k, k), "tied": (k, k), "diag": (n_components, k), "spherical": (n_components,)). Holds the initial covariances when ``warm_start`` is True and receives the fitted covariances. |
| `warm_start` | `bool` | Use the provided weights/means/covariances as the single initialization instead of running ``params.n_init`` restarts. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `weights` | `raft.device_ndarray` | Fitted mixture weights, shape (n_components,) |
| `means` | `raft.device_ndarray` | Fitted component means, shape (n_components, k) |
| `covariances` | `raft.device_ndarray` | Fitted covariances (shape depends on covariance_type) |
| `precisions_chol` | `raft.device_ndarray` | Precision Cholesky factors (shape depends on covariance_type) |
| `precisions` | `raft.device_ndarray` | Precision matrices (shape depends on covariance_type) |
| `labels` | `raft.device_ndarray` | Hard component assignment per sample, shape (m,) |
| `lower_bound` | `float` | Per-sample average log-likelihood of the best fit |
| `n_iter` | `int` | Number of EM iterations of the best fit |
| `converged` | `bool` | Whether the best fit converged within ``params.tol`` |

**Examples**

```python
>>> import cupy as cp
>>>
>>> from cuvs.cluster.gmm import fit, GMMParams
>>>
>>> n_samples = 5000
>>> n_features = 50
>>> n_components = 3
>>>
>>> X = cp.random.random_sample((n_samples, n_features),
...                             dtype=cp.float32)
```

```python
>>> params = GMMParams(n_components=n_components)
>>> weights, means, covariances, precisions_chol, *_ = fit(params, X)
```

## predict

`@auto_sync_resources`
`@auto_convert_output`

```python
def predict(GMMParams params, X, weights, means, precisions_chol, labels=None, resources=None)
```

Hard component labels (argmax responsibility) for new data

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `params` | `GMMParams` | Parameters used to fit the GMM model. |
| `X` | `Input CUDA array interface compliant matrix shape (m, k)` |  |
| `weights` | `Fitted mixture weights, shape (n_components,)` |  |
| `means` | `Fitted component means, shape (n_components, k)` |  |
| `precisions_chol` | `Fitted precision Cholesky factors (shape depends on covariance_type)` |  |
| `labels` | `Optional preallocated CUDA array interface vector shape (m,)` | to hold the output (int32) |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `labels` | `raft.device_ndarray` | Component assignment for each datapoint in X |

**Examples**

```python
>>> import cupy as cp
>>>
>>> from cuvs.cluster.gmm import fit, predict, GMMParams
>>>
>>> X = cp.random.random_sample((5000, 50), dtype=cp.float32)
>>> params = GMMParams(n_components=3)
>>> weights, means, covariances, precisions_chol, *_ = fit(params, X)
>>>
>>> labels = predict(params, X, weights, means, precisions_chol)
```

## predict_proba

`@auto_sync_resources`
`@auto_convert_output`

```python
def predict_proba(GMMParams params, X, weights, means, precisions_chol, resp=None, resources=None)
```

Posterior responsibilities for new data

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `params` | `GMMParams` | Parameters used to fit the GMM model. |
| `X` | `Input CUDA array interface compliant matrix shape (m, k)` |  |
| `weights` | `Fitted mixture weights, shape (n_components,)` |  |
| `means` | `Fitted component means, shape (n_components, k)` |  |
| `precisions_chol` | `Fitted precision Cholesky factors (shape depends on covariance_type)` |  |
| `resp` | `Optional preallocated CUDA array interface matrix shape` | (m, n_components) to hold the output |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `resp` | `raft.device_ndarray` | Posterior probability of each component for each sample |

**Examples**

```python
>>> import cupy as cp
>>>
>>> from cuvs.cluster.gmm import fit, predict_proba, GMMParams
>>>
>>> X = cp.random.random_sample((5000, 50), dtype=cp.float32)
>>> params = GMMParams(n_components=3)
>>> weights, means, covariances, precisions_chol, *_ = fit(params, X)
>>>
>>> resp = predict_proba(params, X, weights, means, precisions_chol)
```

## score_samples

`@auto_sync_resources`
`@auto_convert_output`

```python
def score_samples(GMMParams params, X, weights, means, precisions_chol, log_prob=None, resources=None)
```

Per-sample log-likelihood log p(x_i) for new data

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `params` | `GMMParams` | Parameters used to fit the GMM model. |
| `X` | `Input CUDA array interface compliant matrix shape (m, k)` |  |
| `weights` | `Fitted mixture weights, shape (n_components,)` |  |
| `means` | `Fitted component means, shape (n_components, k)` |  |
| `precisions_chol` | `Fitted precision Cholesky factors (shape depends on covariance_type)` |  |
| `log_prob` | `Optional preallocated CUDA array interface vector shape (m,)` | to hold the output |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `log_prob` | `raft.device_ndarray` | Log-likelihood of each sample under the model |

**Examples**

```python
>>> import cupy as cp
>>>
>>> from cuvs.cluster.gmm import fit, score_samples, GMMParams
>>>
>>> X = cp.random.random_sample((5000, 50), dtype=cp.float32)
>>> params = GMMParams(n_components=3)
>>> weights, means, covariances, precisions_chol, *_ = fit(params, X)
>>>
>>> log_prob = score_samples(params, X, weights, means, precisions_chol)
```

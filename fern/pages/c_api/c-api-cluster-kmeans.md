---
slug: api-reference/c-api-cluster-kmeans
---

# K-Means

_Source header: `c/include/cuvs/cluster/kmeans.h`_

## k-means hyperparameters

_Doxygen group: `kmeans_c_params`_

### cuvsKMeansInitMethod

k-means hyperparameters

```c
typedef enum { ... } cuvsKMeansInitMethod;
```

**Values**

| Name | Value |
| --- | --- |
| `KMeansPlusPlus` | `0` |
| `Random` | `1` |
| `Array` | `2` |

_Source: `c/include/cuvs/cluster/kmeans.h:22`_

### cuvsKMeansParams

Hyper-parameters for the kmeans algorithm

```c
struct cuvsKMeansParams { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_clusters` | `int` | The number of clusters to form as well as the number of centroids to generate (default:8). |
| `init` | `cuvsKMeansInitMethod` | Method for initialization, defaults to k-means++: |
| `max_iter` | `int` | Maximum number of iterations of the k-means algorithm for a single run. |
| `tol` | `double` | Relative tolerance with regards to inertia to declare convergence. |
| `n_init` | `int` | Number of instance k-means algorithm will be run with different seeds. |
| `oversampling_factor` | `double` | Oversampling factor for use in the k-means\|\| algorithm |
| `batch_samples` | `int` | batch_samples and batch_centroids are used to tile 1NN computation which is |
| `batch_centroids` | `int` | if 0 then batch_centroids = n_clusters |
| `inertia_check` | `bool` | Check inertia during iterations for early convergence. |
| `hierarchical` | `bool` | Whether to use hierarchical (balanced) kmeans or not |
| `hierarchical_n_iters` | `int` | For hierarchical k-means , defines the number of training iterations |
| `streaming_batch_size` | `int64_t` | Number of samples to process per GPU batch for the batched (host-data) API. |
| `metric` | `cuvsDistanceType` |  |

_Source: `c/include/cuvs/cluster/kmeans.h:43`_

### cuvsKMeansParamsCreate

Allocate KMeans params, and populate with default values

```c
cuvsError_t cuvsKMeansParamsCreate(cuvsKMeansParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsKMeansParams_t*` | cuvsKMeansParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/cluster/kmeans.h:122`_

### cuvsKMeansParamsDestroy

De-allocate KMeans params

```c
cuvsError_t cuvsKMeansParamsDestroy(cuvsKMeansParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsKMeansParams_t` |  |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/cluster/kmeans.h:130`_

### cuvsKMeansType

Type of k-means algorithm.

```c
typedef enum { ... } cuvsKMeansType;
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_KMEANS_TYPE_KMEANS` | `0` |
| `CUVS_KMEANS_TYPE_KMEANS_BALANCED` | `1` |

_Source: `c/include/cuvs/cluster/kmeans.h:135`_

## k-means clustering APIs

_Doxygen group: `kmeans_c`_

### cuvsKMeansFit

Find clusters with k-means algorithm.

```c
cuvsError_t cuvsKMeansFit(cuvsResources_t res,
cuvsKMeansParams_t params,
DLManagedTensor* X,
DLManagedTensor* sample_weight,
DLManagedTensor* centroids,
double* inertia,
int* n_iter);
```

Initial centroids are chosen with k-means++ algorithm. Empty clusters are reinitialized by choosing new centroids with k-means++ algorithm.

X may reside on either host (CPU) or device (GPU) memory. When X is on the host the data is streamed to the GPU in batches controlled by params-&gt;streaming_batch_size.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | opaque C handle |
| `params` | in | `cuvsKMeansParams_t` | Parameters for KMeans model. |
| `X` | in | `DLManagedTensor*` | Training instances to cluster. The data must be in row-major format. May be on host or device memory. [dim = n_samples x n_features] |
| `sample_weight` | in | `DLManagedTensor*` | Optional weights for each observation in X. Must be on the same memory space as X. [len = n_samples] |
| `centroids` | inout | `DLManagedTensor*` | [in] When init is InitMethod::Array, use centroids as the initial cluster centers. [out] The generated centroids from the kmeans algorithm are stored at the address pointed by 'centroids'. Must be on device. [dim = n_clusters x n_features] |
| `inertia` | out | `double*` | Sum of squared distances of samples to their closest cluster center. |
| `n_iter` | out | `int*` | Number of iterations run. |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/cluster/kmeans.h:176`_

### cuvsKMeansPredict

Predict the closest cluster each sample in X belongs to.

```c
cuvsError_t cuvsKMeansPredict(cuvsResources_t res,
cuvsKMeansParams_t params,
DLManagedTensor* X,
DLManagedTensor* sample_weight,
DLManagedTensor* centroids,
DLManagedTensor* labels,
bool normalize_weight,
double* inertia);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | opaque C handle |
| `params` | in | `cuvsKMeansParams_t` | Parameters for KMeans model. |
| `X` | in | `DLManagedTensor*` | New data to predict. [dim = n_samples x n_features] |
| `sample_weight` | in | `DLManagedTensor*` | Optional weights for each observation in X. [len = n_samples] |
| `centroids` | in | `DLManagedTensor*` | Cluster centroids. The data must be in row-major format. [dim = n_clusters x n_features] |
| `labels` | out | `DLManagedTensor*` | Index of the cluster each sample in X belongs to. [len = n_samples] |
| `normalize_weight` | in | `bool` | True if the weights should be normalized |
| `inertia` | out | `double*` | Sum of squared distances of samples to their closest cluster center. |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/cluster/kmeans.h:203`_

### cuvsKMeansClusterCost

Compute cluster cost

```c
cuvsError_t cuvsKMeansClusterCost(cuvsResources_t res,
DLManagedTensor* X,
DLManagedTensor* centroids,
double* cost);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | opaque C handle |
| `X` | in | `DLManagedTensor*` | Training instances to cluster. The data must be in row-major format. [dim = n_samples x n_features] |
| `centroids` | in | `DLManagedTensor*` | Cluster centroids. The data must be in row-major format. [dim = n_clusters x n_features] |
| `cost` | out | `double*` | Resulting cluster cost |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/cluster/kmeans.h:225`_

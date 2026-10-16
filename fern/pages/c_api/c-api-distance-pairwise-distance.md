---
slug: api-reference/c-api-distance-pairwise-distance
---

# Pairwise Distance

_Source header: `cuvs/distance/pairwise_distance.h`_

## C pairwise distance

<a id="cuvspairwisedistance"></a>
### cuvsPairwiseDistance

Compute pairwise distances for two matrices

```c
cuvsError_t cuvsPairwiseDistance(cuvsResources_t res,
DLManagedTensor* x,
DLManagedTensor* y,
DLManagedTensor* dist,
cuvsDistanceType metric,
float metric_arg);
```

Usage example:

```c
#include <cuvs/core/c_api.h>
#include <cuvs/distance/pairwise_distance.h>

// Create cuvsResources_t
cuvsResources_t res;
cuvsError_t res_create_status = cuvsResourcesCreate(&res);

// Assume a populated `DLManagedTensor` type here
DLManagedTensor x;
DLManagedTensor y;
DLManagedTensor dist;

cuvsPairwiseDistance(res, &x, &y, &dist, L2SqrtUnexpanded, 2.0);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvs resources object for managing expensive resources |
| `x` | in | `DLManagedTensor*` | first set of points (size n*k) |
| `y` | in | `DLManagedTensor*` | second set of points (size m*k) |
| `dist` | out | `DLManagedTensor*` | output distance matrix (size n*m) |
| `metric` | in | [`cuvsDistanceType`](/api-reference/c-api-distance-distance#cuvsdistancetype) | distance to evaluate |
| `metric_arg` | in | `float` | metric argument (used for Minkowski distance) |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

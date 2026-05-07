---
slug: api-reference/c-api-distance-pairwise-distance
---

# Pairwise Distance

_Source header: `c/include/cuvs/distance/pairwise_distance.h`_

## C pairwise distance

_Doxygen group: `pairwise_distance_c`_

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

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvs resources object for managing expensive resources |
| `x` | in | `DLManagedTensor*` | first set of points (size n*k) |
| `y` | in | `DLManagedTensor*` | second set of points (size m*k) |
| `dist` | out | `DLManagedTensor*` | output distance matrix (size n*m) |
| `metric` | in | `cuvsDistanceType` | distance to evaluate |
| `metric_arg` | in | `float` | metric argument (used for Minkowski distance) |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/distance/pairwise_distance.h:48`_

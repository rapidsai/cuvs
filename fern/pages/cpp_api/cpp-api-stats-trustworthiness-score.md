---
slug: api-reference/cpp-api-stats-trustworthiness-score
---

# Trustworthiness Score

_Source header: `cpp/include/cuvs/stats/trustworthiness_score.hpp`_

## Trustworthiness

_Doxygen group: `stats_trustworthiness`_

### stats::trustworthiness_score

Compute the trustworthiness score

```cpp
double trustworthiness_score(
raft::resources const& handle,
raft::device_matrix_view<const float, int64_t, raft::row_major> X,
raft::device_matrix_view<const float, int64_t, raft::row_major> X_embedded,
int n_neighbors,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2SqrtUnexpanded,
int batch_size                      = 512);
```

modified.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `X` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | : Data in original dimension |
| `X_embedded` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | : Data in target dimension (embedding) |
| `n_neighbors` | in | `int` | Number of neighbors considered by trustworthiness score |
| `metric` | in | `cuvs::distance::DistanceType` | Distance metric to use. Euclidean (L2) is used by default Default: `cuvs::distance::DistanceType::L2SqrtUnexpanded`. |
| `batch_size` | in | `int` | Batch size Default: `512`. |

**Returns**

`double`

Trustworthiness score

_Source: `cpp/include/cuvs/stats/trustworthiness_score.hpp:30`_

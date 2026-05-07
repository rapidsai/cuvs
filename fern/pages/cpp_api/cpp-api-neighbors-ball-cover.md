---
slug: api-reference/cpp-api-neighbors-ball-cover
---

# Ball Cover

_Source header: `cpp/include/cuvs/neighbors/ball_cover.hpp`_

## Types

<a id="cuvs-neighbors-ball-cover-index"></a>
### cuvs::neighbors::ball_cover::index

Stores raw index data points, sampled landmarks, the 1-nns of index points

to their closest landmarks, and the ball radii of each landmark. This class is intended to be constructed once and reused across subsequent queries.

```cpp
template <typename idx_t, typename value_t>
struct index : cuvs::neighbors::index { ... };
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `m` | `int64_t` |  |
| `n` | `int64_t` |  |
| `n_landmarks` | `int64_t` |  |
| `X` | `raft::device_matrix_view<const float, idx_t, raft::row_major>` |  |
| `metric` | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#cuvs-distance-distancetype) |  |
| `R_indptr` | `private: raft::device_vector<idx_t, int64_t>` |  |
| `R_1nn_cols` | `raft::device_vector<idx_t, int64_t>` |  |
| `R_1nn_dists` | `raft::device_vector<float, int64_t>` |  |
| `R_closest_landmark_dists` | `raft::device_vector<float, int64_t>` |  |
| `R_radius` | `raft::device_vector<float, int64_t>` |  |
| `R` | `raft::device_matrix<float, int64_t, raft::row_major>` |  |
| `X_reordered` | `raft::device_matrix<float, int64_t, raft::row_major>` |  |
| `index_trained` | `protected: bool` |  |

_Source: `cpp/include/cuvs/neighbors/ball_cover.hpp:35`_

## Random Ball Cover algorithm

<a id="cuvs-neighbors-ball-cover-build"></a>
### cuvs::neighbors::ball_cover::build

Builds and populates a previously unbuilt cuvs::neighbors::ball_cover::index

```cpp
void build(raft::resources const& handle, index<int64_t, float>& index);
```

Usage example:

cuvs::neighbors::ball_cover::index

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | library resource management handle |
| `index` | inout | [`index<int64_t, float>&`](/api-reference/cpp-api-neighbors-ball-cover#cuvs-neighbors-ball-cover-index) | an empty (and not previous built) instance of |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ball_cover.hpp:170`_

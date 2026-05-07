---
slug: api-reference/cpp-api-neighbors-ball-cover
---

# Ball Cover

_Source header: `cpp/include/cuvs/neighbors/ball_cover.hpp`_

## Random Ball Cover algorithm

_Doxygen group: `random_ball_cover`_

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
| `index` | inout | `index<int64_t, float>&` | an empty (and not previous built) instance of |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ball_cover.hpp:170`_

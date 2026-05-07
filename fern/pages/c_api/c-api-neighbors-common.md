---
slug: api-reference/c-api-neighbors-common
---

# Common

_Source header: `c/include/cuvs/neighbors/common.h`_

## Filters APIs

_Doxygen group: `filters`_

### cuvsFilterType

Enum to denote filter type.

```c
enum cuvsFilterType { ... } ;
```

**Values**

| Name | Value |
| --- | --- |
| `NO_FILTER` | `0` |
| `BITSET` | `1` |
| `BITMAP` | `2` |

_Source: `c/include/cuvs/neighbors/common.h:23`_

## Index Merge

_Doxygen group: `index_merge`_

### cuvsMergeStrategy

Strategy for merging indices.

```c
typedef enum { ... } cuvsMergeStrategy;
```

**Values**

| Name | Value |
| --- | --- |
| `MERGE_STRATEGY_PHYSICAL` | `0` |
| `MERGE_STRATEGY_LOGICAL` | `1` |

_Source: `c/include/cuvs/neighbors/common.h:54`_

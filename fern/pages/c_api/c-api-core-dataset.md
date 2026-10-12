---
slug: api-reference/c-api-core-dataset
---

# Dataset

_Source header: `cuvs/core/dataset.h`_

## Types

<a id="cuvsdatasetlayout-t"></a>
### cuvsDatasetLayout_t

Generic dataset layout kind for C API dataset handles.

```c
typedef enum {
  CUVS_DATASET_LAYOUT_STANDARD = 0,
  CUVS_DATASET_LAYOUT_PADDED = 1,
  CUVS_DATASET_LAYOUT_PQ = 2,
  CUVS_DATASET_LAYOUT_BBQ = 3
} cuvsDatasetLayout_t;
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_DATASET_LAYOUT_STANDARD` | `0` |
| `CUVS_DATASET_LAYOUT_PADDED` | `1` |
| `CUVS_DATASET_LAYOUT_PQ` | `2` |
| `CUVS_DATASET_LAYOUT_BBQ` | `3` |

<a id="cuvsdatasetmemtype-t"></a>
### cuvsDatasetMemType_t

Memory space holding a C API dataset handle's data.

```c
typedef enum {
  CUVS_DATASET_MEM_TYPE_HOST = 0,
  CUVS_DATASET_MEM_TYPE_DEVICE = 1
} cuvsDatasetMemType_t;
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_DATASET_MEM_TYPE_HOST` | `0` |
| `CUVS_DATASET_MEM_TYPE_DEVICE` | `1` |

<a id="destroy-addr"></a>
### destroy_addr

Dataset handle representing owning storage or a non-owning view.

`addr` points to C++ dataset storage or view metadata managed by the C API. `mem_type` identifies the memory space, `layout` identifies the data layout, and `is_owning` indicates whether the handle owns its backing data.

```c
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
  cuvsDatasetMemType_t mem_type;
  cuvsDatasetLayout_t layout;
  bool is_owning;
} cuvsDataset;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |
| `dtype` | `DLDataType` |  |
| `mem_type` | [`cuvsDatasetMemType_t`](/api-reference/c-api-core-dataset#cuvsdatasetmemtype-t) |  |
| `layout` | [`cuvsDatasetLayout_t`](/api-reference/c-api-core-dataset#cuvsdatasetlayout-t) |  |
| `is_owning` | `bool` |  |

<a id="cuvscagracompressionparams-t"></a>
### cuvsCagraCompressionParams_t

Compatibility name for PQ dataset parameters; planned for removal in the 27.02 ABI-breaking release.

```c
typedef struct cuvsCagraCompressionParams* cuvsCagraCompressionParams_t;
```

## Dataset concatenation helpers

<a id="cuvscagramergeddatasetoffsets"></a>
### cuvsCagraMergedDatasetOffsets

Compute per-index write offsets for a merged dataset buffer.

```c
cuvsError_t cuvsCagraMergedDatasetOffsets(cuvsResources_t res,
struct cuvsCagraIndex** indices,
size_t num_indices,
cuvsFilter filter,
int64_t* offsets);
```

`cuvsCagraMerge_v2`/`cuvsCagraMergeWithParams_v2` require the caller to have already concatenated every input index's dataset (in `indices` order, applying `filter` if any) into a single buffer and to know each index's starting row within it. For `filter.type == NO_FILTER`, those offsets are just the cumulative sizes of `indices` and this function is not needed. For `BITSET`, the number of surviving rows per index cannot be derived any other way, so call this first.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `indices` | in | [`struct cuvsCagraIndex**`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | Array of input cuvsCagraIndex_t handles that will be passed to merge |
| `num_indices` | in | `size_t` | Number of input indices |
| `filter` | in | [`cuvsFilter`](/api-reference/c-api-core-filter#cuvsfilter) | Filter that will be passed to merge. Only `NO_FILTER` and `BITSET` supported. |
| `offsets` | out | `int64_t*` | Caller-allocated array of `num_indices + 1` int64_t. Entry `i` is the row at which `indices[i]`'s surviving rows must start in the merged buffer; the last entry is the total row count of the merged buffer. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagraconcatenatedatasets"></a>
### cuvsCagraConcatenateDatasets

Concatenate every input index's dataset (in `indices` order) into a freshly allocated, CAGRA-padded, owning device dataset, optionally retaining only the rows selected by `filter`.

```c
cuvsError_t cuvsCagraConcatenateDatasets(cuvsResources_t res,
struct cuvsCagraIndex** indices,
size_t num_indices,
cuvsFilter filter,
cuvsDataset_t* merged_dataset);
```

This is an optional convenience function for building `cuvsCagraMerge_v2`'s `merged_dataset` argument. Callers that already have their own concatenated buffer are not required to use it. Call `cuvsCagraMergedDatasetOffsets` with the same `filter` to get the matching `offsets`.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `indices` | in | [`struct cuvsCagraIndex**`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | Array of input cuvsCagraIndex_t handles to concatenate, in the order they will be passed to `cuvsCagraMerge_v2` |
| `num_indices` | in | `size_t` | Number of input indices |
| `filter` | in | [`cuvsFilter`](/api-reference/c-api-core-filter#cuvsfilter) | Row filter. Pass `\{NO_FILTER, 0\}` (or a zero-initialized `cuvsFilter`) to concatenate all rows. Pass a `BITSET` filter to retain only the surviving rows. |
| `merged_dataset` | out | `cuvsDataset_t*` | Newly allocated owning dataset handle containing the concatenated (and optionally filtered) rows in `indices` order. Caller must destroy it with `cuvsDatasetDestroy` when done. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

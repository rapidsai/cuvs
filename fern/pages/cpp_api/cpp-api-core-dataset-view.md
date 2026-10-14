---
slug: api-reference/cpp-api-core-dataset-view
---

# View

_Source header: `cuvs/core/dataset/view.hpp`_

## Types

<a id="dataset-view"></a>
### dataset_view

A non-owning view of a dataset: a view of the data and, for a compressed dataset, a view of the

dictionary that gives it meaning. Cheap to copy and to pass by value, like an mdspan.

```cpp
template <typename T, typename IdxT, typename SpecT>
struct dataset_view {
  /* The read-only flavor, in which each view type collapses onto its const counterpart. */ static constexpr bool is_const_view;
  static constexpr bool is_compressed;
  /* The distance between the starts of two rows, in elements;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `is_const_view` | `/* The read-only flavor, in which each view type collapses onto its const counterpart. */ static constexpr bool` |  |
| `is_compressed` | `static constexpr bool` |  |
| `elements` | `/* The distance between the starts of two rows, in` |  |

<a id="dataset-view-code-type"></a>
### dataset_view::code_type

What one element of the data slot is

```cpp
using code_type = typename decltype([];
```

- for compressed datasets it is the raw data element type; often is uint8_t but may be a larger type to enforce better alignment.
- for uncompressed datasets it is the value type of the dataset.

**Returns**

[`using code_type = typename`](/api-reference/cpp-api-core-dataset-view#dataset-view-code-type)

<a id="dataset-view-dictionary-type"></a>
### dataset_view::dictionary_type

Optional slot that contains everything to give meaning to the encoded data;

```cpp
using dictionary_type = typename decltype([];
```

`std::monostate` for an uncompressed dataset.

**Returns**

[`using dictionary_type = typename`](/api-reference/cpp-api-core-dataset-view#dataset-view-dictionary-type)

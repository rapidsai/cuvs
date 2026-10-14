---
slug: api-reference/cpp-api-core-dataset-type
---

# Type

_Source header: `cuvs/core/dataset/type.hpp`_

## Types

<a id="dataset"></a>
### dataset

An owning dataset: the data and, for a compressed dataset, the dictionary that gives it meaning.

Ownership is unique, so the type is move-only whenever the underlying storage is.

```cpp
template <typename T, typename IdxT, typename SpecT>
struct dataset {
  using dictionary_type;
  /* NOTE: element type for an owning dataset cannot be const;
  static constexpr bool is_compressed;
  /* @return A read-only view of the whole dataset;
  /* The distance between the starts of two rows, in elements;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `dictionary_type` | `using` | Optional slot that contains everything to give meaning to the encoded data; `std::monostate` for an uncompressed dataset. |
| `const` | [`/* NOTE: element type for an owning dataset cannot be`](/api-reference/cpp-api-core-dataset-type#dataset) |  |
| `is_compressed` | `static constexpr bool` |  |
| `dataset` | `/* @return A read-only view of the whole` |  |
| `elements` | `/* The distance between the starts of two rows, in` |  |

<a id="dictionary-type"></a>
### dictionary_type

Optional slot that contains everything to give meaning to the encoded data;

```cpp
using dictionary_type            = typename view_type::dictionary_type;
```

---
slug: api-reference/cpp-api-core-dataset
---

# Dataset

_Source header: `cuvs/core/dataset.hpp`_

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
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `is_const_view` | `/* The read-only flavor, in which each view type collapses onto its const counterpart. */ static constexpr bool` |  |
| `is_compressed` | `static constexpr bool` |  |

<a id="dataset-view-code-type"></a>
### dataset_view::code_type

What one element of the data slot is

```cpp
using code_type = typename decltype([];
```

- for compressed datasets it is the raw data element type; often is uint8_t but may be a larger type to enforce better alignment.
- for uncompressed datasets it is the value type of the dataset.

**Returns**

[`using code_type = typename`](/api-reference/cpp-api-core-dataset#dataset-view-code-type)

<a id="dataset-view-dictionary-type"></a>
### dataset_view::dictionary_type

Optional slot that contains everything to give meaning to the encoded data;

```cpp
using dictionary_type = typename decltype([];
```

`std::monostate` for an uncompressed dataset.

**Returns**

[`using dictionary_type = typename`](/api-reference/cpp-api-core-dataset#dictionary-type)

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
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `dictionary_type` | `using` | Optional slot that contains everything to give meaning to the encoded data; `std::monostate` for an uncompressed dataset. |
| `const` | [`/* NOTE: element type for an owning dataset cannot be`](/api-reference/cpp-api-core-dataset#dataset) |  |
| `is_compressed` | `static constexpr bool` |  |
| `dataset` | `/* @return A read-only view of the whole` |  |

<a id="dictionary-type"></a>
### dictionary_type

Optional slot that contains everything to give meaning to the encoded data;

```cpp
using dictionary_type            = typename view_type::dictionary_type;
```

<a id="empty-spec"></a>
### empty_spec

Empty dataset specification: contains no data, but keeps the dimension of the dataset implement

the common interface.

```cpp
struct empty_spec;
```

<a id="mdarray-spec"></a>
### mdarray_spec

Dense plain/padded dataset implemented via raft::mdarray.

```cpp
template <typename LayoutPolicy, typename ContainerPolicy>
struct mdarray_spec;
```

<a id="sparse-spec"></a>
### sparse_spec

Sparse dataset specification implemented via the raft sparse matrix hierarchy. `SparseLayoutT`

selects the representation (`csr_layout`, `coo_layout`).

NOTE: only the memory space of the container policy is used here; raft's sparse types bind the policy themselves, separately for the values, the offsets and the indices.

```cpp
template <typename SparseLayoutT, typename ContainerPolicy>
struct sparse_spec;
```

<a id="csr-layout"></a>
### csr_layout

CSR representation: one compressed row per vector of the dataset.

NOTE: the row offsets are 64-bit for the same reason the dense extents are: the number of non-zeros is not bounded by the index type. The column indices are feature ids, so they are bounded by the dataset dimension rather than by its size.

```cpp
struct csr_layout;
```

<a id="coo-layout"></a>
### coo_layout

COO representation: one (row, column, value) triple per non-zero.

NOTE: unlike CSR, the row component is an index into the dataset rather than an offset into the non-zeros, hence the dataset index type.

```cpp
struct coo_layout;
```

<a id="vpq-spec"></a>
### vpq_spec

VQ+PQ compressed dataset specification, mirroring the VPQ dataset of cuvs/neighbors/common.hpp:

the two codebooks form the dictionary, while the data slot keeps the encoded rows - the VQ label in the row prefix, followed by the packed PQ codes.

`BookPolicy` is the container policy of the codebooks (elements of `MathT`) and `CodePolicy` the one of the encoded rows (elements of `uint8_t`); a bound policy cannot be rebound to another element type, hence the two parameters.

```cpp
template <typename MathT, typename BookPolicy, typename CodePolicy>
struct vpq_spec {
  /* The encoded rows are stored like the rows of any other dense dataset;
  only how a row is read and where the dimension comes from are specific to the compression. */ template <typename IdxT> using storage_spec;
  /* The type the codebooks are stored in;
  the rows themselves are always packed codes. */ using math_type;
  /* [vq_n_centers, dim] */ using vq_book_type;
  /* [n_books, pq_n_centers, pq_len], where `n_books` is 1 for a codebook shared by all the subspaces, or `pq_dim` for one codebook per subspace. A row-major 3d array with ` n_books;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `dataset` | `/* The encoded rows are stored like the rows of any other dense` |  |
| `storage_spec` | `only how a row is read and where the dimension comes from are specific to the compression. */ template <typename IdxT> using` |  |
| `in` | `/* The type the codebooks are stored` |  |
| `math_type` | `the rows themselves are always packed codes. */ using` |  |
| `vq_book_type` | `/* [vq_n_centers, dim] */ using` |  |
| `n_books` | `/* [n_books, pq_n_centers, pq_len], where \`n_books\` is 1 for a codebook shared by all the subspaces, or \`pq_dim\` for one codebook per subspace. A row-major 3d array with \`` |  |

<a id="sq-spec"></a>
### sq_spec

Scalar-quantized dataset specification: every component of a vector becomes one code, so the

dictionary is only the range the codes are mapped back onto and the dimension is still a property of the data.

```cpp
template <typename MathT, typename CodePolicy>
struct sq_spec {
  /* One code per component, so the codes are laid out exactly like a dense dataset. */ template <typename IdxT> using storage_spec;
  /* Nothing to allocate: the whole dictionary is the interval the codes are dequantized into. The members are named as in cuvs::preprocessing::quantize::scalar::quantizer. */ struct dictionary_type;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `storage_spec` | [`/* One code per component, so the codes are laid out exactly like a dense dataset. */ template <typename IdxT> using`](/api-reference/cpp-api-core-dataset#dataset) |  |
| `dictionary_type` | [`/* Nothing to allocate: the whole dictionary is the interval the codes are dequantized into. The members are named as in cuvs::preprocessing::quantize::scalar::quantizer. */ struct`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer) |  |

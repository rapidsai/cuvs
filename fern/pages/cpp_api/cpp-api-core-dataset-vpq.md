---
slug: api-reference/cpp-api-core-dataset-vpq
---

# Vpq

_Source header: `cuvs/core/dataset/vpq.hpp`_

## Types

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

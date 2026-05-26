# Using cuVS APIs

cuVS provides APIs for C, C++, Python, Java, Rust, and Go. For CAGRA build and search examples in each language, see the [CAGRA indexing guide](neighbors/cagra.md).

## Filtering vector indexes

cuVS supports different types of filtering depending on the vector index being used. The main method used by vector indexes is pre-filtering, which accounts for filtered vectors before computing closest neighbors and avoids unnecessary distance calculations.

### Bitset

A bitset is an array of bits where each bit can have two possible values: `0` and `1`. In the context of filtering, `0` means the corresponding vector is filtered out and will not be present in the search results.

Bitsets are optimized to use as little memory as possible and are available through RAFT. See the RAFT [bitset API documentation](https://docs.rapids.ai/api/raft/stable/cpp_api/core_bitset/) for more information. When calling a search function of an ANN index, the bitset length should match the number of vectors present in the database.

For CAGRA bitset examples, see [Using Filters](neighbors/cagra.md#using-filters).

### Bitmap

A bitmap is based on the same principle as a bitset, but in two dimensions. This allows users to provide a different bitset for each query being searched. See the RAFT [bitmap API documentation](https://docs.rapids.ai/api/raft/stable/cpp_api/core_bitmap/) for more information.

For Brute-force bitmap examples, see [Using Filters](neighbors/bruteforce.md#using-filters).

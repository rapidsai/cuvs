---
slug: api-reference/cpp-api-preprocessing-spectral-embedding
---

# Spectral Embedding

_Source header: `cpp/include/cuvs/preprocessing/spectral_embedding.hpp`_

## Spectral Embedding

_Doxygen group: `spectral_embedding`_

### cuvs::preprocessing::spectral_embedding::transform

Perform spectral embedding on input dataset

```cpp
void transform(raft::resources const& handle,
params config,
raft::device_matrix_view<float, int, raft::row_major> dataset,
raft::device_matrix_view<float, int, raft::col_major> embedding);
```

This function computes the spectral embedding of the input dataset by: 1. Constructing a k-nearest neighbors graph from the input data 2. Computing the graph Laplacian (normalized or unnormalized) 3. Finding the eigenvectors corresponding to the smallest eigenvalues 4. Using these eigenvectors as the embedding coordinates

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | RAFT resource handle for managing CUDA resources |
| `config` | in | `params` | Parameters controlling the spectral embedding algorithm |
| `dataset` | in | `raft::device_matrix_view<float, int, raft::row_major>` | Input dataset in row-major format [n_samples x n_features] |
| `embedding` | out | `raft::device_matrix_view<float, int, raft::col_major>` | Output embedding in column-major format [n_samples x n_components] |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/spectral_embedding.hpp:115`_

**Additional overload:** `cuvs::preprocessing::spectral_embedding::transform`

Perform spectral embedding using a precomputed connectivity graph

```cpp
void transform(raft::resources const& handle,
params config,
raft::device_coo_matrix_view<float, int, int, int> connectivity_graph,
raft::device_matrix_view<float, int, raft::col_major> embedding);
```

This function computes the spectral embedding from a precomputed sparse connectivity graph (e.g., from a k-NN search or custom similarity matrix). This is useful when you want to use a custom graph construction method or when you have a precomputed similarity/affinity matrix. The function: 1. Converts the COO matrix to the graph Laplacian 2. Computes eigenvectors of the Laplacian 3. Returns the eigenvectors as the embedding

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | RAFT resource handle for managing CUDA resources |
| `config` | in | `params` | Parameters controlling the spectral embedding algorithm (n_neighbors parameter is ignored when using precomputed graph) |
| `connectivity_graph` | in | `raft::device_coo_matrix_view<float, int, int, int>` | Precomputed sparse connectivity/affinity graph in COO format representing weighted connections between samples |
| `embedding` | out | `raft::device_matrix_view<float, int, raft::col_major>` | Output embedding in column-major format [n_samples x n_components] |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/spectral_embedding.hpp:167`_

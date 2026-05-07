---
slug: api-reference/cpp-api-preprocessing-quantize-pq
---

# PQ

_Source header: `cpp/include/cuvs/preprocessing/quantize/pq.hpp`_

## Product Quantizer utilities

_Doxygen group: `pq`_

### cuvs::preprocessing::quantize::pq::params

Product Quantizer parameters.

```cpp
struct params { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `params` | `params(uint32_t pq_bits, uint32_t pq_dim, bool use_subspaces, bool use_vq, uint32_t vq_n_centers, uint32_t kmeans_n_iters, cuvs::cluster::kmeans::kmeans_type` | Simplified constructor that will build an appropriate kmeans params object. |
| `pq_bits` | `uint32_t` | The bit length of the vector element after compression by PQ. |
| `pq_dim` | `uint32_t` | The dimensionality of the vector after compression by PQ. |
| `use_subspaces` | `bool` | Whether to use subspaces for product quantization (PQ). |
| `use_vq` | `bool` | Whether to use Vector Quantization (KMeans) before product quantization (PQ). |
| `vq_n_centers` | `uint32_t` | Vector Quantization (VQ) codebook size - number of "coarse cluster centers". |
| `kmeans_params` | `kmeans_params_variant` | K-means parameters for PQ codebook training. |
| `max_train_points_per_pq_code` | `uint32_t` | The max number of data points to use per PQ code during PQ codebook training. Using more data |
| `max_train_points_per_vq_cluster` | `uint32_t` | The max number of data points to use per VQ cluster during training. |

_Source: `cpp/include/cuvs/preprocessing/quantize/pq.hpp:30`_

### cuvs::preprocessing::quantize::pq::params

Simplified constructor that will build an appropriate kmeans params object.

```cpp
params(uint32_t pq_bits,
uint32_t pq_dim,
bool use_subspaces,
bool use_vq,
uint32_t vq_n_centers,
uint32_t kmeans_n_iters,
cuvs::cluster::kmeans::kmeans_type pq_kmeans_type =
cuvs::cluster::kmeans::kmeans_type::KMeansBalanced,
uint32_t max_train_points_per_pq_code    = 256,
uint32_t max_train_points_per_vq_cluster = 1024)
: pq_bits(pq_bits),
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `pq_bits` |  | `uint32_t` |  |
| `pq_dim` |  | `uint32_t` |  |
| `use_subspaces` |  | `bool` |  |
| `use_vq` |  | `bool` |  |
| `vq_n_centers` |  | `uint32_t` |  |
| `kmeans_n_iters` |  | `uint32_t` |  |
| `pq_kmeans_type` |  | `cuvs::cluster::kmeans::kmeans_type` | Default: `cuvs::cluster::kmeans::kmeans_type::KMeansBalanced`. |
| `max_train_points_per_pq_code` |  | `uint32_t` | Default: `256`. |
| `max_train_points_per_vq_cluster` |  | `uint32_t` | Default: `1024`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/pq.hpp:34`_

### cuvs::preprocessing::quantize::pq::build

Initializes a product quantizer to be used later for quantizing the dataset.

```cpp
quantizer<float> build(raft::resources const& res,
const params params,
raft::device_matrix_view<const float, int64_t> dataset);
```

The use of a pool memory resource is recommended for more consistent training performance. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `params` | in | `const params` | configure product quantizer, e.g. quantile |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t>` | a row-major matrix view on device or host |

**Returns**

`quantizer<float>`

quantizer

_Source: `cpp/include/cuvs/preprocessing/quantize/pq.hpp:169`_

### cuvs::preprocessing::quantize::pq::build

```cpp
quantizer<float> build(raft::resources const& res,
const params params,
raft::host_matrix_view<const float, int64_t> dataset);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | `const params` |  |
| `dataset` |  | `raft::host_matrix_view<const float, int64_t>` |  |

**Returns**

`quantizer<float>`

_Source: `cpp/include/cuvs/preprocessing/quantize/pq.hpp:174`_

### cuvs::preprocessing::quantize::pq::transform

Applies quantization transform to given dataset

```cpp
void transform(raft::resources const& res,
const quantizer<float>& quant,
raft::device_matrix_view<const float, int64_t> dataset,
raft::device_matrix_view<uint8_t, int64_t> codes_out,
std::optional<raft::device_vector_view<uint32_t, int64_t>> vq_labels = std::nullopt);
```

Usage example: used, optional

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quant` | in | `const quantizer<float>&` | a product quantizer |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t>` | a row-major matrix view on device or host |
| `codes_out` | out | `raft::device_matrix_view<uint8_t, int64_t>` | a row-major matrix view on device containing the PQ codes |
| `vq_labels` | out | `std::optional<raft::device_vector_view<uint32_t, int64_t>>` | a vector view on device containing the VQ labels when VQ is Default: `std::nullopt`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/pq.hpp:201`_

### cuvs::preprocessing::quantize::pq::transform

```cpp
void transform(raft::resources const& res,
const quantizer<float>& quant,
raft::host_matrix_view<const float, int64_t> dataset,
raft::device_matrix_view<uint8_t, int64_t> codes_out,
std::optional<raft::device_vector_view<uint32_t, int64_t>> vq_labels = std::nullopt);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `quant` |  | `const quantizer<float>&` |  |
| `dataset` |  | `raft::host_matrix_view<const float, int64_t>` |  |
| `codes_out` |  | `raft::device_matrix_view<uint8_t, int64_t>` |  |
| `vq_labels` |  | `std::optional<raft::device_vector_view<uint32_t, int64_t>>` | Default: `std::nullopt`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/pq.hpp:208`_

### cuvs::preprocessing::quantize::pq::get_quantized_dim

Get the dimension of the quantized dataset (in bytes)

```cpp
inline int64_t get_quantized_dim(const params& config);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `config` | in | `const params&` | product quantizer parameters |

**Returns**

`inline int64_t`

the dimension of the quantized dataset

_Source: `cpp/include/cuvs/preprocessing/quantize/pq.hpp:220`_

### cuvs::preprocessing::quantize::pq::inverse_transform

Applies inverse quantization transform to given dataset

```cpp
void inverse_transform(
raft::resources const& res,
const quantizer<float>& quant,
raft::device_matrix_view<const uint8_t, int64_t> pq_codes,
raft::device_matrix_view<float, int64_t> out,
std::optional<raft::device_vector_view<const uint32_t, int64_t>> vq_labels = std::nullopt);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quant` | in | `const quantizer<float>&` | a product quantizer |
| `pq_codes` | in | `raft::device_matrix_view<const uint8_t, int64_t>` | a row-major matrix view on device containing the PQ codes |
| `out` | out | `raft::device_matrix_view<float, int64_t>` | a row-major matrix view on device |
| `vq_labels` | in | `std::optional<raft::device_vector_view<const uint32_t, int64_t>>` | a vector view on device containing the VQ labels when VQ is used, optional Default: `std::nullopt`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/pq.hpp:235`_

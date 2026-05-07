---
slug: api-reference/c-api-preprocessing-quantize-pq
---

# PQ

_Source header: `c/include/cuvs/preprocessing/quantize/pq.h`_

## C API for Product Quantizer

_Doxygen group: `preprocessing_c_pq`_

### cuvsProductQuantizerParams

Product quantizer parameters.

```c
struct cuvsProductQuantizerParams { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `pq_bits` | `uint32_t` | The bit length of the vector element after compression by PQ. Possible values: within [4, 16]. Hint: the smaller the 'pq_bits', the smaller the index size and the better the search performance, but the lower the recall. |
| `pq_dim` | `uint32_t` | The dimensionality of the vector after compression by PQ. When zero, an optimal value is selected using a heuristic. TODO: at the moment `dim` must be a multiple `pq_dim`. |
| `use_subspaces` | `bool` | Whether to use subspaces for product quantization (PQ). When true, one PQ codebook is used for each subspace. Otherwise, a single PQ codebook is used. |
| `use_vq` | `bool` | Whether to use Vector Quantization (KMeans) before product quantization (PQ). When true, VQ is used before PQ. When false, only product quantization is used. |
| `vq_n_centers` | `uint32_t` | Vector Quantization (VQ) codebook size - number of "coarse cluster centers". When zero, an optimal value is selected using a heuristic. When one, only product quantization is used. |
| `kmeans_n_iters` | `uint32_t` | The number of iterations searching for kmeans centers (both VQ & PQ phases). |
| `pq_kmeans_type` | `cuvsKMeansType` | The type of kmeans algorithm to use for PQ training. |
| `max_train_points_per_pq_code` | `uint32_t` | The max number of data points to use per PQ code during PQ codebook training. Using more data points per PQ code may increase the quality of PQ codebook but may also increase the build time. We will use `pq_n_centers * max_train_points_per_pq_code` training points to train each PQ codebook. |
| `max_train_points_per_vq_cluster` | `uint32_t` | The max number of data points to use per VQ cluster. |

_Source: `c/include/cuvs/preprocessing/quantize/pq.h:24`_

### cuvsProductQuantizerParamsCreate

Allocate Product Quantizer params, and populate with default values

```c
cuvsError_t cuvsProductQuantizerParamsCreate(cuvsProductQuantizerParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsProductQuantizerParams_t*` | cuvsProductQuantizerParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/preprocessing/quantize/pq.h:85`_

### cuvsProductQuantizerParamsDestroy

De-allocate Product Quantizer params

```c
cuvsError_t cuvsProductQuantizerParamsDestroy(cuvsProductQuantizerParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsProductQuantizerParams_t` |  |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/preprocessing/quantize/pq.h:93`_

### cuvsProductQuantizerCreate

Allocate Product Quantizer

```c
cuvsError_t cuvsProductQuantizerCreate(cuvsProductQuantizer_t* quantizer);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `quantizer` | in | `cuvsProductQuantizer_t*` | cuvsProductQuantizer_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/preprocessing/quantize/pq.h:114`_

### cuvsProductQuantizerDestroy

De-allocate Product Quantizer

```c
cuvsError_t cuvsProductQuantizerDestroy(cuvsProductQuantizer_t quantizer);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `quantizer` | in | `cuvsProductQuantizer_t` |  |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/preprocessing/quantize/pq.h:122`_

### cuvsProductQuantizerBuild

Builds a product quantizer to be used later for quantizing the dataset.

```c
cuvsError_t cuvsProductQuantizerBuild(cuvsResources_t res,
cuvsProductQuantizerParams_t params,
DLManagedTensor* dataset,
cuvsProductQuantizer_t quantizer);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | raft resource |
| `params` | in | `cuvsProductQuantizerParams_t` | Parameters for product quantizer training |
| `dataset` | in | `DLManagedTensor*` | a row-major host or device matrix |
| `quantizer` | out | `cuvsProductQuantizer_t` | trained product quantizer |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/preprocessing/quantize/pq.h:132`_

### cuvsProductQuantizerTransform

Applies product quantization transform to the given dataset

```c
cuvsError_t cuvsProductQuantizerTransform(cuvsResources_t res,
cuvsProductQuantizer_t quantizer,
DLManagedTensor* dataset,
DLManagedTensor* codes_out,
DLManagedTensor* vq_labels);
```

This applies product quantization to a dataset.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | raft resource |
| `quantizer` | in | `cuvsProductQuantizer_t` | product quantizer |
| `dataset` | in | `DLManagedTensor*` | a row-major host or device matrix to transform |
| `codes_out` | out | `DLManagedTensor*` | a row-major device matrix to store transformed data |
| `vq_labels` | out | `DLManagedTensor*` | a device vector to store VQ labels. Optional, can be NULL. |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/preprocessing/quantize/pq.h:149`_

### cuvsProductQuantizerInverseTransform

Applies product quantization inverse transform to the given quantized codes

```c
cuvsError_t cuvsProductQuantizerInverseTransform(cuvsResources_t res,
cuvsProductQuantizer_t quantizer,
DLManagedTensor* pq_codes,
DLManagedTensor* out,
DLManagedTensor* vq_labels);
```

This applies product quantization inverse transform to the given quantized codes.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | raft resource |
| `quantizer` | in | `cuvsProductQuantizer_t` | product quantizer |
| `pq_codes` | in | `DLManagedTensor*` | a row-major device matrix of quantized codes |
| `out` | out | `DLManagedTensor*` | a row-major device matrix to store the original data |
| `vq_labels` | out | `DLManagedTensor*` | a device vector containing the VQ labels when VQ is used. Optional, can be NULL. |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/preprocessing/quantize/pq.h:167`_

### cuvsProductQuantizerGetPqBits

Get the bit length of the vector element after compression by PQ.

```c
cuvsError_t cuvsProductQuantizerGetPqBits(cuvsProductQuantizer_t quantizer, uint32_t* pq_bits);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `quantizer` | in | `cuvsProductQuantizer_t` | product quantizer |
| `pq_bits` | out | `uint32_t*` | bit length of the vector element after compression by PQ |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/preprocessing/quantize/pq.h:179`_

### cuvsProductQuantizerGetPqDim

Get the dimensionality of the vector after compression by PQ.

```c
cuvsError_t cuvsProductQuantizerGetPqDim(cuvsProductQuantizer_t quantizer, uint32_t* pq_dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `quantizer` | in | `cuvsProductQuantizer_t` | product quantizer |
| `pq_dim` | out | `uint32_t*` | dimensionality of the vector after compression by PQ |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/preprocessing/quantize/pq.h:187`_

### cuvsProductQuantizerGetPqCodebook

Get the PQ codebook.

```c
cuvsError_t cuvsProductQuantizerGetPqCodebook(cuvsProductQuantizer_t quantizer,
DLManagedTensor* pq_codebook);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `quantizer` | in | `cuvsProductQuantizer_t` | product quantizer |
| `pq_codebook` | out | `DLManagedTensor*` | PQ codebook |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/preprocessing/quantize/pq.h:195`_

### cuvsProductQuantizerGetVqCodebook

Get the VQ codebook.

```c
cuvsError_t cuvsProductQuantizerGetVqCodebook(cuvsProductQuantizer_t quantizer,
DLManagedTensor* vq_codebook);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `quantizer` | in | `cuvsProductQuantizer_t` | product quantizer |
| `vq_codebook` | out | `DLManagedTensor*` | VQ codebook |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/preprocessing/quantize/pq.h:204`_

### cuvsProductQuantizerGetEncodedDim

Get the encoded dimension of the quantized dataset.

```c
cuvsError_t cuvsProductQuantizerGetEncodedDim(cuvsProductQuantizer_t quantizer,
uint32_t* encoded_dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `quantizer` | in | `cuvsProductQuantizer_t` | product quantizer |
| `encoded_dim` | out | `uint32_t*` | encoded dimension of the quantized dataset |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/preprocessing/quantize/pq.h:212`_

### cuvsProductQuantizerGetUseVq

Get whether VQ is used.

```c
cuvsError_t cuvsProductQuantizerGetUseVq(cuvsProductQuantizer_t quantizer, bool* use_vq);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `quantizer` | in | `cuvsProductQuantizer_t` | product quantizer |
| `use_vq` | out | `bool*` | whether VQ is used |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/preprocessing/quantize/pq.h:221`_

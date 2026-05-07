---
slug: api-reference/cpp-api-preprocessing-quantize-scalar
---

# Scalar

_Source header: `cpp/include/cuvs/preprocessing/quantize/scalar.hpp`_

## Scalar quantizer utilities

_Doxygen group: `scalar`_

### cuvs::preprocessing::quantize::scalar::params

quantizer parameters.

```cpp
struct params { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `quantile` | `float` | Specifies how many outliers at top & bottom will be ignored. |

_Source: `cpp/include/cuvs/preprocessing/quantize/scalar.hpp:26`_

### cuvs::preprocessing::quantize::scalar::train

Initializes a scalar quantizer to be used later for quantizing the dataset.

```cpp
quantizer<double> train(raft::resources const& res,
const params params,
raft::device_matrix_view<const double, int64_t> dataset);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `params` | in | `const params` | configure scalar quantizer, e.g. quantile |
| `dataset` | in | `raft::device_matrix_view<const double, int64_t>` | a row-major matrix view on device |

**Returns**

`quantizer<double>`

quantizer

_Source: `cpp/include/cuvs/preprocessing/quantize/scalar.hpp:67`_

### cuvs::preprocessing::quantize::scalar::train

Initializes a scalar quantizer to be used later for quantizing the dataset.

```cpp
quantizer<double> train(raft::resources const& res,
const params params,
raft::host_matrix_view<const double, int64_t> dataset);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `params` | in | `const params` | configure scalar quantizer, e.g. quantile |
| `dataset` | in | `raft::host_matrix_view<const double, int64_t>` | a row-major matrix view on host |

**Returns**

`quantizer<double>`

quantizer

_Source: `cpp/include/cuvs/preprocessing/quantize/scalar.hpp:87`_

### cuvs::preprocessing::quantize::scalar::transform

Applies quantization transform to given dataset

```cpp
void transform(raft::resources const& res,
const quantizer<double>& quantizer,
raft::device_matrix_view<const double, int64_t> dataset,
raft::device_matrix_view<int8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | `const quantizer<double>&` | a scalar quantizer |
| `dataset` | in | `raft::device_matrix_view<const double, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<int8_t, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/scalar.hpp:110`_

### cuvs::preprocessing::quantize::scalar::transform

Applies quantization transform to given dataset

```cpp
void transform(raft::resources const& res,
const quantizer<double>& quantizer,
raft::host_matrix_view<const double, int64_t> dataset,
raft::host_matrix_view<int8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | `const quantizer<double>&` | a scalar quantizer |
| `dataset` | in | `raft::host_matrix_view<const double, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<int8_t, int64_t>` | a row-major matrix view on host |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/scalar.hpp:134`_

### cuvs::preprocessing::quantize::scalar::inverse_transform

Perform inverse quantization step on previously quantized dataset

```cpp
void inverse_transform(raft::resources const& res,
const quantizer<double>& quantizer,
raft::device_matrix_view<const int8_t, int64_t> dataset,
raft::device_matrix_view<double, int64_t> out);
```

Note that depending on the chosen data types train dataset the conversion is not lossless. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | `const quantizer<double>&` | a scalar quantizer |
| `dataset` | in | `raft::device_matrix_view<const int8_t, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<double, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/scalar.hpp:161`_

### cuvs::preprocessing::quantize::scalar::inverse_transform

Perform inverse quantization step on previously quantized dataset

```cpp
void inverse_transform(raft::resources const& res,
const quantizer<double>& quantizer,
raft::host_matrix_view<const int8_t, int64_t> dataset,
raft::host_matrix_view<double, int64_t> out);
```

Note that depending on the chosen data types train dataset the conversion is not lossless. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | `const quantizer<double>&` | a scalar quantizer |
| `dataset` | in | `raft::host_matrix_view<const int8_t, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<double, int64_t>` | a row-major matrix view on host |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/scalar.hpp:187`_

### cuvs::preprocessing::quantize::scalar::train

Initializes a scalar quantizer to be used later for quantizing the dataset.

```cpp
quantizer<float> train(raft::resources const& res,
const params params,
raft::device_matrix_view<const float, int64_t> dataset);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `params` | in | `const params` | configure scalar quantizer, e.g. quantile |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t>` | a row-major matrix view on device |

**Returns**

`quantizer<float>`

quantizer

_Source: `cpp/include/cuvs/preprocessing/quantize/scalar.hpp:208`_

### cuvs::preprocessing::quantize::scalar::train

Initializes a scalar quantizer to be used later for quantizing the dataset.

```cpp
quantizer<float> train(raft::resources const& res,
const params params,
raft::host_matrix_view<const float, int64_t> dataset);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `params` | in | `const params` | configure scalar quantizer, e.g. quantile |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t>` | a row-major matrix view on host |

**Returns**

`quantizer<float>`

quantizer

_Source: `cpp/include/cuvs/preprocessing/quantize/scalar.hpp:228`_

### cuvs::preprocessing::quantize::scalar::transform

Applies quantization transform to given dataset

```cpp
void transform(raft::resources const& res,
const quantizer<float>& quantizer,
raft::device_matrix_view<const float, int64_t> dataset,
raft::device_matrix_view<int8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | `const quantizer<float>&` | a scalar quantizer |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<int8_t, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/scalar.hpp:251`_

### cuvs::preprocessing::quantize::scalar::transform

Applies quantization transform to given dataset

```cpp
void transform(raft::resources const& res,
const quantizer<float>& quantizer,
raft::host_matrix_view<const float, int64_t> dataset,
raft::host_matrix_view<int8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | `const quantizer<float>&` | a scalar quantizer |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<int8_t, int64_t>` | a row-major matrix view on host |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/scalar.hpp:275`_

### cuvs::preprocessing::quantize::scalar::inverse_transform

Perform inverse quantization step on previously quantized dataset

```cpp
void inverse_transform(raft::resources const& res,
const quantizer<float>& quantizer,
raft::device_matrix_view<const int8_t, int64_t> dataset,
raft::device_matrix_view<float, int64_t> out);
```

Note that depending on the chosen data types train dataset the conversion is not lossless. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | `const quantizer<float>&` | a scalar quantizer |
| `dataset` | in | `raft::device_matrix_view<const int8_t, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<float, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/scalar.hpp:301`_

### cuvs::preprocessing::quantize::scalar::inverse_transform

Perform inverse quantization step on previously quantized dataset

```cpp
void inverse_transform(raft::resources const& res,
const quantizer<float>& quantizer,
raft::host_matrix_view<const int8_t, int64_t> dataset,
raft::host_matrix_view<float, int64_t> out);
```

Note that depending on the chosen data types train dataset the conversion is not lossless. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | `const quantizer<float>&` | a scalar quantizer |
| `dataset` | in | `raft::host_matrix_view<const int8_t, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<float, int64_t>` | a row-major matrix view on host |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/scalar.hpp:327`_

### cuvs::preprocessing::quantize::scalar::train

Initializes a scalar quantizer to be used later for quantizing the dataset.

```cpp
quantizer<half> train(raft::resources const& res,
const params params,
raft::device_matrix_view<const half, int64_t> dataset);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `params` | in | `const params` | configure scalar quantizer, e.g. quantile |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t>` | a row-major matrix view on device |

**Returns**

`quantizer<half>`

quantizer

_Source: `cpp/include/cuvs/preprocessing/quantize/scalar.hpp:348`_

### cuvs::preprocessing::quantize::scalar::train

Initializes a scalar quantizer to be used later for quantizing the dataset.

```cpp
quantizer<half> train(raft::resources const& res,
const params params,
raft::host_matrix_view<const half, int64_t> dataset);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `params` | in | `const params` | configure scalar quantizer, e.g. quantile |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t>` | a row-major matrix view on host |

**Returns**

`quantizer<half>`

quantizer

_Source: `cpp/include/cuvs/preprocessing/quantize/scalar.hpp:368`_

### cuvs::preprocessing::quantize::scalar::transform

Applies quantization transform to given dataset

```cpp
void transform(raft::resources const& res,
const quantizer<half>& quantizer,
raft::device_matrix_view<const half, int64_t> dataset,
raft::device_matrix_view<int8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | `const quantizer<half>&` | a scalar quantizer |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<int8_t, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/scalar.hpp:391`_

### cuvs::preprocessing::quantize::scalar::transform

Applies quantization transform to given dataset

```cpp
void transform(raft::resources const& res,
const quantizer<half>& quantizer,
raft::host_matrix_view<const half, int64_t> dataset,
raft::host_matrix_view<int8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | `const quantizer<half>&` | a scalar quantizer |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<int8_t, int64_t>` | a row-major matrix view on host |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/scalar.hpp:415`_

### cuvs::preprocessing::quantize::scalar::inverse_transform

Perform inverse quantization step on previously quantized dataset

```cpp
void inverse_transform(raft::resources const& res,
const quantizer<half>& quantizer,
raft::device_matrix_view<const int8_t, int64_t> dataset,
raft::device_matrix_view<half, int64_t> out);
```

Note that depending on the chosen data types train dataset the conversion is not lossless. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | `const quantizer<half>&` | a scalar quantizer |
| `dataset` | in | `raft::device_matrix_view<const int8_t, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<half, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/scalar.hpp:441`_

### cuvs::preprocessing::quantize::scalar::inverse_transform

Perform inverse quantization step on previously quantized dataset

```cpp
void inverse_transform(raft::resources const& res,
const quantizer<half>& quantizer,
raft::host_matrix_view<const int8_t, int64_t> dataset,
raft::host_matrix_view<half, int64_t> out);
```

Note that depending on the chosen data types train dataset the conversion is not lossless. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | `const quantizer<half>&` | a scalar quantizer |
| `dataset` | in | `raft::host_matrix_view<const int8_t, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<half, int64_t>` | a row-major matrix view on host |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/scalar.hpp:467`_

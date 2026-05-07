---
slug: api-reference/cpp-api-preprocessing-quantize-binary
---

# Binary

_Source header: `cpp/include/cuvs/preprocessing/quantize/binary.hpp`_

## Binary quantizer utilities

<a id="cuvs-preprocessing-quantize-binary-bit-threshold"></a>
### cuvs::preprocessing::quantize::binary::bit_threshold

quantizer algorithms. The mean and sampling_median thresholds are calculated separately

for each dimension.

```cpp
enum class bit_threshold { ... } ;
```

**Values**

| Name | Value |
| --- | --- |
| `zero` | `` |
| `mean` | `` |
| `sampling_median` | `` |

_Source: `cpp/include/cuvs/preprocessing/quantize/binary.hpp:27`_

<a id="cuvs-preprocessing-quantize-binary-params"></a>
### cuvs::preprocessing::quantize::binary::params

quantizer parameters.

```cpp
struct params { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `threshold` | [`bit_threshold`](/api-reference/cpp-api-preprocessing-quantize-binary#cuvs-preprocessing-quantize-binary-bit-threshold) | Threshold method for binarization. |
| `sampling_ratio` | `float` | Specifies the sampling ratio. |

_Source: `cpp/include/cuvs/preprocessing/quantize/binary.hpp:32`_

<a id="cuvs-preprocessing-quantize-binary-quantizer"></a>
### cuvs::preprocessing::quantize::binary::quantizer

Construct a quantizer with an empty threshold vector.

```cpp
quantizer(raft::resources const& res) : threshold(raft::make_device_vector<T, int64_t>(res, 0));
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/binary.hpp:54`_

<a id="cuvs-preprocessing-quantize-binary-train"></a>
### cuvs::preprocessing::quantize::binary::train

Initializes a binary quantizer to be used later for quantizing the dataset.

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
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-binary#cuvs-preprocessing-quantize-binary-params) | configure binary quantizer, e.g. threshold |
| `dataset` | in | `raft::device_matrix_view<const double, int64_t>` | a row-major matrix view on device |

**Returns**

`quantizer<double>`

quantizer

_Source: `cpp/include/cuvs/preprocessing/quantize/binary.hpp:73`_

**Additional overload:** `cuvs::preprocessing::quantize::binary::train`

Initializes a binary quantizer to be used later for quantizing the dataset.

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
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-binary#cuvs-preprocessing-quantize-binary-params) | configure binary quantizer, e.g. threshold |
| `dataset` | in | `raft::host_matrix_view<const double, int64_t>` | a row-major matrix view on host |

**Returns**

`quantizer<double>`

quantizer

_Source: `cpp/include/cuvs/preprocessing/quantize/binary.hpp:93`_

<a id="cuvs-preprocessing-quantize-binary-transform"></a>
### cuvs::preprocessing::quantize::binary::transform

Applies binary quantization transform to given dataset. If a dataset element is positive,

```cpp
void transform(raft::resources const& res,
const quantizer<double>& quantizer,
raft::device_matrix_view<const double, int64_t> dataset,
raft::device_matrix_view<uint8_t, int64_t> out);
```

set the corresponding bit to 1.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | `const quantizer<double>&` | a binary quantizer |
| `dataset` | in | `raft::device_matrix_view<const double, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<uint8_t, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/binary.hpp:119`_

**Additional overload:** `cuvs::preprocessing::quantize::binary::transform`

Applies binary quantization transform to given dataset. If a dataset element is positive,

```cpp
void transform(raft::resources const& res,
const quantizer<double>& quantizer,
raft::host_matrix_view<const double, int64_t> dataset,
raft::host_matrix_view<uint8_t, int64_t> out);
```

set the corresponding bit to 1.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | `const quantizer<double>&` | a binary quantizer |
| `dataset` | in | `raft::host_matrix_view<const double, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<uint8_t, int64_t>` | a row-major matrix view on host |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/binary.hpp:146`_

**Additional overload:** `cuvs::preprocessing::quantize::binary::train`

Initializes a binary quantizer to be used later for quantizing the dataset.

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
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-binary#cuvs-preprocessing-quantize-binary-params) | configure binary quantizer, e.g. threshold |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t>` | a row-major matrix view on device |

**Returns**

`quantizer<float>`

quantizer

_Source: `cpp/include/cuvs/preprocessing/quantize/binary.hpp:167`_

**Additional overload:** `cuvs::preprocessing::quantize::binary::train`

Initializes a binary quantizer to be used later for quantizing the dataset.

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
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-binary#cuvs-preprocessing-quantize-binary-params) | configure binary quantizer, e.g. threshold |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t>` | a row-major matrix view on host |

**Returns**

`quantizer<float>`

quantizer

_Source: `cpp/include/cuvs/preprocessing/quantize/binary.hpp:187`_

**Additional overload:** `cuvs::preprocessing::quantize::binary::transform`

Applies binary quantization transform to given dataset. If a dataset element is positive,

```cpp
void transform(raft::resources const& res,
const quantizer<float>& quantizer,
raft::device_matrix_view<const float, int64_t> dataset,
raft::device_matrix_view<uint8_t, int64_t> out);
```

set the corresponding bit to 1.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | `const quantizer<float>&` | a binary quantizer |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<uint8_t, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/binary.hpp:215`_

**Additional overload:** `cuvs::preprocessing::quantize::binary::transform`

Applies binary quantization transform to given dataset. If a dataset element is positive,

```cpp
void transform(raft::resources const& res,
const quantizer<float>& quantizer,
raft::host_matrix_view<const float, int64_t> dataset,
raft::host_matrix_view<uint8_t, int64_t> out);
```

set the corresponding bit to 1.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | `const quantizer<float>&` | a binary quantizer |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<uint8_t, int64_t>` | a row-major matrix view on host |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/binary.hpp:244`_

**Additional overload:** `cuvs::preprocessing::quantize::binary::train`

Initializes a binary quantizer to be used later for quantizing the dataset.

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
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-binary#cuvs-preprocessing-quantize-binary-params) | configure binary quantizer, e.g. threshold |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t>` | a row-major matrix view on device |

**Returns**

`quantizer<half>`

quantizer

_Source: `cpp/include/cuvs/preprocessing/quantize/binary.hpp:265`_

**Additional overload:** `cuvs::preprocessing::quantize::binary::train`

Initializes a binary quantizer to be used later for quantizing the dataset.

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
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-binary#cuvs-preprocessing-quantize-binary-params) | configure binary quantizer, e.g. threshold |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t>` | a row-major matrix view on host |

**Returns**

`quantizer<half>`

quantizer

_Source: `cpp/include/cuvs/preprocessing/quantize/binary.hpp:285`_

**Additional overload:** `cuvs::preprocessing::quantize::binary::transform`

Applies binary quantization transform to given dataset.

```cpp
void transform(raft::resources const& res,
const quantizer<half>& quantizer,
raft::device_matrix_view<const half, int64_t> dataset,
raft::device_matrix_view<uint8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | `const quantizer<half>&` | a binary quantizer |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<uint8_t, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/binary.hpp:312`_

**Additional overload:** `cuvs::preprocessing::quantize::binary::transform`

Applies binary quantization transform to given dataset.

```cpp
void transform(raft::resources const& res,
const quantizer<half>& quantizer,
raft::host_matrix_view<const half, int64_t> dataset,
raft::host_matrix_view<uint8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | `const quantizer<half>&` | a binary quantizer |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<uint8_t, int64_t>` | a row-major matrix view on host |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/binary.hpp:340`_

<a id="cuvs-preprocessing-quantize-binary-deprecated"></a>
### cuvs::preprocessing::quantize::binary::[[deprecated

[deprecated] Applies binary quantization transform to given dataset. If a dataset element

```cpp
[[deprecated("please create and specify a quantizer")]] void transform(
raft::resources const& res,
raft::device_matrix_view<const double, int64_t> dataset,
raft::device_matrix_view<uint8_t, int64_t> out);
```

is positive, set the corresponding bit to 1.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | `"please create and specify a quantizer"` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/binary.hpp:354`_

**Additional overload:** `cuvs::preprocessing::quantize::binary::[[deprecated`

[deprecated] Applies binary quantization transform to given dataset. If a dataset element

```cpp
[[deprecated("please create and specify a quantizer")]] void transform(
raft::resources const& res,
raft::host_matrix_view<const double, int64_t> dataset,
raft::host_matrix_view<uint8_t, int64_t> out);
```

is positive, set the corresponding bit to 1.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | `"please create and specify a quantizer"` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/binary.hpp:368`_

**Additional overload:** `cuvs::preprocessing::quantize::binary::[[deprecated`

[deprecated] Applies binary quantization transform to given dataset. If a dataset element

```cpp
[[deprecated("please create and specify a quantizer")]] void transform(
raft::resources const& res,
raft::device_matrix_view<const float, int64_t> dataset,
raft::device_matrix_view<uint8_t, int64_t> out);
```

is positive, set the corresponding bit to 1.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | `"please create and specify a quantizer"` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/binary.hpp:382`_

**Additional overload:** `cuvs::preprocessing::quantize::binary::[[deprecated`

[deprecated] Applies binary quantization transform to given dataset. If a dataset element

```cpp
[[deprecated("please create and specify a quantizer")]] void transform(
raft::resources const& res,
raft::host_matrix_view<const float, int64_t> dataset,
raft::host_matrix_view<uint8_t, int64_t> out);
```

is positive, set the corresponding bit to 1.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | `"please create and specify a quantizer"` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/binary.hpp:396`_

**Additional overload:** `cuvs::preprocessing::quantize::binary::[[deprecated`

[deprecated] Applies binary quantization transform to given dataset. If a dataset element

```cpp
[[deprecated("please create and specify a quantizer")]] void transform(
raft::resources const& res,
raft::device_matrix_view<const half, int64_t> dataset,
raft::device_matrix_view<uint8_t, int64_t> out);
```

is positive, set the corresponding bit to 1.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | `"please create and specify a quantizer"` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/binary.hpp:410`_

**Additional overload:** `cuvs::preprocessing::quantize::binary::[[deprecated`

[deprecated] Applies binary quantization transform to given dataset. If a dataset element

```cpp
[[deprecated("please create and specify a quantizer")]] void transform(
raft::resources const& res,
raft::host_matrix_view<const half, int64_t> dataset,
raft::host_matrix_view<uint8_t, int64_t> out);
```

is positive, set the corresponding bit to 1.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | `"please create and specify a quantizer"` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/preprocessing/quantize/binary.hpp:424`_

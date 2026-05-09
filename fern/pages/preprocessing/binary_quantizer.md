# Binary Quantizer

Binary quantization compresses each vector into a packed bit vector. Each input value becomes one bit, and each bit records whether the value is greater than a threshold.

Use binary quantization when you want very compact vectors for workflows based on bitwise distances such as Hamming distance. It is a strong compression step, so it trades away most magnitude information for much smaller storage and faster bitwise comparisons.

## Example API Usage

[C API](/api-reference/c-api-preprocessing-quantize-binary) | [C++ API](/api-reference/cpp-api-preprocessing-quantize-binary) | [Python API](/api-reference/python-api-preprocessing-quantize-binary)

### Training a quantizer

Training is needed when thresholds are learned from data, such as per-dimension means or sampled medians. The Python API currently exposes the direct zero-threshold transform path.

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>
#include <cuvs/preprocessing/quantize/binary.h>

cuvsResources_t res;
cuvsBinaryQuantizerParams_t params;
cuvsBinaryQuantizer_t quantizer;
DLManagedTensor *dataset;

load_dataset(dataset);

cuvsResourcesCreate(&res);
cuvsBinaryQuantizerParamsCreate(&params);
cuvsBinaryQuantizerCreate(&quantizer);

params->threshold = MEAN;
params->sampling_ratio = 0.1f;

cuvsBinaryQuantizerTrain(res, params, dataset, quantizer);

cuvsBinaryQuantizerDestroy(quantizer);
cuvsBinaryQuantizerParamsDestroy(params);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/preprocessing/quantize/binary.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

namespace binary = cuvs::preprocessing::quantize::binary;

raft::device_resources res;
raft::device_matrix_view<const float, int64_t> dataset = load_dataset();

binary::params params;
params.threshold = binary::bit_threshold::mean;
params.sampling_ratio = 0.1f;

auto quantizer = binary::train(res, params, dataset);
```

</Tab>
<Tab title="Python">

```python
import cupy as cp

from cuvs.preprocessing.quantize import binary

dataset = cp.asarray(load_dataset(), dtype=cp.float32)
codes = binary.transform(dataset)
```

</Tab>
</Tabs>

### Transforming data

Transforming packs every group of eight input dimensions into one byte. In C and C++, use the trained quantizer when using learned thresholds. Python uses a zero threshold and sets a bit when the corresponding input value is positive.

<Tabs>
<Tab title="C">

```c
DLManagedTensor *codes;
uint64_t packed_dim = (n_features + 7) / 8;

allocate_uint8_matrix(codes, n_rows, packed_dim);

cuvsBinaryQuantizerTransformWithParams(res, quantizer, dataset, codes);
```

</Tab>
<Tab title="C++">

```cpp
auto packed_dim = (dataset.extent(1) + 7) / 8;
auto codes = raft::make_device_matrix<uint8_t, int64_t>(
    res, dataset.extent(0), packed_dim);

binary::transform(res, quantizer, dataset, codes.view());
```

</Tab>
<Tab title="Python">

```python
codes = binary.transform(dataset)
```

</Tab>
</Tabs>

## How Binary Quantization works

Binary quantization compares each value with a threshold. If the value is greater than the threshold, the output bit is set to `1`; otherwise it is set to `0`.

The threshold can be fixed at zero, learned as a per-dimension mean, or learned as a sampled per-dimension median. The output is packed into bytes, so a vector with `D` features needs `ceil(D / 8)` bytes.

## When to use Binary Quantization

Use binary quantization when compact storage and bitwise comparison speed matter more than preserving floating-point magnitude. It is especially useful for binary vector search with bitwise Hamming distance.

Avoid binary quantization when vector magnitude or fine-grained distance information is important. Because each value becomes only one bit, this is the most aggressive quantizer in the preprocessing guide.

## Configuration parameters

### Train parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `threshold` | `mean` | Threshold strategy. `zero` uses zero, `mean` learns a per-dimension mean, and `sampling_median` estimates a per-dimension median from samples. |
| `sampling_ratio` | `0.1` | Fraction of rows sampled when using the sampled median threshold. Higher values can improve the median estimate but increase training work. |

## Tuning

Use `zero` when the data is already centered or when you want the fastest path. Use `mean` when dimensions have different offsets. Use `sampling_median` when outliers skew the mean and a more robust threshold is needed.

Increase `sampling_ratio` only when the sampled median is unstable. It does not affect the compressed size, only threshold quality and training cost.

## Memory footprint

Binary quantization is usually dominated by the input matrix and the packed output matrix.

Variables:

- `N`: Number of vectors.
- `D`: Number of features per vector.
- `B_x`: Bytes per input element.
- `B_t`: Bytes per threshold value.

### Packed vectors

Each row stores one bit per input feature:

$$
\begin{aligned}
\text{packed\_dim}
  &= \left\lceil \frac{D}{8} \right\rceil \\
\text{codes\_size}
  &= N \cdot \text{packed\_dim}
\end{aligned}
$$

### Thresholds

Learned thresholds store one threshold per input dimension:

$$
\text{threshold\_size}
  \approx D \cdot B_t
$$

The zero-threshold path does not need a learned threshold vector.

### Transform peak

The transform peak is approximately:

$$
\begin{aligned}
\text{transform\_peak}
  \approx&\ N \cdot D \cdot B_x
   + \text{codes\_size} \\
  &+ \text{threshold\_size}
   + \text{scratch}
\end{aligned}
$$

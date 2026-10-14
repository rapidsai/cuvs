---
slug: api-reference/cpp-api-core-dataset-sq
---

# SQ

_Source header: `cuvs/core/dataset/sq.hpp`_

## Types

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
| `storage_spec` | [`/* One code per component, so the codes are laid out exactly like a dense dataset. */ template <typename IdxT> using`](/api-reference/cpp-api-core-dataset-type#dataset) |  |
| `dictionary_type` | [`/* Nothing to allocate: the whole dictionary is the interval the codes are dequantized into. The members are named as in cuvs::preprocessing::quantize::scalar::quantizer. */ struct`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer) |  |

---
slug: api-reference/cpp-api-core-dataset-bitmap
---

# Bitmap

_Source header: `cuvs/core/dataset/bitmap.hpp`_

## Types

<a id="device-bitmap-dataset-view"></a>
### device_bitmap_dataset_view

A non-owning view of a row-major bitmap dataset.

The rows are packed without padding: bit `(row, col)` is stored at linear bit offset `row * dim() + col`. The referenced storage must outlive the view and every operation using it.

```cpp
template <typename BitmapT, typename IdxT>
struct device_bitmap_dataset_view;
```

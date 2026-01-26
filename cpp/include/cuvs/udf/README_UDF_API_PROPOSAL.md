# cuVS UDF (User-Defined Function) API Proposal

This folder contains a proposed UDF API for cuVS custom distance metrics.

## Design Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  User writes (via macro):                                                   │
│                                                                             │
│    CUVS_METRIC(my_l2, {                                                     │
│        acc += cuvs::udf::squared_diff(x, y);  // Just works!                │
│    })                                                                       │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  x and y are point<T, AccT, Veclen> which provides:                         │
│                                                                             │
│    x.raw()      - Raw packed storage (for power users)                      │
│    x[i]         - Unpacked element access                                   │
│    x.size()     - Number of elements (4 for packed int8, 1 for float)       │
│    x.is_packed() - Whether data is packed (constexpr)                       │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Helper functions deduce Veclen automatically:                              │
│                                                                             │
│    cuvs::udf::squared_diff(x, y)  - (x-y)² optimal for ALL types            │
│    cuvs::udf::abs_diff(x, y)      - |x-y| optimal for ALL types             │
│    cuvs::udf::dot_product(x, y)   - x·y optimal for ALL types               │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  At runtime, cuVS wraps raw values and calls your metric:                   │
│                                                                             │
│    point_t x{x_raw}, y{y_raw};                                              │
│    my_l2<T, AccT, Veclen>{}(acc, x, y);                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Files

```
cpp/include/cuvs/udf/
├── README_UDF_API_PROPOSAL.md  # This file
├── point.cuh                   # point<T,AccT,Veclen> wrapper + helpers
├── metric_interface.cuh        # Base interface (compile-time safety)
├── metric_macro.hpp            # CUVS_METRIC macro
├── metric_source.hpp           # metric_source struct
├── compiler.hpp                # Internal JIT compiler interface
└── packed_helpers.cuh          # [DEPRECATED] Legacy helpers

cpp/src/udf/
└── compiler.cpp                # JIT compiler implementation

examples/cpp/src/
├── udf_simple_metric.cu        # Basic example
├── udf_chebyshev_metric.cu     # L∞ distance
├── udf_int8_metric.cu          # int8/uint8 - now easy!
└── udf_weighted_metric.cu      # Custom headers
```

## Quick Start

### 1. Define Your Metric - The Easy Way

```cpp
#include <cuvs/udf/metric_macro.hpp>

// L2 distance - works for float, half, int8, uint8!
CUVS_METRIC(my_l2, {
    acc += cuvs::udf::squared_diff(x, y);
})

// L1 distance - works for all types!
CUVS_METRIC(my_l1, {
    acc += cuvs::udf::abs_diff(x, y);
})

// Inner product - works for all types!
CUVS_METRIC(my_dot, {
    acc += cuvs::udf::dot_product(x, y);
})
```

### 2. Custom Logic with Element Access

```cpp
// Chebyshev (L∞) distance - max absolute difference
CUVS_METRIC(chebyshev, {
    for (int i = 0; i < x.size(); ++i) {
        auto diff = (x[i] > y[i]) ? (x[i] - y[i]) : (y[i] - x[i]);
        if (diff > acc) acc = static_cast<AccT>(diff);
    }
})

// Per-dimension weighted L2
CUVS_METRIC(weighted_l2, {
    for (int i = 0; i < x.size(); ++i) {
        auto diff = x[i] - y[i];
        auto weight = AccT{1} + AccT{i} * AccT{0.1};  // Custom weights
        acc += weight * diff * diff;
    }
})
```

### 3. Use in Search

```cpp
#include <cuvs/neighbors/ivf_flat.hpp>

int main() {
    auto index = cuvs::neighbors::ivf_flat::deserialize(res, "index.bin");
    
    cuvs::neighbors::ivf_flat::search_params params;
    params.udf.metric = my_l2_udf();  // Auto-generated function!
    
    cuvs::neighbors::ivf_flat::search(res, params, index, queries, neighbors, distances);
}
```

## The `point<T, AccT, Veclen>` Wrapper

The key innovation is wrapping raw values in `point<T, AccT, Veclen>`:

```cpp
template <typename T, typename AccT, int Veclen>
struct point {
    storage_type data_;
    
    // Raw access for power users
    __device__ storage_type raw() const;
    
    // Element access - handles unpacking automatically
    __device__ T operator[](int i) const;
    
    // Compile-time queries
    static constexpr int size();        // 4 for packed int8, 1 for float
    static constexpr bool is_packed();  // true for int8/uint8 with Veclen > 1
};
```

### Benefits

1. **Helpers deduce Veclen automatically** - no template args needed!
2. **Element access `x[i]`** - unpacks automatically for int8/uint8
3. **Type queries** - `is_packed()`, `size()` for conditional logic
4. **Raw access** - `x.raw()` for power users who need intrinsics

## Helper Functions

All helpers deduce `Veclen` from the `point` type - no manual template args!

| Helper | Description | int8/uint8 Implementation |
|--------|-------------|---------------------------|
| `squared_diff(x, y)` | (x-y)² | `__vabsdiffs4` + `__dp4a` |
| `abs_diff(x, y)` | \|x-y\| | `__vabsdiffs4` + byte sum |
| `dot_product(x, y)` | x·y | `__dp4a` |
| `product(x, y)` | element-wise × | `__dp4a` |
| `sum(x, y)` | element-wise + | unpacked loop |
| `max_elem(x, y)` | max element | unpacked loop |

## Supported Types

| Data Type | Accumulator | `x.size()` | Complexity |
|-----------|-------------|------------|------------|
| `float` | `float` | 1 | ⭐ Easy |
| `__half` | `__half` | 1 | ⭐ Easy |
| `int8_t` | `int32_t` | 4 (packed) | ⭐ Easy with helpers! |
| `uint8_t` | `uint32_t` | 4 (packed) | ⭐ Easy with helpers! |

### int8/uint8 - Now Easy!

**Before** (manual intrinsics):
```cpp
CUVS_METRIC(old_way, {
    if constexpr (std::is_same_v<T, int8_t> && Veclen > 1) {
        auto diff = __vabsdiffs4(x, y);  // Must know this!
        acc = raft::dp4a(diff, diff, acc);  // And this!
    } else {
        auto diff = x - y;
        acc += diff * diff;
    }
})
```

**After** (with point wrapper):
```cpp
CUVS_METRIC(new_way, {
    acc += cuvs::udf::squared_diff(x, y);  // Just works!
})
```

## Power User Mode

For maximum control, you can still access raw storage and use intrinsics:

```cpp
CUVS_METRIC(power_user, {
    if constexpr (decltype(x)::is_packed()) {
        // Use SIMD intrinsics directly
        acc = __dp4a(x.raw(), y.raw(), acc);
    } else {
        acc += x.raw() * y.raw();
    }
})
```

## Error Handling

```cpp
try {
    search(res, params, index, queries, neighbors, distances);
} catch (const cuvs::udf::compilation_error& e) {
    std::cerr << "UDF compilation failed:\n" << e.what() << std::endl;
}
```

## Summary

| Feature | Benefit |
|---------|---------|
| **point wrapper** | Clean API, no raw AccT confusion |
| **Helper functions** | No intrinsics knowledge needed |
| **Auto Veclen deduction** | No template args in helpers |
| **Element access `x[i]`** | Custom logic without intrinsics |
| **Compile-time queries** | `is_packed()`, `size()` for branching |
| **Raw access `x.raw()`** | Power users can still use intrinsics |

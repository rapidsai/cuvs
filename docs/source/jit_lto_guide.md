# JIT LTO (Just-In-Time Link-Time Optimization) Guide

## Background

### What is JIT LTO?

[JIT LTO (Just-In-Time Link-Time Optimization)](https://developer.nvidia.com/blog/cuda-12-0-compiler-support-for-runtime-lto-using-nvjitlink-library/) is a CUDA compilation strategy that enables dynamic kernel compilation and linking at runtime. Instead of pre-compiling all possible kernel variants (which would result in an explosion of binary size), JIT LTO compiles kernel **fragments** separately and links them together on-demand when a specific kernel configuration is needed.

### Fragment Terminology

A **fragment** is a self-contained, compilable unit of CUDA code that can be linked with other fragments to form a complete kernel. In the JIT LTO system:

- **Entrypoint Fragment**: The main kernel function that serves as the entry point. This is always the `__global__` kernel function.
- **Device Function Fragments**: Separate fragments containing device functions (e.g., distance computations, filters, post-processing) that are called by the entrypoint kernel.
- **Fragment Key**: A unique identifier for a fragment, typically constructed from template parameters and configuration values.
- **Fatbin**: The compiled binary representation of a fragment, embedded in the executable.

The key advantage is that device functions can be compiled independently and reused across multiple kernel entrypoints, reducing compilation time and binary size.

### How It Works

1. **Build Time**: Fragments are compiled into fatbins and embedded in the executable.
2. **Runtime**: When a kernel needs to be launched:
   - The planner identifies which fragments are needed based on the configuration
   - Fragments are loaded from the embedded fatbins
   - Nvjitlink (Link-Time Optimization) links the fragments together
   - The linked kernel is cached and launched

## Walkthrough Example

Let's walk through creating a JIT LTO kernel system for a search kernel with templated device functions.

### Step 1: Define the Kernel and Device Functions

We start with a kernel that has templated device functions that we want to separate into fragments:

**`search_kernel.cuh`**:

```cpp
#pragma once

#include <cuda_runtime.h>

namespace example::detail {

// Device function for distance computation
template <typename T>
__device__ float compute_distance_euclidean(T a, T b) {
    T diff = a - b;
    return diff * diff;
}

template <typename T>
__device__ float compute_distance_inner_product(T a, T b) {
    return -a * b;  // Negative for max inner product search
}

// Device function for filtering
template <typename IdxT>
__device__ bool apply_filter_none(uint32_t query_id, IdxT node_id, void* filter_data) {
    return true;
}

template <typename IdxT>
__device__ bool apply_filter_bitset(uint32_t query_id, IdxT node_id, void* filter_data) {
    // Simplified - actual implementation would check bitset
    return true;
}

// Main kernel - will use generic extern device functions
template <typename T, typename OutT, typename IdxT, bool UseOptimizedPath, int Veclen>
__global__ void search_kernel(
    const T* dataset,
    const T* queries,
    IdxT* results,
    OutT* distances,  // Output distance type
    uint32_t num_queries,
    uint32_t dataset_size,
    void* filter_data) {

    uint32_t query_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_id >= num_queries) return;

    OutT best_dist = std::numeric_limits<OutT>::max();
    IdxT best_idx = 0;

    for (IdxT i = 0; i < dataset_size; ++i) {
        // Call generic extern device functions (implementations linked from fragments)
        if (!apply_filter<IdxT>(query_id, i, filter_data)) continue;

        OutT dist = static_cast<OutT>(compute_distance<T>(queries[query_id], dataset[i]));

        // Use optimized path if enabled
        if constexpr (UseOptimizedPath) {
            // Optimized implementation
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = i;
            }
        } else {
            // Standard implementation
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = i;
            }
        }
    }

    results[query_id] = best_idx;
    distances[query_id] = best_dist;
}

} // namespace example::detail
```

### Step 2: Create Device Function Fragments

We'll create separate header files for each device function variant. Each implements the generic function signature that the kernel expects:

**`compute_distance_euclidean.cuh`**:

```cpp
#pragma once

namespace example::detail {

// Implements the generic compute_distance function for euclidean distance
template <typename T>
__device__ float compute_distance(T a, T b) {
    T diff = a - b;
    return diff * diff;
}

} // namespace example::detail
```

**`compute_distance_inner_product.cuh`**:

```cpp
#pragma once

namespace example::detail {

// Implements the generic compute_distance function for inner product
template <typename T>
__device__ float compute_distance(T a, T b) {
    return -a * b;  // Negative for max inner product search
}

} // namespace example::detail
```

**`filter_none.cuh`**:

```cpp
#pragma once

namespace example::detail {

// Implements the generic apply_filter function for no filtering
template <typename IdxT>
__device__ bool apply_filter(uint32_t query_id, IdxT node_id, void* filter_data) {
    return true;
}

} // namespace example::detail
```

**`filter_bitset.cuh`**:

```cpp
#pragma once

namespace example::detail {

// Implements the generic apply_filter function for bitset filtering
template <typename IdxT>
__device__ bool apply_filter(uint32_t query_id, IdxT node_id, void* filter_data) {
    // Actual bitset implementation
    return true;
}

} // namespace example::detail
```

### Step 3: Create JSON Matrix Files

JSON matrix files define all the parameter combinations that need to be compiled. The build system uses these to generate `.cu` files from `.cu.in` templates.

**How JSON Cross-Product Works**:
- The build system computes a modified **Cartesian product** (cross-product) of all parameter combinations.
- **Leaf nodes** are the actual values. These can be strings, numbers, booleans, or `null`, but only strings should be used, even for numbers, for example ``"1"``.
- Related values can be grouped together in a dictionary consisting of single values. Any dictionary key in such a dictionary's ancestry will not be used in the final product, and should be prefixed with `_` to indicate that it is used only for grouping.
- Keys containing only leaf nodes will be used in the final product, and should not be prefixed with `_`.
- The matrix product algorithm will automatically warn if the proper naming convention (`_` prefix or not) is not followed.
- Each group expands to create multiple combinations, and all groups are cross-multiplied.

For example, if you have:
```json
{
  "_data_type": [{"data_type": "float"}, {"data_type": "half"}],
  "_index": [{"idx_type": "uint32_t"}, {"idx_type": "int64_t"}],
  "capacity": ["1", "2"]
}
```

This generates 2 × 2 × 2 = 8 combinations:
- `{data_type: "float", idx_type: "uint32_t", capacity: "1"}`
- `{data_type: "float", idx_type: "uint32_t", capacity: "2"}`
- `{data_type: "float", idx_type: "int64_t", capacity: "1"}`
- ... and so on

When a group contains nested arrays (like `veclen: ["1", "4"]`), those are also expanded within that group before the cross-product is computed.

#### `compute_distance_matrix.json`

```json
{
  "_distance_type": [
    {
      "distance_name": "euclidean",
      "header_file": "example/jit_lto_kernels/compute_distance_euclidean.cuh"
    },
    {
      "distance_name": "inner_product",
      "header_file": "example/jit_lto_kernels/compute_distance_inner_product.cuh"
    }
  ],
  "_data_type": [
    {
      "data_type": "float",
      "type_abbrev": "f"
    },
    {
      "data_type": "__half",
      "type_abbrev": "h"
    }
  ]
}
```

#### `filter_matrix.json`

```json
{
  "filter_name": [
    "filter_none",
    "filter_bitset"
  ],
  "_index": [
    {
      "idx_type": "uint32_t",
      "idx_abbrev": "ui"
    },
    {
      "idx_type": "int64_t",
      "idx_abbrev": "l"
    }
  ]
}
```

#### `search_kernel_matrix.json`

This example demonstrates conditional combinations: `OutT` can be `float` or `double` when `T` is `float`, but only `float` when `T` is `__half`.

```json
{
  "_data_type": [
    {
      "data_type": "float",
      "type_abbrev": "f",
      "_output_type": [
        {
          "out_type": "float",
          "out_abbrev": "f"
        },
        {
          "out_type": "double",
          "out_abbrev": "d"
        }
      ]
    },
    {
      "data_type": "__half",
      "type_abbrev": "h",
      "_output_type": [
        {
          "out_type": "float",
          "out_abbrev": "f"
        }
      ]
    }
  ],
  "_index": [
    {
      "idx_type": "uint32_t",
      "idx_abbrev": "ui"
    },
    {
      "idx_type": "int64_t",
      "idx_abbrev": "l"
    }
  ],
  "_optimized": [
    {
      "optimized_name": "optimized",
      "optimized_value": "true",
      "veclen": ["1", "4"]
    },
    {
      "optimized_name": "standard",
      "optimized_value": "false",
      "veclen": ["8", "16"]
    }
  ]
}
```

This generates 24 combinations (3 data/output type combinations × 2 index types × 4 optimized/veclen combinations):
- `float` + `float` + `uint32_t` + `optimized` + `veclen=1`
- `float` + `float` + `uint32_t` + `optimized` + `veclen=4`
- `float` + `float` + `uint32_t` + `standard` + `veclen=8`
- `float` + `float` + `uint32_t` + `standard` + `veclen=16`
- `float` + `double` + `uint32_t` + `optimized` + `veclen=1`
- `float` + `double` + `uint32_t` + `optimized` + `veclen=4`
- `float` + `double` + `uint32_t` + `standard` + `veclen=8`
- `float` + `double` + `uint32_t` + `standard` + `veclen=16`
- `__half` + `float` + `uint32_t` + `optimized` + `veclen=1`
- `__half` + `float` + `uint32_t` + `optimized` + `veclen=4`
- `__half` + `float` + `uint32_t` + `standard` + `veclen=8`
- `__half` + `float` + `uint32_t` + `standard` + `veclen=16`
- ... and the same with `int64_t` (total: 24 combinations)

### Step 4: Create `.cu.in` Template Files

The `.cu.in` files are templates that get instantiated for each combination in the JSON matrix. They contain explicit template instantiations.

#### `compute_distance_kernel.cu.in`

```text
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "@header_file@"

namespace example::detail {

// Instantiate the generic compute_distance device function template
// The specific implementation (euclidean or inner_product) comes from the header
template __device__ float compute_distance<@data_type@>(@data_type@, @data_type@);

} // namespace example::detail
```

#### `filter_kernel.cu.in`

```text
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "example/jit_lto_kernels/@filter_name@.cuh"

namespace example::detail {

// Instantiate the generic apply_filter device function template
// The specific implementation (filter_none or filter_bitset) comes from the header
template __device__ bool apply_filter<@idx_type@>(uint32_t, @idx_type@, void*);

} // namespace example::detail
```

#### Update `search_kernel.cuh` with Extern Declarations

The kernel header needs to declare generic extern device functions so the kernel code can call them. The specific implementations will be linked from fragments at runtime:

**`search_kernel.cuh`**:

```cpp
#pragma once

#include <cuda_runtime.h>

namespace example::detail {

// Forward declare generic extern device functions that will be linked from fragments
// The specific implementations (euclidean, inner_product, etc.) are resolved at link time
template <typename T>
extern __device__ float compute_distance(T, T);

template <typename IdxT>
extern __device__ bool apply_filter(uint32_t, IdxT, void*);

// Main kernel - uses generic extern device functions
template <typename T, typename OutT, typename IdxT, bool UseOptimizedPath, int Veclen>
__global__ void search_kernel(
    const T* dataset,
    const T* queries,
    IdxT* results,
    OutT* distances,  // Output distance type
    uint32_t num_queries,
    uint32_t dataset_size,
    void* filter_data) {

    uint32_t query_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_id >= num_queries) return;

    OutT best_dist = std::numeric_limits<OutT>::max();
    IdxT best_idx = 0;

    for (IdxT i = 0; i < dataset_size; ++i) {
        // Call generic extern device functions (specific implementations linked from fragments)
        if (!apply_filter<IdxT>(query_id, i, filter_data)) continue;

        OutT dist = static_cast<OutT>(compute_distance<T>(queries[query_id], dataset[i]));

        // Use optimized path if enabled
        if constexpr (UseOptimizedPath) {
            // Optimized implementation
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = i;
            }
        } else {
            // Standard implementation
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = i;
            }
        }
    }

    results[query_id] = best_idx;
    distances[query_id] = best_dist;
}

} // namespace example::detail
```

#### `search_kernel.cu.in`

The `.cu.in` file only contains the explicit template instantiation:

```text
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "example/jit_lto_kernels/search_kernel.cuh"

namespace example::detail {

// Instantiate the kernel template
template __global__ void search_kernel<@data_type@, @out_type@, @idx_type@, @optimized_value@, @veclen@>(
    const @data_type@*, const @data_type@*, @idx_type@*, @out_type@*,
    uint32_t, uint32_t, void*);

} // namespace example::detail
```

**Note**: The kernel uses generic function templates (`compute_distance<T>` and `apply_filter<IdxT>`) that are resolved at link time. The specific implementations (euclidean vs inner_product, filter_none vs filter_bitset) are provided by the fragments that get linked together.

### Step 5: Create `.cpp.in` Template Files for Embedding

The `.cpp.in` files register the compiled fatbins so they can be loaded at runtime. The fragment key used for registration is constructed as: `registerAlgorithm` constructor string + `"_"` + `make_fragment_key<Ts...>()`, where `Ts...` are the template parameters passed to `registerAlgorithm`.

**Important**: In the `.cpp.in` files (which become `.cpp` files), we use **tags** (like `tag_f`, `tag_h`) instead of real types (like `float`, `__half`) in the `registerAlgorithm` template parameters. This avoids including heavy headers that define the actual types, significantly improving compilation times. The tags are lightweight empty structs that serve only as compile-time identifiers.

#### `compute_distance_embedded.cpp.in`

```text
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/detail/jit_lto/RegisterKernelFragment.hpp>
#include <cuvs/detail/jit_lto/registration_tags.hpp>
#include "@embedded_header_file@"

using namespace example::detail;

namespace {

__attribute__((__constructor__)) void register_kernel()
{
  // Note: Fragment keys should include parameter names along with their values for better readability.
  registerAlgorithm<tag_@type_abbrev@>(
    "@distance_name@_data_@data_type@",
    embedded_fatbin,
    sizeof(embedded_fatbin));
}

}
```

#### `filter_embedded.cpp.in`

```text
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/detail/jit_lto/RegisterKernelFragment.hpp>
#include <cuvs/detail/jit_lto/registration_tags.hpp>
#include "@embedded_header_file@"

using namespace example::detail;

namespace {

__attribute__((__constructor__)) void register_kernel()
{
  registerAlgorithm<tag_@idx_abbrev@>(
    "@filter_name@_index_@idx_type@",
    embedded_fatbin,
    sizeof(embedded_fatbin));
}

}
```

#### `search_kernel_embedded.cpp.in`

```text
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/detail/jit_lto/RegisterKernelFragment.hpp>
#include <cuvs/detail/jit_lto/registration_tags.hpp>
#include "@embedded_header_file@"

using namespace example::detail;

namespace {

__attribute__((__constructor__)) void register_kernel()
{
  // Note: Non-type template parameters (like bool) cannot be handled by make_fragment_key,
  // so they must be included in the key string. Type information in template parameters
  // doesn't need to be repeated in the key.
  registerAlgorithm<tag_@type_abbrev@, tag_@out_abbrev@, tag_@idx_abbrev@>(
    "@optimized_name@_veclen_@veclen@",
    embedded_fatbin,
    sizeof(embedded_fatbin));
}

}
```

### Step 6: Create the Planner

The planner is responsible for:
1. Identifying which fragments are needed for a given configuration
2. Building a unique key for the fragment combination
3. Requesting the fragments from the fragment database
4. Linking them together to create a launchable kernel

**CRITICAL**: The fragment keys constructed in the planner methods must match **EXACTLY** with the keys used in the corresponding `.cpp.in` registration files. Any mismatch will result in runtime linking failures.

**`search_planner.hpp`**:

```cpp
#pragma once

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/MakeFragmentKey.hpp>
#include <cuvs/detail/jit_lto/registration_tags.hpp>
#include <string>

template <typename DataTag, typename OutTag, typename IndexTag>
struct SearchPlanner : AlgorithmPlanner {
  SearchPlanner(bool use_optimized = false, int veclen = 1)
    : AlgorithmPlanner("search_kernel",
                      (use_optimized ? "_optimized" : "_standard") + "_veclen_" + std::to_string(veclen) +
                      make_fragment_key<DataTag, OutTag, IndexTag>())
  {
  }

  void add_compute_distance_device_function(std::string distance_name)
  {
    // Build fragment key: distance_name + "_data_" + make_fragment_key<DataTag>()
    auto key = distance_name + "_data_";
    auto params = make_fragment_key<DataTag>();
    key += params;
    this->device_functions.push_back(key);
  }

  void add_filter_device_function(std::string filter_name)
  {
    // Build fragment key: filter_name + "_index_" + make_fragment_key<IndexTag>()
    auto key = filter_name + "_index_";
    auto params = make_fragment_key<IndexTag>();
    key += params;
    this->device_functions.push_back(key);
  }
};
```

### Step 7: Integrate with Code Path

Now we integrate the planner into the actual search function:

**`search_jit.cuh`**:

```cpp
#pragma once

#include "search_planner.hpp"
#include <cuvs/detail/jit_lto/registration_tags.hpp>
#include <raft/core/device_resources.hpp>

namespace example::detail {

// Type tag helpers
template <typename T>
constexpr auto get_data_type_tag() {
  if constexpr (std::is_same_v<T, float>) return tag_f{};
  if constexpr (std::is_same_v<T, __half>) return tag_h{};
}

template <typename IdxT>
constexpr auto get_idx_type_tag() {
  if constexpr (std::is_same_v<IdxT, uint32_t>) return tag_ui{};
  if constexpr (std::is_same_v<IdxT, int64_t>) return tag_l{};
}

template <typename T, typename OutT, typename IdxT>
void search_jit(
    raft::device_resources const& handle,
    const T* dataset,
    const T* queries,
    IdxT* results,
    OutT* distances,
    uint32_t num_queries,
    uint32_t dataset_size,
    std::string distance_type,  // "euclidean" or "inner_product"
    std::string filter_type,     // "filter_none" or "filter_bitset"
    bool use_optimized = false,  // Use optimized kernel path
    int veclen = 1,              // Vectorization length
    void* filter_data = nullptr) {

  // Type tag helpers for output type
  template <typename OutType>
  constexpr auto get_out_type_tag() {
    if constexpr (std::is_same_v<OutType, float>) return tag_f{};
    if constexpr (std::is_same_v<OutType, double>) return tag_d{};
  }

  // Create planner with type tags and boolean parameter
  // Note: The boolean is appended to the fragment key since make_fragment_key
  // cannot handle non-type template parameters
  auto planner = SearchPlanner<decltype(get_data_type_tag<T>()),
                               decltype(get_out_type_tag<OutT>()),
                               decltype(get_idx_type_tag<IdxT>())>(use_optimized);

  // Add required device function fragments
  // The DataTag is already provided to the planner template, so we just pass the distance name
  planner.add_compute_distance_device_function(distance_type);
  planner.add_filter_device_function(filter_type);

  // Get the launcher (this will build/link fragments if needed)
  auto launcher = planner.get_launcher();

  // Launch configuration
  dim3 block(256);
  dim3 grid((num_queries + block.x - 1) / block.x);

  // Launch the kernel - arguments are passed directly
  launcher->dispatch(
      raft::resource::get_cuda_stream(handle),
      grid,
      block,
      0,  // shared memory size
      dataset,
      queries,
      results,
      distances,
      num_queries,
      dataset_size,
      filter_data);
}

} // namespace example::detail
```

## Key Concepts

### Fragment Keys

Fragment keys uniquely identify fragments. They're constructed from:
- Template parameter types (using `make_fragment_key<>()`)
- Configuration values (e.g., "euclidean", "filter_none")
- Parameter values (e.g., veclen, capacity)

**Critical**: The fragment key must match **exactly** between:
- The registration in the `.cpp.in` file (the second argument to `registerAlgorithm`)
- The lookup in the planner's `device_functions` vector

**Key Construction**: The full fragment key is constructed as:
```
entrypoint_name + "_" + make_fragment_key<Ts...>
```

Where:
- `entrypoint_name` is the first argument to the `AlgorithmPlanner` constructor (e.g., `"search_kernel"`)
- `make_fragment_key<Ts...>` converts the template tag types to a string representation
- The `"_"` separator connects them

For device function fragments, the key is constructed as: `function_name + "_" + param_name + "_" + make_fragment_key<Tag>()` where `Tag` is the template parameter and `param_name` is a descriptive name for the parameter (e.g., `"data"`, `"index"`). Device functions are looked up separately from entrypoint kernels.

**Naming Convention**: Fragment keys should include parameter names along with their values for better readability. For example, use `"euclidean_data_float"` instead of `"euclidean_float"`, or `"filter_none_index_uint32_t"` instead of `"filter_none_uint32_t"`. This makes it clear what each value represents when debugging or inspecting fragment keys.

If the keys don't match exactly (including case, underscores, and order), the fragment will not be found at runtime and linking will fail.

**Important**: The fragment database matches fragments by both the template tags and the key string together. For device functions, the key string must include the type information (via `make_fragment_key`) to match what the planner constructs.

For example:
- In `compute_distance_embedded.cpp.in`: `registerAlgorithm<tag_f>("euclidean_data_float", ...)` - the key includes function name, parameter name, and type
- In `SearchPlanner::add_compute_distance_device_function()`: must produce `key = distance_name + "_data_" + make_fragment_key<DataTag>()` for lookup (e.g., `"euclidean_data_float"`)

**Non-Type Template Parameters**: For non-type template parameters (like `bool`, `int`, etc.), `make_fragment_key` cannot be used since it only works with types. Instead, prepend the value as a string directly to the key:
- In the planner constructor: `(use_optimized ? "_optimized" : "_standard") + "_veclen_" + std::to_string(veclen) + make_fragment_key<DataTag, OutTag, IndexTag>()` - this produces something like `"_optimized_veclen_1_f_f_ui"`
- In the registration: `"@optimized_name@_veclen_@veclen@"` - type information is in the template parameters, only the non-type parameter values (optimized/standard and veclen) are in the key

Any mismatch will result in a runtime error when trying to link the fragments.

### Registration Tags

Registration tags are type-safe identifiers used to organize fragments. They're typically empty structs:

```cpp
struct tag_f {};  // float
struct tag_h {};  // half
struct tag_ui {}; // uint32_t
struct tag_l {};  // int64_t
```

These tags are used in `registerAlgorithm<>()` to create a hierarchical organization of fragments.

**Why Tags Instead of Real Types?**: Using tags instead of real types (like `float`, `__half`) in the `.cpp.in` files avoids including heavy headers that define those types. This significantly improves compilation times since the generated `.cpp` files don't need to pull in CUDA headers, type definitions, or other dependencies. Tags are lightweight compile-time identifiers that don't require any runtime overhead or additional includes.

### AlgorithmLauncher

The `AlgorithmLauncher` is the runtime handle for a linked kernel. It:
- Holds a `cudaKernel_t` handle to the linked kernel
- Provides `call()` and `call_cooperative()` methods to launch the kernel
- Manages the lifetime of the `cudaLibrary_t` that contains the kernel

### Fragment Database

The fragment database is a global registry that:
- Stores all registered fragments (from `__attribute__((__constructor__))` functions)
- Allows lookup by fragment key
- Manages the linking process via NVRTCLTO

## Best Practices

1. **Minimize Includes**: JIT LTO fragments should have minimal includes, especially avoiding host-side headers. Extract device-only code into separate headers.

2. **Fragment Granularity**: Balance between too many small fragments (overhead) and too few large fragments (less reuse). Device functions that are reused across multiple kernels are good candidates for separate fragments.

3. **Naming Consistency**: Ensure fragment keys match exactly between registration and lookup. Use helper functions to construct keys consistently.

4. **Type Safety**: Use registration tags to provide compile-time type safety and avoid runtime string mismatches.

5. **Caching**: The `AlgorithmPlanner::get_launcher()` method caches linked kernels, so repeated calls with the same configuration are efficient.

## Example: IVF Flat

IVF Flat uses JIT LTO with:
- **Metric fragments**: Euclidean and inner product distance computations (16 fatbins)
- **Post-lambda fragments**: Identity, sqrt, and compose post-processing (3 fatbins)
- **Interleaved scan fragments**: Main search kernel with various configurations (320 fatbins)
- **Filter fragments**: None and bitset filters (2 fatbins)

**Total: 341 fatbins** that can be combined into many more kernel variants at runtime.

### Step 8: Integrate with CMake Build System

To integrate JIT LTO kernels into the CMake build system, add calls to `generate_jit_lto_kernels()` in your main `CMakeLists.txt` file (typically in `cpp/CMakeLists.txt`).

The `generate_jit_lto_kernels()` function (defined in `cmake/modules/generate_jit_lto_kernels.cmake`) takes:
- `NAME_FORMAT`: Format string for generated kernel names (using `@variable@` syntax)
- `MATRIX_JSON_FILE`: Path to the JSON matrix file
- `KERNEL_INPUT_FILE`: Path to the `.cu.in` template
- `EMBEDDED_INPUT_FILE`: Path to the `.cpp.in` template
- `OUTPUT_DIRECTORY`: Where generated files are placed
- `KERNEL_LINK_LIBRARIES`: Interface library with compilation settings

Call `generate_jit_lto_kernels()` once for each fragment type (compute_distance, filter, search_kernel, etc.). The function reads the JSON matrix, computes the cross-product of all combinations, generates `.cu` and `.cpp` files from the templates, compiles them into fatbins, and returns a list of generated source files that should be added to your JIT LTO library target.

See the CUVS `cpp/CMakeLists.txt` file for a complete example of how to set up the interface library, call `generate_jit_lto_kernels()` for each fragment type, and create the final library target.

## Summary

JIT LTO enables:
- **Reduced binary size**: Compile fragments once, combine many ways
- **Faster compilation**: Fragments compile independently
- **Runtime flexibility**: Link fragments on-demand based on configuration
- **Code reuse**: Device function fragments shared across kernels

The process involves:
1. Separating device functions into fragment headers
2. Creating JSON matrices defining parameter combinations
3. Creating `.cu.in` templates for explicit instantiations
4. Creating `.cpp.in` templates for fatbin registration
5. Creating a planner to manage fragment dependencies
6. Integrating the planner into the code path to launch kernels
7. **Adding CMake integration** to generate and compile all fragment variants

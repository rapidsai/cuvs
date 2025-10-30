# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

#!/usr/bin/env python3
"""
Generate CUDA kernel instantiation files for IVF-Flat interleaved scan.
This script generates kernel files programmatically based on type combinations.
"""

from pathlib import Path
import itertools


# Define the parameter space for kernel generation
CAPACITIES = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
ASCENDING_VALUES = [True, False]
COMPUTE_NORM_VALUES = [True, False]

# Data type configurations: (data_type, acc_type, veclens, type_abbrev, acc_abbrev)
# Each data type has veclen=1 and one optimized larger veclen
DATA_TYPE_CONFIGS = [
    ('float', 'float', [1, 4], 'f', 'f'),
    ('__half', '__half', [1, 8], 'h', 'h'),
    ('uint8_t', 'uint32_t', [1, 16], 'uc', 'ui'),
    ('int8_t', 'int32_t', [1, 16], 'sc', 'i'),
]

IDX_TYPE = 'int64_t'
IDX_TYPE_ABBREV = 'l'

# Metric configurations for device functions
METRIC_CONFIGS = [
    'euclidean',
    'inner_prod',
]

# Filter configurations
FILTER_CONFIGS = [
    'filter_none',
    'filter_bitset',
]

# Post lambda configurations
POST_LAMBDA_CONFIGS = [
    'post_identity',
    'post_sqrt',
    'post_compose',
]


def generate_kernel_combinations():
    """Generate all valid kernel parameter combinations."""
    kernels = []

    for data_type, acc_type, veclens, type_abbrev, acc_abbrev in DATA_TYPE_CONFIGS:
        for capacity, veclen, ascending, compute_norm in itertools.product(
            CAPACITIES, veclens, ASCENDING_VALUES, COMPUTE_NORM_VALUES
        ):
            kernels.append({
                'capacity': capacity,
                'veclen': veclen,
                'ascending': ascending,
                'compute_norm': compute_norm,
                'data_type': data_type,
                'acc_type': acc_type,
                'idx_type': IDX_TYPE,
                'type_abbrev': type_abbrev,
                'acc_abbrev': acc_abbrev,
                'idx_abbrev': IDX_TYPE_ABBREV,
            })

    return kernels


def generate_filename(params):
    """Generate filename from kernel parameters."""
    capacity = params['capacity']
    veclen = params['veclen']
    ascending = 'true' if params['ascending'] else 'false'
    compute_norm = 'true' if params['compute_norm'] else 'false'
    type_abbrev = params['type_abbrev']
    acc_abbrev = params['acc_abbrev']
    idx_abbrev = params['idx_abbrev']

    return f"interleaved_scan_kernel_{capacity}_{veclen}_{ascending}_{compute_norm}_{type_abbrev}_{acc_abbrev}_{idx_abbrev}.cu"


def generate_cuda_file_content(params):
    """Generate the content of a CUDA kernel instantiation file."""
    capacity = params['capacity']
    veclen = params['veclen']
    ascending = 'true' if params['ascending'] else 'false'
    compute_norm = 'true' if params['compute_norm'] else 'false'
    data_type = params['data_type']
    acc_type = params['acc_type']
    idx_type = params['idx_type']

    content = f"""/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// This file is auto-generated. Do not edit manually.

#ifdef BUILD_KERNEL

#include <neighbors/ivf_flat/ivf_flat_interleaved_scan_kernel.cuh>

namespace cuvs::neighbors::ivf_flat::detail {{

// Instantiate the kernel template
template __global__ void interleaved_scan_kernel<{capacity}, {veclen}, {ascending}, {compute_norm}, {data_type}, {acc_type}, {idx_type}>(
    const uint32_t, const {data_type}*, const uint32_t*, const {data_type}* const*, const uint32_t*,
    const uint32_t, const uint32_t, const uint32_t, const uint32_t, const uint32_t*, const uint32_t,
    {idx_type}* const* const, uint32_t*, {idx_type}, {idx_type}, uint32_t*, float*);

}}  // namespace cuvs::neighbors::ivf_flat::detail

#else

#include <cuvs/detail/jit_lto/RegisterKernelFragment.h>
#include <cuvs/detail/jit_lto/ivf_flat/interleaved_scan_tags.hpp>
#include "interleaved_scan_kernel_{capacity}_{veclen}_{ascending}_{compute_norm}_{params['type_abbrev']}_{params['acc_abbrev']}_{params['idx_abbrev']}.h"

using namespace cuvs::neighbors::ivf_flat::detail;

__attribute__((__constructor__)) static void register_kernel_{capacity}_{veclen}_{ascending}_{compute_norm}_{params['type_abbrev']}_{params['acc_abbrev']}_{params['idx_abbrev']}()
{{
  registerAlgorithm<tag_{params['type_abbrev']},
    tag_acc_{params['acc_abbrev']},
    tag_idx_{params['idx_abbrev']}>(
    "interleaved_scan_kernel_{capacity}_{veclen}_{ascending}_{compute_norm}",
    embedded_interleaved_scan_kernel_{capacity}_{veclen}_{ascending}_{compute_norm}_{params['type_abbrev']}_{params['acc_abbrev']}_{params['idx_abbrev']},
    sizeof(embedded_interleaved_scan_kernel_{capacity}_{veclen}_{ascending}_{compute_norm}_{params['type_abbrev']}_{params['acc_abbrev']}_{params['idx_abbrev']}));
}}

#endif
"""
    return content


def generate_metric_device_function_content(metric_name, veclen, data_type, acc_type):
    """Generate content for a metric device function file."""
    type_abbrev = {'float': 'f', '__half': 'h', 'uint8_t': 'uc', 'int8_t': 'sc'}[data_type]
    acc_abbrev = {'float': 'f', '__half': 'h', 'uint32_t': 'ui', 'int32_t': 'i'}[acc_type]

    if metric_name == 'euclidean':
        header_file = 'neighbors/ivf_flat/jit_lto_kernels/metric_euclidean_dist.cuh'
    else:  # inner_prod
        header_file = 'neighbors/ivf_flat/jit_lto_kernels/metric_inner_product.cuh'

    content = f"""/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// This file is auto-generated. Do not edit manually.

#ifdef BUILD_KERNEL

#include <{header_file}>

namespace cuvs::neighbors::ivf_flat::detail {{

// Instantiate the device function template
template __device__ void compute_dist<{veclen}, {data_type}, {acc_type}>({acc_type}&, {acc_type}, {acc_type});

}}  // namespace cuvs::neighbors::ivf_flat::detail

#else

#include <cuvs/detail/jit_lto/RegisterKernelFragment.h>
#include <cuvs/detail/jit_lto/ivf_flat/interleaved_scan_tags.hpp>
#include "metric_{metric_name}_{veclen}_{type_abbrev}_{acc_abbrev}.h"

using namespace cuvs::neighbors::ivf_flat::detail;

__attribute__((__constructor__)) static void register_metric_{metric_name}_{veclen}_{type_abbrev}_{acc_abbrev}()
{{
  registerAlgorithm<tag_{type_abbrev},
    tag_acc_{acc_abbrev}>(
    "{metric_name}_{veclen}",
    embedded_metric_{metric_name}_{veclen}_{type_abbrev}_{acc_abbrev},
    sizeof(embedded_metric_{metric_name}_{veclen}_{type_abbrev}_{acc_abbrev}));
}}

#endif
"""
    return content


def generate_filter_device_function_content(filter_name):
    """Generate content for a filter device function file."""
    if filter_name == 'filter_none':
        header_file = 'neighbors/ivf_flat/jit_lto_kernels/filter_none.cuh'
    else:  # filter_bitset
        header_file = 'neighbors/ivf_flat/jit_lto_kernels/filter_bitset.cuh'

    content = f"""/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// This file is auto-generated. Do not edit manually.

#ifdef BUILD_KERNEL

#include <{header_file}>

namespace cuvs::neighbors::ivf_flat::detail {{

// Instantiate the device function template
template __device__ bool sample_filter<int64_t>(int64_t* const* const, const uint32_t, const uint32_t, const uint32_t, uint32_t*, int64_t, int64_t);

}}  // namespace cuvs::neighbors::ivf_flat::detail

#else

#include <cuvs/detail/jit_lto/RegisterKernelFragment.h>
#include "{filter_name}.h"

__attribute__((__constructor__)) static void register_{filter_name}()
{{
  registerAlgorithm(
    "{filter_name}",
    embedded_{filter_name},
    sizeof(embedded_{filter_name}));
}}

#endif
"""
    return content


def generate_post_lambda_device_function_content(post_lambda_name):
    """Generate content for a post lambda device function file."""
    if post_lambda_name == 'post_identity':
        header_file = 'neighbors/ivf_flat/jit_lto_kernels/post_identity.cuh'
    elif post_lambda_name == 'post_sqrt':
        header_file = 'neighbors/ivf_flat/jit_lto_kernels/post_sqrt.cuh'
    else:  # post_compose
        header_file = 'neighbors/ivf_flat/jit_lto_kernels/post_compose.cuh'

    content = f"""/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// This file is auto-generated. Do not edit manually.

#ifdef BUILD_KERNEL

#include <{header_file}>

namespace cuvs::neighbors::ivf_flat::detail {{

// Instantiate the device function template
template __device__ float post_process<float>(float);

}}  // namespace cuvs::neighbors::ivf_flat::detail

#else

#include <cuvs/detail/jit_lto/RegisterKernelFragment.h>
#include "{post_lambda_name}.h"

__attribute__((__constructor__)) static void register_{post_lambda_name}()
{{
  registerAlgorithm(
    "{post_lambda_name}",
    embedded_{post_lambda_name},
    sizeof(embedded_{post_lambda_name}));
}}

#endif
"""
    return content


def generate_metric_device_functions(output_base_dir):
    """Generate all metric device function files."""
    metric_dir = output_base_dir / 'metric_device_functions'
    metric_dir.mkdir(parents=True, exist_ok=True)

    metric_files = []

    for metric_name in METRIC_CONFIGS:
        for data_type, acc_type, veclens, type_abbrev, acc_abbrev in DATA_TYPE_CONFIGS:
            for veclen in veclens:
                filename = f"metric_{metric_name}_{veclen}_{type_abbrev}_{acc_abbrev}.cu"
                filepath = metric_dir / filename

                content = generate_metric_device_function_content(metric_name, veclen, data_type, acc_type)

                # Only write if content has changed
                if not filepath.exists() or filepath.read_text() != content:
                    filepath.write_text(content)

                metric_files.append(filename)

    return metric_files


def generate_filter_device_functions(output_base_dir):
    """Generate all filter device function files."""
    filter_dir = output_base_dir / 'filter_device_functions'
    filter_dir.mkdir(parents=True, exist_ok=True)

    filter_files = []

    for filter_name in FILTER_CONFIGS:
        filename = f"{filter_name}.cu"
        filepath = filter_dir / filename

        content = generate_filter_device_function_content(filter_name)

        # Only write if content has changed
        if not filepath.exists() or filepath.read_text() != content:
            filepath.write_text(content)

        filter_files.append(filename)

    return filter_files


def generate_post_lambda_device_functions(output_base_dir):
    """Generate all post lambda device function files."""
    post_lambda_dir = output_base_dir / 'post_lambda_device_functions'
    post_lambda_dir.mkdir(parents=True, exist_ok=True)

    post_lambda_files = []

    for post_lambda_name in POST_LAMBDA_CONFIGS:
        filename = f"{post_lambda_name}.cu"
        filepath = post_lambda_dir / filename

        content = generate_post_lambda_device_function_content(post_lambda_name)

        # Only write if content has changed
        if not filepath.exists() or filepath.read_text() != content:
            filepath.write_text(content)

        post_lambda_files.append(filename)

    return post_lambda_files


def main():
    import sys

    # Get the script directory
    script_dir = Path(__file__).parent.absolute()

    # Output directory - use CMAKE_CURRENT_BINARY_DIR if provided, otherwise use source dir
    output_base_dir = Path(sys.argv[1]).absolute() if len(sys.argv) > 1 else script_dir

    # Kernel name - use provided name if available, otherwise default to "interleaved_scan"
    kernel_name = sys.argv[2] if len(sys.argv) > 2 else "interleaved_scan"

    output_dir = output_base_dir / 'interleaved_scan_kernels'
    output_dir.mkdir(parents=True, exist_ok=True)

    kernels = generate_kernel_combinations()

    # Generate kernel files
    generated_files = []
    for params in kernels:
        filename = generate_filename(params)
        filepath = output_dir / filename

        content = generate_cuda_file_content(params)

        # Only write if content has changed
        if not filepath.exists() or filepath.read_text() != content:
            filepath.write_text(content)

        generated_files.append(filename)

    # Generate metric device function files
    metric_files = generate_metric_device_functions(output_base_dir)

    # Generate filter device function files
    filter_files = generate_filter_device_functions(output_base_dir)

    # Generate post lambda device function files
    post_lambda_files = generate_post_lambda_device_functions(output_base_dir)

    # Generate CMake file listing all generated files
    cmake_file = output_base_dir / f'{kernel_name}.cmake'

    cmake_content = "# Auto-generated file listing all kernel and device function files\n\n"

    # Set relative path lists
    cmake_content += "set(INTERLEAVED_SCAN_KERNEL_FILES\n"
    for filename in sorted(generated_files):
        cmake_content += f"  generated_kernels/interleaved_scan_kernels/{filename}\n"
    cmake_content += ")\n\n"

    cmake_content += "set(METRIC_DEVICE_FUNCTION_FILES\n"
    for filename in sorted(metric_files):
        cmake_content += f"  generated_kernels/metric_device_functions/{filename}\n"
    cmake_content += ")\n\n"

    cmake_content += "set(FILTER_DEVICE_FUNCTION_FILES\n"
    for filename in sorted(filter_files):
        cmake_content += f"  generated_kernels/filter_device_functions/{filename}\n"
    cmake_content += ")\n\n"

    cmake_content += "set(POST_LAMBDA_DEVICE_FUNCTION_FILES\n"
    for filename in sorted(post_lambda_files):
        cmake_content += f"  generated_kernels/post_lambda_device_functions/{filename}\n"
    cmake_content += ")\n\n"

    # Add logic to prepend CMAKE_CURRENT_BINARY_DIR and set variables to PARENT_SCOPE
    cmake_content += f"""# Prepend the binary directory path to all kernel files
set(FULL_PATH_KERNEL_FILES)
foreach(kernel_file ${{INTERLEAVED_SCAN_KERNEL_FILES}})
  list(APPEND FULL_PATH_KERNEL_FILES ${{CMAKE_CURRENT_BINARY_DIR}}/${{kernel_file}})
endforeach()

# Prepend the binary directory path to all metric device function files
set(FULL_PATH_METRIC_FILES)
foreach(metric_file ${{METRIC_DEVICE_FUNCTION_FILES}})
  list(APPEND FULL_PATH_METRIC_FILES ${{CMAKE_CURRENT_BINARY_DIR}}/${{metric_file}})
endforeach()

# Prepend the binary directory path to all filter device function files
set(FULL_PATH_FILTER_FILES)
foreach(filter_file ${{FILTER_DEVICE_FUNCTION_FILES}})
  list(APPEND FULL_PATH_FILTER_FILES ${{CMAKE_CURRENT_BINARY_DIR}}/${{filter_file}})
endforeach()

# Prepend the binary directory path to all post lambda device function files
set(FULL_PATH_POST_LAMBDA_FILES)
foreach(post_lambda_file ${{POST_LAMBDA_DEVICE_FUNCTION_FILES}})
  list(APPEND FULL_PATH_POST_LAMBDA_FILES ${{CMAKE_CURRENT_BINARY_DIR}}/${{post_lambda_file}})
endforeach()

# Return the lists to parent scope
set(INTERLEAVED_SCAN_KERNEL_FILES
    ${{FULL_PATH_KERNEL_FILES}}
    PARENT_SCOPE
)
set(METRIC_DEVICE_FUNCTION_FILES
    ${{FULL_PATH_METRIC_FILES}}
    PARENT_SCOPE
)
set(FILTER_DEVICE_FUNCTION_FILES
    ${{FULL_PATH_FILTER_FILES}}
    PARENT_SCOPE
)
set(POST_LAMBDA_DEVICE_FUNCTION_FILES
    ${{FULL_PATH_POST_LAMBDA_FILES}}
    PARENT_SCOPE
)
set(INTERLEAVED_SCAN_KERNELS_TARGET
    generate_{kernel_name}_kernels_target
    PARENT_SCOPE
)
"""

    # Only write if content has changed
    if not cmake_file.exists() or cmake_file.read_text() != cmake_content:
        cmake_file.write_text(cmake_content)


if __name__ == '__main__':
    main()

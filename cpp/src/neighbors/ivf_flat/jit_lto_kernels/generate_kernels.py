# =============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================


#!/usr/bin/env python3
"""
Simplified script to generate CUDA kernel files for interleaved_scan_kernel instantiations.
Reads from interleaved_scan_kernels.txt and generates individual .cu files.
"""

import re
import os
from pathlib import Path


def parse_template_parameters(template_str):
    """Parse template parameters from a template string with nested templates."""
    params = []
    current_param = ''
    depth = 0

    for char in template_str:
        if char == '<':
            depth += 1
        elif char == '>':
            depth -= 1
        elif char == ',' and depth == 0:
            params.append(current_param.strip())
            current_param = ''
            continue
        current_param += char

    if current_param:
        params.append(current_param.strip())

    return params


def get_type_abbreviation(type_str):
    """Get abbreviation for type names."""
    type_map = {
        'float': 'f',
        '__half': 'h',
        'unsigned char': 'uc',
        'signed char': 'sc',
        'unsigned int': 'ui',
        'int': 'i',
        'long': 'l'
    }
    return type_map.get(type_str, type_str)


def get_filter_abbreviation(filter_str):
    """Get abbreviation for filter types."""
    if 'none_sample_filter' in filter_str:
        return 'n'
    elif 'bitset_filter' in filter_str:
        return 'b'
    return 'unknown'


def get_distance_abbreviation(dist_str):
    """Get abbreviation for distance metric types."""
    if 'inner_prod_dist' in dist_str:
        match = re.search(r'inner_prod_dist<(\d+),', dist_str)
        if match:
            return f'inner_{match.group(1)}'
    elif 'euclidean_dist' in dist_str:
        match = re.search(r'euclidean_dist<(\d+),', dist_str)
        if match:
            return f'euclidean_{match.group(1)}'
    return 'unknown'


def get_final_op_abbreviation(op_str):
    """Get abbreviation for final operator types."""
    if 'identity_op' in op_str:
        return 'id'
    elif 'sqrt_op' in op_str:
        return 'sqrt'
    elif 'compose_op' in op_str:
        return 'compose'
    return 'unknown'


def generate_filename(params):
    """Generate filename from template parameters (WITHOUT metric, filter, and post lambda)."""
    # params[0]: Capacity (numeric)
    # params[1]: Veclen (numeric)
    # params[2]: Ascending (bool)
    # params[3]: ComputeNorm (bool)
    # params[4]: T (type)
    # params[5]: AccT (type)
    # params[6]: IdxT (type)
    # params[7]: FilterT (filter type - EXCLUDED from filename)
    # params[8]: Lambda/MetricTag (metric type - EXCLUDED from filename)
    # params[9]: PostLambda (final operator - EXCLUDED from filename)

    parts = [
        params[0],  # Capacity
        params[1],  # Veclen
        params[2],  # Ascending
        params[3],  # ComputeNorm
        get_type_abbreviation(params[4]),  # T
        get_type_abbreviation(params[5]),  # AccT
        get_type_abbreviation(params[6]),  # IdxT
        # params[7] EXCLUDED - filter
        # params[8] EXCLUDED - metric
        # params[9] EXCLUDED - post lambda
    ]

    return f"interleaved_scan_kernel_{'_'.join(parts)}.cu"


def generate_register_function_name(params):
    """Generate the registration function name from template parameters (WITHOUT metric, filter, and post lambda)."""
    parts = [
        params[0],  # Capacity
        params[1],  # Veclen
        params[2],  # Ascending
        params[3],  # ComputeNorm
        get_type_abbreviation(params[4]),  # T
        get_type_abbreviation(params[5]),  # AccT
        get_type_abbreviation(params[6]),  # IdxT
        # params[7] EXCLUDED - filter
        # params[8] EXCLUDED - metric
        # params[9] EXCLUDED - post lambda
    ]

    return f"interleaved_scan_kernel_{'_'.join(parts)}"


def param_to_tag(param_index, param_value, all_params):
    """Convert a parameter to its corresponding tag type.

    param_index: Index of the parameter (0-9)
    param_value: The actual parameter value (C++ type string)
    all_params: All 10 parameters (needed for templated tags)
    """
    # Data type (param 4: T)
    if param_index == 4:
        type_map = {
            'float': 'tag_float',
            '__half': 'tag_half',
            'int8_t': 'tag_int8',
            'uint8_t': 'tag_uint8',
            'signed char': 'tag_int8',
            'unsigned char': 'tag_uint8'
        }
        return type_map.get(param_value, param_value)

    # Accumulator type (param 5: AccT)
    elif param_index == 5:
        acc_map = {
            'float': 'tag_acc_float',
            '__half': 'tag_acc_half',
            'int32_t': 'tag_acc_int32',
            'uint32_t': 'tag_acc_uint32',
            'int': 'tag_acc_int32',
            'unsigned int': 'tag_acc_uint32',
            'signed int': 'tag_acc_int32'
        }
        return acc_map.get(param_value, param_value)

    # Index type (param 6: IdxT) - always int64_t
    elif param_index == 6:
        return 'tag_idx_int64'

    # Sample filter type (param 7: IvfSampleFilterT)
    elif param_index == 7:
        # Get the IdxT tag
        idx_tag = param_to_tag(6, all_params[6], all_params)

        # Determine filter implementation tag
        if 'bitset_filter' in param_value:
            filter_impl_tag = 'tag_filter_bitset_impl'
        elif 'none_sample_filter' in param_value:
            filter_impl_tag = 'tag_filter_none_impl'
        else:
            filter_impl_tag = 'tag_filter_none_impl'

        # Return templated tag_filter with tag types
        return f'tag_filter<{idx_tag}, {filter_impl_tag}>'

    # Distance metric (param 8: Lambda)
    elif param_index == 8:
        # Extract veclen from the Lambda type
        veclen_match = re.search(r'<(\d+),', param_value)
        veclen = veclen_match.group(1) if veclen_match else all_params[1]

        # Get tags for T and AccT
        T_tag = param_to_tag(4, all_params[4], all_params)
        AccT_tag = param_to_tag(5, all_params[5], all_params)

        # Return templated tag based on metric type
        if 'euclidean_dist' in param_value:
            return f'tag_metric_euclidean<{veclen}, {T_tag}, {AccT_tag}>'
        elif 'inner_prod_dist' in param_value:
            return f'tag_metric_inner_product<{veclen}, {T_tag}, {AccT_tag}>'
        return param_value

    # Post-processing lambda (param 9: PostLambda)
    elif param_index == 9:
        if 'identity_op' in param_value:
            return 'tag_post_identity'
        elif 'sqrt_op' in param_value:
            return 'tag_post_sqrt'
        elif 'compose_op' in param_value:
            return 'tag_post_compose'
        return param_value

    return param_value


def generate_cuda_file_content(params):
    """Generate the content of a CUDA kernel file."""
    filename = generate_register_function_name(params)
    embedded_var_name = f"embedded_{filename}"

    # The kernel now has 8 template parameters (removed MetricTag and FilterT)
    # params[0-3]: Capacity, Veclen, Ascending, ComputeNorm
    # params[4]: T (data type)
    # params[5]: AccT (accumulator type)
    # params[6]: IdxT (index type)
    # params[7]: IvfSampleFilterT (filter type - NOT used in template anymore)
    # params[8]: Lambda (metric - NOT used in template anymore)
    # params[9]: PostLambda (post-processing operator)

    # Template parameters without MetricTag, FilterT, and PostLambda (params 0-6)
    template_params_list = params[0:7]
    template_params = ', '.join(template_params_list)

    # Convert params 4-6 to tag types for registerAlgorithm (NO metric/filter/postlambda tags)
    tag_params = [param_to_tag(i, params[i], params) for i in [4, 5, 6]]
    register_template_params = ', '.join(tag_params)

    # Create the string parameter with first four params (Capacity, Veclen, Ascending, ComputeNorm)
    string_param = f"interleaved_scan_kernel_{params[0]}_{params[1]}_{params[2]}_{params[3]}"

    # Function parameters for the kernel instantiation (updated signature - PostLambda removed)
    content = f"""/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef BUILD_KERNEL

#include <neighbors/ivf_flat/ivf_flat_interleaved_scan_kernel.cuh>

namespace cuvs::neighbors::ivf_flat::detail {{

template __global__ void interleaved_scan_kernel<{template_params}>(unsigned int, {params[4]} const*, unsigned int const*, {params[4]} const* const*, unsigned int const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int const*, unsigned int, {params[6]}* const* const, unsigned int*, {params[6]}, {params[6]}, unsigned int*, float*);

}}  // namespace cuvs::neighbors::ivf_flat::detail

#else

#include "{filename}.h"
#include <cuvs/detail/jit_lto/RegisterKernelFragment.h>
#include <neighbors/ivf_flat/jit_lto_kernels/interleaved_scan_tags.hpp>

__attribute__((__constructor__)) static void register_{filename}()
{{
using namespace cuvs::neighbors::ivf_flat::detail;
registerAlgorithm<
    {register_template_params}>("{string_param}",
            {embedded_var_name},
            sizeof({embedded_var_name}));
}}

#endif
"""

    return content


def generate_metric_device_function_content(metric_name, veclen, data_type, acc_type):
    """Generate content for a metric device function file."""
    # Map types to their tag equivalents
    # Mapping for data types (T)
    data_type_to_tag = {
        'float': 'tag_float',
        '__half': 'tag_half',
        'int8_t': 'tag_int8',
        'uint8_t': 'tag_uint8',
    }

    # Mapping for accumulator types (AccT)
    acc_type_to_tag = {
        'float': 'tag_acc_float',
        '__half': 'tag_acc_half',
        'int32_t': 'tag_acc_int32',
        'uint32_t': 'tag_acc_uint32',
    }

    # Get abbreviated names for filename
    type_abbrev = {
        'float': 'f',
        '__half': 'h',
        'int8_t': 'i8',
        'uint8_t': 'u8',
        'int32_t': 'i32',
        'uint32_t': 'u32',
    }

    data_tag = data_type_to_tag.get(data_type, data_type)
    acc_tag = acc_type_to_tag.get(acc_type, acc_type)

    # Determine which header to include and implementation struct based on metric
    if metric_name == 'euclidean':
        header_file = 'neighbors/ivf_flat/jit_lto_kernels/metric_euclidean_dist.cuh'
        metric_impl = 'euclidean_dist'
    elif metric_name == 'inner_prod':
        header_file = 'neighbors/ivf_flat/jit_lto_kernels/metric_inner_product.cuh'
        metric_impl = 'inner_prod_dist'
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    content = f"""/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef BUILD_KERNEL

#include "{header_file}"

namespace cuvs::neighbors::ivf_flat::detail {{

template __device__ void compute_dist<{veclen}, {data_type}, {acc_type}>({acc_type}&, {acc_type}, {acc_type});

}}  // namespace cuvs::neighbors::ivf_flat::detail

#else

#include "{metric_name}_{veclen}_{type_abbrev[data_type]}_{type_abbrev[acc_type]}.h"
#include <cuvs/detail/jit_lto/RegisterKernelFragment.h>
#include <neighbors/ivf_flat/jit_lto_kernels/interleaved_scan_tags.hpp>

__attribute__((__constructor__)) static void register_{metric_name}_{veclen}_{type_abbrev[data_type]}_{type_abbrev[acc_type]}()
{{
using namespace cuvs::neighbors::ivf_flat::detail;
registerAlgorithm<{data_tag}, {acc_tag}>("{metric_name}_{veclen}",
            embedded_{metric_name}_{veclen}_{type_abbrev[data_type]}_{type_abbrev[acc_type]},
            sizeof(embedded_{metric_name}_{veclen}_{type_abbrev[data_type]}_{type_abbrev[acc_type]}));
}}

#endif
"""
    return content


def generate_filter_device_function_content(filter_name):
    """Generate content for a filter device function file."""
    # Determine which header to include based on filter name
    if filter_name == 'filter_none':
        header_file = 'neighbors/ivf_flat/jit_lto_kernels/filter_none.cuh'
    elif filter_name == 'filter_bitset':
        header_file = 'neighbors/ivf_flat/jit_lto_kernels/filter_bitset.cuh'
    else:
        raise ValueError(f"Unknown filter: {filter_name}")

    content = f"""/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef BUILD_KERNEL

#include "{header_file}"

namespace cuvs::neighbors::ivf_flat::detail {{

template __device__ bool sample_filter(int64_t* const* const inds_ptrs, const uint32_t query_ix, const uint32_t cluster_ix, const uint32_t sample_ix, uint32_t* bitset_ptr, int64_t bitset_len, int64_t original_nbits);

}}  // namespace cuvs::neighbors::ivf_flat::detail

#else

#include "{filter_name}.h"
#include <cuvs/detail/jit_lto/RegisterKernelFragment.h>
#include <neighbors/ivf_flat/jit_lto_kernels/interleaved_scan_tags.hpp>

__attribute__((__constructor__)) static void register_{filter_name}()
{{
using namespace cuvs::neighbors::ivf_flat::detail;
registerAlgorithm("{filter_name}",
            embedded_{filter_name},
            sizeof(embedded_{filter_name}));
}}

#endif
"""
    return content


def generate_metric_device_functions(script_dir, output_base_dir):
    """Generate all metric device function files."""
    # Define all combinations we need
    # Based on the kernel signatures, we have:
    # - Veclen: 1, 2, 4, 8, 16
    # - Data types: float, __half, int8_t, uint8_t
    # - Acc types: float (for float), __half (for __half), int32_t (for int8_t), uint32_t (for uint8_t)
    # - Metrics: euclidean, inner_prod

    type_combinations = [
        ('float', 'float'),
        ('__half', '__half'),
        ('int8_t', 'int32_t'),
        ('uint8_t', 'uint32_t'),
    ]

    veclens = [1, 2, 4, 8, 16]
    metrics = ['euclidean', 'inner_prod']

    output_dir = output_base_dir / 'metric_device_functions'
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []

    type_abbrev = {
        'float': 'f',
        '__half': 'h',
        'int8_t': 'i8',
        'uint8_t': 'u8',
        'int32_t': 'i32',
        'uint32_t': 'u32',
    }

    for metric in metrics:
        for veclen in veclens:
            for data_type, acc_type in type_combinations:
                filename = f"{metric}_{veclen}_{type_abbrev[data_type]}_{type_abbrev[acc_type]}.cu"
                file_content = generate_metric_device_function_content(metric, veclen, data_type, acc_type)

                # Write file only if it doesn't exist or content has changed
                output_file = output_dir / filename
                should_write = True
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        existing_content = f.read()
                    should_write = (existing_content != file_content)

                if should_write:
                    with open(output_file, 'w') as f:
                        f.write(file_content)

                generated_files.append(filename)

    print(f"Generated {len(generated_files)} metric device function files")
    return generated_files


def generate_filter_device_functions(script_dir, output_base_dir):
    """Generate all filter device function files."""
    filters = ['filter_none', 'filter_bitset']

    output_dir = output_base_dir / 'filter_device_functions'
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []

    for filter_name in filters:
        filename = f"{filter_name}.cu"
        file_content = generate_filter_device_function_content(filter_name)

        # Write file only if it doesn't exist or content has changed
        output_file = output_dir / filename
        should_write = True
        if output_file.exists():
            with open(output_file, 'r') as f:
                existing_content = f.read()
            should_write = (existing_content != file_content)

        if should_write:
            with open(output_file, 'w') as f:
                f.write(file_content)

        generated_files.append(filename)

    print(f"Generated {len(generated_files)} filter device function files")
    return generated_files


def generate_post_lambda_device_function_content(post_lambda_name):
    """Generate content for a post lambda device function file."""
    # Determine which header to include based on post lambda name
    if post_lambda_name == 'post_identity':
        header_file = 'neighbors/ivf_flat/jit_lto_kernels/post_identity.cuh'
    elif post_lambda_name == 'post_sqrt':
        header_file = 'neighbors/ivf_flat/jit_lto_kernels/post_sqrt.cuh'
    elif post_lambda_name == 'post_compose':
        header_file = 'neighbors/ivf_flat/jit_lto_kernels/post_compose.cuh'
    else:
        raise ValueError(f"Unknown post lambda: {post_lambda_name}")

    content = f"""/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef BUILD_KERNEL

#include "{header_file}"

namespace cuvs::neighbors::ivf_flat::detail {{

template __device__ float post_process(float val);

}}  // namespace cuvs::neighbors::ivf_flat::detail

#else

#include "{post_lambda_name}.h"
#include <cuvs/detail/jit_lto/RegisterKernelFragment.h>
#include <neighbors/ivf_flat/jit_lto_kernels/interleaved_scan_tags.hpp>

__attribute__((__constructor__)) static void register_{post_lambda_name}()
{{
using namespace cuvs::neighbors::ivf_flat::detail;
registerAlgorithm("{post_lambda_name}",
            embedded_{post_lambda_name},
            sizeof(embedded_{post_lambda_name}));
}}

#endif
"""
    return content


def generate_post_lambda_device_functions(script_dir, output_base_dir):
    """Generate all post lambda device function files."""
    post_lambdas = ['post_identity', 'post_sqrt', 'post_compose']

    output_dir = output_base_dir / 'post_lambda_device_functions'
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []

    for post_lambda_name in post_lambdas:
        filename = f"{post_lambda_name}.cu"
        file_content = generate_post_lambda_device_function_content(post_lambda_name)

        # Write file only if it doesn't exist or content has changed
        output_file = output_dir / filename
        should_write = True
        if output_file.exists():
            with open(output_file, 'r') as f:
                existing_content = f.read()
            should_write = (existing_content != file_content)

        if should_write:
            with open(output_file, 'w') as f:
                f.write(file_content)

        generated_files.append(filename)

    print(f"Generated {len(generated_files)} post lambda device function files")
    return generated_files


def main():
    import sys

    # Get the script directory to find the kernels file
    script_dir = Path(__file__).parent.absolute()

    # Read the kernels file (in the same directory as this script)
    kernels_file = script_dir / 'interleaved_scan_kernels.txt'
    if not kernels_file.exists():
        print(f"Error: {kernels_file} not found!")
        return

    with open(kernels_file, 'r') as f:
        lines = f.readlines()

    # Output directory - use command line argument if provided, otherwise use source dir
    if len(sys.argv) > 1:
        output_base_dir = Path(sys.argv[1]).absolute()
    else:
        output_base_dir = script_dir

    output_dir = output_base_dir / 'interleaved_scan_kernels'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse all kernels and generate files
    # Use a dict to deduplicate by filename (since we exclude metric from filename)
    unique_kernels = {}

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        # Extract the full template from the function signature
        start = line.find('interleaved_scan_kernel<')
        if start == -1:
            continue

        start += len('interleaved_scan_kernel<')
        depth = 1
        end = start

        while depth > 0 and end < len(line):
            if line[end] == '<':
                depth += 1
            elif line[end] == '>':
                depth -= 1
            end += 1

        template_str = line[start:end-1]
        params = parse_template_parameters(template_str)

        if len(params) != 10:
            print(f"Warning: Line {line_num} has {len(params)} parameters, expected 10")
            continue

        # Generate filename and content
        filename = generate_filename(params)

        # Only generate if we haven't seen this filename yet (deduplication)
        if filename not in unique_kernels:
            file_content = generate_cuda_file_content(params)
            unique_kernels[filename] = file_content

    # Write all unique kernel files
    generated_files = []
    for filename, file_content in unique_kernels.items():
        output_file = output_dir / filename
        should_write = True
        if output_file.exists():
            with open(output_file, 'r') as f:
                existing_content = f.read()
            should_write = (existing_content != file_content)

        if should_write:
            with open(output_file, 'w') as f:
                f.write(file_content)

        generated_files.append(filename)

        if len(generated_files) % 100 == 0:
            print(f"Generated {len(generated_files)} files...")

    print(f"\nGenerated {len(generated_files)} CUDA kernel files")

    # Generate metric device function files
    metric_files = generate_metric_device_functions(script_dir, output_base_dir)

    # Generate filter device function files
    filter_files = generate_filter_device_functions(script_dir, output_base_dir)

    # Generate post lambda device function files
    post_lambda_files = generate_post_lambda_device_functions(script_dir, output_base_dir)

    # Generate CMake file with all filenames
    # CMake file goes to the binary directory
    cmake_dir = output_base_dir
    cmake_dir.mkdir(parents=True, exist_ok=True)
    cmake_file = cmake_dir / 'interleaved_scan.cmake'

    # Generate CMake content
    # Paths are now relative to CMAKE_CURRENT_BINARY_DIR
    cmake_content = "# Auto-generated list of interleaved scan kernel files\n"
    cmake_content += "# Generated by generate_kernels.py\n\n"
    cmake_content += "set(INTERLEAVED_SCAN_KERNEL_FILES\n"
    for filename in sorted(generated_files):
        cmake_content += f"  generated_kernels/interleaved_scan_kernels/{filename}\n"
    cmake_content += ")\n\n"

    # Add metric device function files
    cmake_content += "set(METRIC_DEVICE_FUNCTION_FILES\n"
    for filename in sorted(metric_files):
        cmake_content += f"  generated_kernels/metric_device_functions/{filename}\n"
    cmake_content += ")\n\n"

    # Add filter device function files
    cmake_content += "set(FILTER_DEVICE_FUNCTION_FILES\n"
    for filename in sorted(filter_files):
        cmake_content += f"  generated_kernels/filter_device_functions/{filename}\n"
    cmake_content += ")\n\n"

    # Add post lambda device function files
    cmake_content += "set(POST_LAMBDA_DEVICE_FUNCTION_FILES\n"
    for filename in sorted(post_lambda_files):
        cmake_content += f"  generated_kernels/post_lambda_device_functions/{filename}\n"
    cmake_content += ")\n"

    # Only write if content has changed
    should_write_cmake = True
    if cmake_file.exists():
        with open(cmake_file, 'r') as f:
            existing_cmake = f.read()
        should_write_cmake = (existing_cmake != cmake_content)

    if should_write_cmake:
        with open(cmake_file, 'w') as f:
            f.write(cmake_content)
        print(f"Updated CMake file: {cmake_file}")
    else:
        print(f"CMake file unchanged: {cmake_file}")

if __name__ == '__main__':
    main()

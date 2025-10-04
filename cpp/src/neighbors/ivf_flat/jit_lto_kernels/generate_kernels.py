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
    """Generate filename from template parameters."""
    # params[0]: kBlockSize (numeric)
    # params[1]: VecLen (numeric)
    # params[2]: kManageLocalTopK (bool)
    # params[3]: kPrecompBaseDiff (bool)
    # params[4]: T (type)
    # params[5]: AccT (type)
    # params[6]: IdxT (type)
    # params[7]: FilterT (filter type)
    # params[8]: DistanceT (distance metric)
    # params[9]: FinalLambda (final operator)

    parts = [
        params[0],  # kBlockSize
        params[1],  # VecLen
        params[2],  # kManageLocalTopK
        params[3],  # kPrecompBaseDiff
        get_type_abbreviation(params[4]),  # T
        get_type_abbreviation(params[5]),  # AccT
        get_type_abbreviation(params[6]),  # IdxT
        get_filter_abbreviation(params[7]),  # FilterT
        get_distance_abbreviation(params[8]),  # DistanceT
        get_final_op_abbreviation(params[9])  # FinalLambda
    ]

    return f"interleaved_scan_kernel_{'_'.join(parts)}.cu"


def generate_register_function_name(params):
    """Generate the registration function name from template parameters."""
    parts = [
        params[0],  # kBlockSize
        params[1],  # VecLen
        params[2],  # kManageLocalTopK
        params[3],  # kPrecompBaseDiff
        get_type_abbreviation(params[4]),  # T
        get_type_abbreviation(params[5]),  # AccT
        get_type_abbreviation(params[6]),  # IdxT
        get_filter_abbreviation(params[7]),  # FilterT
        get_distance_abbreviation(params[8]),  # DistanceT
        get_final_op_abbreviation(params[9])  # FinalLambda
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

    # Format template parameters for the template instantiation (all 10 params)
    template_params = ', '.join(params)

    # Convert params 4-9 to tag types for registerAlgorithm
    tag_params = [param_to_tag(i, params[i], params) for i in range(4, 10)]
    register_template_params = ', '.join(tag_params)

    # Create the string parameter with first four params (Capacity, Veclen, Ascending, ComputeNorm)
    string_param = f"interleaved_scan_kernel_{params[0]}_{params[1]}_{params[2]}_{params[3]}"

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

#include "../../ivf_flat_interleaved_scan.cuh"

template __global__ void cuvs::neighbors::ivf_flat::detail::interleaved_scan_kernel<{template_params}>({params[8]}, {params[9]}, unsigned int, {params[4]} const*, unsigned int const*, {params[4]} const* const*, unsigned int const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int const*, unsigned int, {params[7]}, unsigned int*, float*);

#else

#include "{filename}.h"
#include <cuvs/detail/jit_lto/RegisterKernelFragment.h>
#include "../interleaved_scan_tags.hpp"

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


def main():
    # Get the script directory to find the kernels file
    script_dir = Path(__file__).parent.absolute()

    # Read the kernels file (in the same directory as this script)
    kernels_file = script_dir / 'interleaved_scan_kernels.txt'
    if not kernels_file.exists():
        print(f"Error: {kernels_file} not found!")
        return

    with open(kernels_file, 'r') as f:
        lines = f.readlines()

    # Output directory (same directory as the script)
    output_dir = script_dir

    # Parse all kernels and generate files
    generated_files = []

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
        file_content = generate_cuda_file_content(params)

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

        if line_num % 100 == 0:
            print(f"Generated {line_num} files...")

    print(f"\nGenerated {len(generated_files)} CUDA kernel files")

    # Generate CMake file with all filenames
    # We're generating in the source tree at: cpp/src/neighbors/ivf_flat/jit_lto_kernels/
    # CMake file goes to: cpp/cmake/jit_lto_kernels_list/
    cmake_dir = script_dir.parent.parent.parent.parent / 'cmake' / 'jit_lto_kernels_list'
    cmake_dir.mkdir(parents=True, exist_ok=True)
    cmake_file = cmake_dir / 'interleaved_scan.cmake'

    # Generate CMake content
    cmake_content = "# Auto-generated list of interleaved scan kernel files\n"
    cmake_content += "# Generated by generate_kernels.py\n\n"
    cmake_content += "set(INTERLEAVED_SCAN_KERNEL_FILES\n"
    for filename in sorted(generated_files):
        cmake_content += f"  src/neighbors/ivf_flat/jit_lto_kernels/interleaved_scan_kernels/{filename}\n"
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

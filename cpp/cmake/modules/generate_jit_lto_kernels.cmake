# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

function(parse_data_type_configs config data_type acc_type veclens type_abbrev acc_abbrev)
  if(config MATCHES [==[^([^,]+),([^,]+),\[([0-9]+(,[0-9]+)*)?\],([^,]+),([^,]+)$]==])
    set(${data_type}
        "${CMAKE_MATCH_1}"
        PARENT_SCOPE
    )
    set(${acc_type}
        "${CMAKE_MATCH_2}"
        PARENT_SCOPE
    )
    string(REPLACE "," ";" veclens_value "${CMAKE_MATCH_3}")
    set(${veclens}
        "${veclens_value}"
        PARENT_SCOPE
    )
    set(${type_abbrev}
        "${CMAKE_MATCH_5}"
        PARENT_SCOPE
    )
    set(${acc_abbrev}
        "${CMAKE_MATCH_6}"
        PARENT_SCOPE
    )
  else()
    message(FATAL_ERROR "Invalid data type config: ${config}")
  endif()
endfunction()

function(generate_jit_lto_kernels)
  set(generated_kernels_dir "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels")
  string(TIMESTAMP year "%Y")

  set(capacities 0 1 2 4 8 16 32 64 128 256)
  set(ascending_values true false)
  set(compute_norm_values true false)
  set(data_type_configs "float,float,[1,4],f,f" "__half,__half,[1,8],h,h"
                        "uint8_t,uint32_t,[1,16],uc,ui" "int8_t,int32_t,[1,16],sc,i"
  )
  set(idx_type int64_t)
  set(idx_abbrev l)
  set(metric_configs euclidean inner_prod)
  set(filter_configs filter_none filter_bitset)
  set(post_lambda_configs post_identity post_sqrt post_compose)

  foreach(config IN LISTS data_type_configs)
    parse_data_type_configs("${config}" data_type acc_type veclens type_abbrev acc_abbrev)
    foreach(veclen IN LISTS veclens)
      foreach(capacity IN LISTS capacities)
        foreach(ascending IN LISTS ascending_values)
          foreach(compute_norm IN LISTS compute_norm_values)
            set(filename
                "${generated_kernels_dir}/interleaved_scan_kernels/interleaved_scan_kernel_${capacity}_${veclen}_${ascending}_${compute_norm}_${type_abbrev}_${acc_abbrev}_${idx_abbrev}.cu"
            )
            configure_file(
              "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/jit_lto_kernels/interleaved_scan_kernel.cu.in"
              "${filename}"
              @ONLY
            )
            list(APPEND INTERLEAVED_SCAN_KERNEL_FILES "${filename}")
          endforeach()
        endforeach()
      endforeach()

      foreach(metric_name IN LISTS metric_configs)
        if(metric_name STREQUAL "euclidean")
          set(header_file "neighbors/ivf_flat/jit_lto_kernels/metric_euclidean_dist.cuh")
        elseif(metric_name STREQUAL "inner_prod")
          set(header_file "neighbors/ivf_flat/jit_lto_kernels/metric_inner_product.cuh")
        endif()

        set(filename
            "${generated_kernels_dir}/metric_device_functions/metric_${metric_name}_${veclen}_${type_abbrev}_${acc_abbrev}.cu"
        )
        configure_file(
          "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/jit_lto_kernels/metric.cu.in"
          "${filename}"
          @ONLY
        )
        list(APPEND METRIC_DEVICE_FUNCTION_FILES "${filename}")
      endforeach()
    endforeach()
  endforeach()

  foreach(filter_name IN LISTS filter_configs)
    if(filter_name STREQUAL "filter_none")
      set(header_file "neighbors/ivf_flat/jit_lto_kernels/filter_none.cuh")
    elseif(filter_name STREQUAL "filter_bitset")
      set(header_file "neighbors/ivf_flat/jit_lto_kernels/filter_bitset.cuh")
    endif()

    set(filename
        "${generated_kernels_dir}/filter_device_functions/${filter_name}.cu"
    )
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/jit_lto_kernels/filter.cu.in"
      "${filename}" @ONLY
    )
    list(APPEND FILTER_DEVICE_FUNCTION_FILES "${filename}")
  endforeach()

  foreach(post_lambda_name IN LISTS post_lambda_configs)
    if(post_lambda_name STREQUAL "post_identity")
      set(header_file "neighbors/ivf_flat/jit_lto_kernels/post_identity.cuh")
    elseif(post_lambda_name STREQUAL "post_sqrt")
      set(header_file "neighbors/ivf_flat/jit_lto_kernels/post_sqrt.cuh")
    elseif(post_lambda_name STREQUAL "post_compose")
      set(header_file "neighbors/ivf_flat/jit_lto_kernels/post_compose.cuh")
    endif()

    set(filename "${generated_kernels_dir}/post_lambda_device_functions/${post_lambda_name}.cu")
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/jit_lto_kernels/post_lambda.cu.in"
      "${filename}" @ONLY
    )
    list(APPEND POST_LAMBDA_DEVICE_FUNCTION_FILES "${filename}")
  endforeach()

  set(INTERLEAVED_SCAN_KERNEL_FILES
      "${INTERLEAVED_SCAN_KERNEL_FILES}"
      PARENT_SCOPE
  )
  set(METRIC_DEVICE_FUNCTION_FILES
      "${METRIC_DEVICE_FUNCTION_FILES}"
      PARENT_SCOPE
  )
  set(FILTER_DEVICE_FUNCTION_FILES
      "${FILTER_DEVICE_FUNCTION_FILES}"
      PARENT_SCOPE
  )
  set(POST_LAMBDA_DEVICE_FUNCTION_FILES
      "${POST_LAMBDA_DEVICE_FUNCTION_FILES}"
      PARENT_SCOPE
  )
endfunction()

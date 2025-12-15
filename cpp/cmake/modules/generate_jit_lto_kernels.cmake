# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

function(embed_jit_lto_fatbin)
  set(options)
  set(one_value FATBIN_TARGET FATBIN_SOURCE EMBEDDED_TARGET EMBEDDED_HEADER EMBEDDED_ARRAY)
  set(multi_value)

  cmake_parse_arguments(_JIT_LTO "${options}" "${one_value}" "${multi_value}" ${ARGN})

  find_package(CUDAToolkit REQUIRED)
  find_program(
    bin_to_c
    NAMES bin2c
    PATHS ${CUDAToolkit_BIN_DIR}
  )

  add_library(${_JIT_LTO_FATBIN_TARGET} OBJECT "${_JIT_LTO_FATBIN_SOURCE}")
  target_compile_definitions(${_JIT_LTO_FATBIN_TARGET} PRIVATE BUILD_KERNEL)
  target_include_directories(
    ${_JIT_LTO_FATBIN_TARGET}
    PRIVATE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../c/include>"
  )
  target_compile_options(
    ${_JIT_LTO_FATBIN_TARGET}
    PRIVATE -Xfatbin=--compress-all
            --compress-mode=size
            "$<$<COMPILE_LANGUAGE:CXX>:${CUVS_CXX_FLAGS}>"
            "$<$<COMPILE_LANGUAGE:CUDA>:${CUVS_CUDA_FLAGS}>"
  )
  set_target_properties(
    ${_JIT_LTO_FATBIN_TARGET}
    PROPERTIES CUDA_ARCHITECTURES ${JIT_LTO_TARGET_ARCHITECTURE}
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               CUDA_SEPARABLE_COMPILATION ON
               CUDA_FATBIN_COMPILATION ON
               POSITION_INDEPENDENT_CODE ON
               INTERPROCEDURAL_OPTIMIZATION ON
  )
  target_link_libraries(${_JIT_LTO_FATBIN_TARGET} PRIVATE rmm::rmm raft::raft CCCL::CCCL)

  add_custom_command(
    OUTPUT "${_JIT_LTO_EMBEDDED_HEADER}"
    COMMAND "${bin_to_c}" -c -p 0x0 --name "${_JIT_LTO_EMBEDDED_ARRAY}" --static
            $<TARGET_OBJECTS:${_JIT_LTO_FATBIN_TARGET}> > "${_JIT_LTO_EMBEDDED_HEADER}"
    DEPENDS $<TARGET_OBJECTS:${_JIT_LTO_FATBIN_TARGET}>
  )
  target_sources(
    ${_JIT_LTO_EMBEDDED_TARGET} PRIVATE "${_JIT_LTO_FATBIN_SOURCE}" "${_JIT_LTO_EMBEDDED_HEADER}"
  )
  cmake_path(GET _JIT_LTO_EMBEDDED_HEADER PARENT_PATH header_dir)
  target_include_directories(${_JIT_LTO_EMBEDDED_TARGET} PRIVATE "${header_dir}")
endfunction()

function(parse_jit_lto_data_type_configs config)
  set(options)
  set(one_value DATA_TYPE ACC_TYPE VECLENS TYPE_ABBREV ACC_ABBREV)
  set(multi_value)

  cmake_parse_arguments(_JIT_LTO "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(config MATCHES [==[^([^,]+),([^,]+),\[([0-9]+(,[0-9]+)*)?\],([^,]+),([^,]+)$]==])
    if(_JIT_LTO_DATA_TYPE)
      set(${_JIT_LTO_DATA_TYPE}
          "${CMAKE_MATCH_1}"
          PARENT_SCOPE
      )
    endif()
    if(_JIT_LTO_ACC_TYPE)
      set(${_JIT_LTO_ACC_TYPE}
          "${CMAKE_MATCH_2}"
          PARENT_SCOPE
      )
    endif()
    if(_JIT_LTO_VECLENS)
      string(REPLACE "," ";" veclens_value "${CMAKE_MATCH_3}")
      set(${_JIT_LTO_VECLENS}
          "${veclens_value}"
          PARENT_SCOPE
      )
    endif()
    if(_JIT_LTO_TYPE_ABBREV)
      set(${_JIT_LTO_TYPE_ABBREV}
          "${CMAKE_MATCH_5}"
          PARENT_SCOPE
      )
    endif()
    if(_JIT_LTO_ACC_ABBREV)
      set(${_JIT_LTO_ACC_ABBREV}
          "${CMAKE_MATCH_6}"
          PARENT_SCOPE
      )
    endif()
  else()
    message(FATAL_ERROR "Invalid data type config: ${config}")
  endif()
endfunction()

# cmake-lint: disable=R0915
function(generate_jit_lto_kernels target)
  add_library(${target} OBJECT)
  target_include_directories(
    ${target}
    PRIVATE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../c/include>"
  )
  set_target_properties(
    ${target}
    PROPERTIES POSITION_INDEPENDENT_CODE ON
  )

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
    parse_jit_lto_data_type_configs(
      "${config}" DATA_TYPE data_type ACC_TYPE acc_type VECLENS veclens TYPE_ABBREV type_abbrev
      ACC_ABBREV acc_abbrev
    )
    foreach(veclen IN LISTS veclens)
      foreach(capacity IN LISTS capacities)
        foreach(ascending IN LISTS ascending_values)
          foreach(compute_norm IN LISTS compute_norm_values)
            set(kernel_name
                "interleaved_scan_kernel_${capacity}_${veclen}_${ascending}_${compute_norm}_${type_abbrev}_${acc_abbrev}_${idx_abbrev}"
            )
            set(filename
                "${generated_kernels_dir}/interleaved_scan_kernels/fatbin_${kernel_name}.cu"
            )
            configure_file(
              "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/jit_lto_kernels/interleaved_scan_kernel.cu.in"
              "${filename}"
              @ONLY
            )
            embed_jit_lto_fatbin(
              FATBIN_TARGET "fatbin_${kernel_name}"
              FATBIN_SOURCE "${filename}"
              EMBEDDED_TARGET "${target}"
              EMBEDDED_HEADER "${generated_kernels_dir}/interleaved_scan_kernels/${kernel_name}.h"
              EMBEDDED_ARRAY "embedded_${kernel_name}"
            )
          endforeach()
        endforeach()
      endforeach()

      foreach(metric_name IN LISTS metric_configs)
        if(metric_name STREQUAL "euclidean")
          set(header_file "neighbors/ivf_flat/jit_lto_kernels/metric_euclidean_dist.cuh")
        elseif(metric_name STREQUAL "inner_prod")
          set(header_file "neighbors/ivf_flat/jit_lto_kernels/metric_inner_product.cuh")
        endif()

        set(kernel_name "metric_${metric_name}_${veclen}_${type_abbrev}_${acc_abbrev}")
        set(filename "${generated_kernels_dir}/metric_device_functions/fatbin_${kernel_name}.cu")
        configure_file(
          "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/jit_lto_kernels/metric.cu.in"
          "${filename}" @ONLY
        )
        embed_jit_lto_fatbin(
          FATBIN_TARGET "fatbin_${kernel_name}"
          FATBIN_SOURCE "${filename}"
          EMBEDDED_TARGET "${target}"
          EMBEDDED_HEADER "${generated_kernels_dir}/metric_device_functions/${kernel_name}.h"
          EMBEDDED_ARRAY "embedded_${kernel_name}"
        )
      endforeach()
    endforeach()
  endforeach()

  foreach(filter_name IN LISTS filter_configs)
    if(filter_name STREQUAL "filter_none")
      set(header_file "neighbors/ivf_flat/jit_lto_kernels/filter_none.cuh")
    elseif(filter_name STREQUAL "filter_bitset")
      set(header_file "neighbors/ivf_flat/jit_lto_kernels/filter_bitset.cuh")
    endif()

    set(kernel_name "${filter_name}")
    set(filename "${generated_kernels_dir}/filter_device_functions/fatbin_${kernel_name}.cu")
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/jit_lto_kernels/filter.cu.in"
      "${filename}" @ONLY
    )
    embed_jit_lto_fatbin(
      FATBIN_TARGET "fatbin_${kernel_name}"
      FATBIN_SOURCE "${filename}"
      EMBEDDED_TARGET "${target}"
      EMBEDDED_HEADER "${generated_kernels_dir}/filter_device_functions/${kernel_name}.h"
      EMBEDDED_ARRAY "embedded_${kernel_name}"
    )
  endforeach()

  foreach(post_lambda_name IN LISTS post_lambda_configs)
    if(post_lambda_name STREQUAL "post_identity")
      set(header_file "neighbors/ivf_flat/jit_lto_kernels/post_identity.cuh")
    elseif(post_lambda_name STREQUAL "post_sqrt")
      set(header_file "neighbors/ivf_flat/jit_lto_kernels/post_sqrt.cuh")
    elseif(post_lambda_name STREQUAL "post_compose")
      set(header_file "neighbors/ivf_flat/jit_lto_kernels/post_compose.cuh")
    endif()

    set(kernel_name "${post_lambda_name}")
    set(filename "${generated_kernels_dir}/post_lambda_device_functions/${post_lambda_name}.cu")
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/jit_lto_kernels/post_lambda.cu.in"
      "${filename}" @ONLY
    )
    embed_jit_lto_fatbin(
      FATBIN_TARGET "fatbin_${kernel_name}"
      FATBIN_SOURCE "${filename}"
      EMBEDDED_TARGET "${target}"
      EMBEDDED_HEADER "${generated_kernels_dir}/post_lambda_device_functions/${kernel_name}.h"
      EMBEDDED_ARRAY "embedded_${kernel_name}"
    )
  endforeach()
endfunction()

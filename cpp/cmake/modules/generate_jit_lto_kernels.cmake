# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
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
  add_library(${target} STATIC)
  target_include_directories(
    ${target}
    PRIVATE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../c/include>"
  )
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)

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
        set(header_file "neighbors/ivf_flat/jit_lto_kernels/metric_${metric_name}.cuh")

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
    set(header_file "neighbors/ivf_flat/jit_lto_kernels/${filter_name}.cuh")

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
    set(header_file "neighbors/ivf_flat/jit_lto_kernels/${post_lambda_name}.cuh")

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

  # Generate CAGRA device function fragments
  set(cagra_data_types "float" "__half" "uint8_t" "int8_t")
  set(cagra_data_type_abbrevs "f" "h" "uc" "sc")
  set(cagra_index_type "uint32_t")
  set(cagra_index_abbrev "ui")
  set(cagra_distance_type "float")
  set(cagra_distance_abbrev "f")
  set(cagra_metrics "L2Expanded" "InnerProduct" "CosineExpanded")
  set(cagra_metric_abbrevs "l2" "ip" "cos")
  set(cagra_team_sizes 8 16 32)
  set(cagra_dataset_block_dims 128 256 512)
  set(cagra_pq_bits 8)
  set(cagra_pq_lens 2 4)
  set(cagra_codebook_type "half")

  # Generate standard descriptor fragments
  foreach(data_idx IN ITEMS 0 1 2 3)
    list(GET cagra_data_types ${data_idx} data_type)
    list(GET cagra_data_type_abbrevs ${data_idx} type_abbrev)
    foreach(metric_idx IN ITEMS 0 1 2)
      list(GET cagra_metrics ${metric_idx} metric)
      list(GET cagra_metric_abbrevs ${metric_idx} metric_name)
      foreach(team_size IN LISTS cagra_team_sizes)
        foreach(dataset_block_dim IN LISTS cagra_dataset_block_dims)
          # setup_workspace_standard
          set(kernel_name
              "setup_workspace_standard_${metric_name}_t${team_size}_dim${dataset_block_dim}_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}"
          )
          set(filename "${generated_kernels_dir}/cagra_device_functions/fatbin_${kernel_name}.cu")
          set(metric_cpp "cuvs::distance::DistanceType::${metric}")
          set(data_type "${data_type}")
          set(index_type "${cagra_index_type}")
          set(distance_type "${cagra_distance_type}")
          set(idx_abbrev "${cagra_index_abbrev}")
          set(dist_abbrev "${cagra_distance_abbrev}")
          configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/setup_workspace_standard.cu.in"
            "${filename}"
            @ONLY
          )
          embed_jit_lto_fatbin(
            FATBIN_TARGET "fatbin_${kernel_name}"
            FATBIN_SOURCE "${filename}"
            EMBEDDED_TARGET "${target}"
            EMBEDDED_HEADER "${generated_kernels_dir}/cagra_device_functions/${kernel_name}.h"
            EMBEDDED_ARRAY "embedded_${kernel_name}"
          )

          # compute_distance_standard
          set(kernel_name
              "compute_distance_standard_${metric_name}_t${team_size}_dim${dataset_block_dim}_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}"
          )
          set(filename "${generated_kernels_dir}/cagra_device_functions/fatbin_${kernel_name}.cu")
          set(metric_cpp "cuvs::distance::DistanceType::${metric}")
          set(data_type "${data_type}")
          set(index_type "${cagra_index_type}")
          set(distance_type "${cagra_distance_type}")
          set(idx_abbrev "${cagra_index_abbrev}")
          set(dist_abbrev "${cagra_distance_abbrev}")
          configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/compute_distance_standard.cu.in"
            "${filename}"
            @ONLY
          )
          embed_jit_lto_fatbin(
            FATBIN_TARGET "fatbin_${kernel_name}"
            FATBIN_SOURCE "${filename}"
            EMBEDDED_TARGET "${target}"
            EMBEDDED_HEADER "${generated_kernels_dir}/cagra_device_functions/${kernel_name}.h"
            EMBEDDED_ARRAY "embedded_${kernel_name}"
          )
        endforeach()
      endforeach()
    endforeach()
  endforeach()

  # Generate VPQ descriptor fragments (only for L2Expanded and float/half)
  foreach(data_idx IN ITEMS 0 1)
    list(GET cagra_data_types ${data_idx} data_type)
    list(GET cagra_data_type_abbrevs ${data_idx} type_abbrev)
    foreach(team_size IN LISTS cagra_team_sizes)
      foreach(dataset_block_dim IN LISTS cagra_dataset_block_dims)
        foreach(pq_len IN LISTS cagra_pq_lens)
          # setup_workspace_vpq
          set(kernel_name
              "setup_workspace_vpq_l2_t${team_size}_dim${dataset_block_dim}_${cagra_pq_bits}pq_${pq_len}subd_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}"
          )
          set(filename "${generated_kernels_dir}/cagra_device_functions/fatbin_${kernel_name}.cu")
          set(metric_cpp "cuvs::distance::DistanceType::L2Expanded")
          set(metric_name "l2")
          set(pq_bits "${cagra_pq_bits}")
          set(codebook_type "${cagra_codebook_type}")
          set(data_type "${data_type}")
          set(index_type "${cagra_index_type}")
          set(distance_type "${cagra_distance_type}")
          set(idx_abbrev "${cagra_index_abbrev}")
          set(dist_abbrev "${cagra_distance_abbrev}")
          configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/setup_workspace_vpq.cu.in"
            "${filename}"
            @ONLY
          )
          embed_jit_lto_fatbin(
            FATBIN_TARGET "fatbin_${kernel_name}"
            FATBIN_SOURCE "${filename}"
            EMBEDDED_TARGET "${target}"
            EMBEDDED_HEADER "${generated_kernels_dir}/cagra_device_functions/${kernel_name}.h"
            EMBEDDED_ARRAY "embedded_${kernel_name}"
          )

          # compute_distance_vpq
          set(kernel_name
              "compute_distance_vpq_l2_t${team_size}_dim${dataset_block_dim}_${cagra_pq_bits}pq_${pq_len}subd_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}"
          )
          set(filename "${generated_kernels_dir}/cagra_device_functions/fatbin_${kernel_name}.cu")
          set(metric_cpp "cuvs::distance::DistanceType::L2Expanded")
          set(metric_name "l2")
          set(pq_bits "${cagra_pq_bits}")
          set(codebook_type "${cagra_codebook_type}")
          set(idx_abbrev "${cagra_index_abbrev}")
          set(dist_abbrev "${cagra_distance_abbrev}")
          set(data_type "${data_type}")
          set(index_type "${cagra_index_type}")
          set(distance_type "${cagra_distance_type}")
          configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/compute_distance_vpq.cu.in"
            "${filename}"
            @ONLY
          )
          embed_jit_lto_fatbin(
            FATBIN_TARGET "fatbin_${kernel_name}"
            FATBIN_SOURCE "${filename}"
            EMBEDDED_TARGET "${target}"
            EMBEDDED_HEADER "${generated_kernels_dir}/cagra_device_functions/${kernel_name}.h"
            EMBEDDED_ARRAY "embedded_${kernel_name}"
          )
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endfunction()

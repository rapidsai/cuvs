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

# cmake-lint: disable=R0915,R0912
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

  # Generate IVF Flat sample filter fragments using shared implementation
  foreach(filter_name IN LISTS filter_configs)
    set(header_file "neighbors/detail/jit_lto_kernels/${filter_name}.cuh")
    set(kernel_name "${filter_name}_${idx_abbrev}")
    set(filename "${generated_kernels_dir}/filter_device_functions/fatbin_${kernel_name}.cu")
    set(source_index_type "int64_t")
    set(namespace "cuvs::neighbors::detail")
    set(filter_name_var "${filter_name}")
    set(kernel_name_var "${kernel_name}")
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/jit_lto_kernels/filter.cu.in" "${filename}"
      @ONLY
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
  set(cagra_metrics "L2Expanded" "InnerProduct" "CosineExpanded" "BitwiseHamming")
  set(cagra_metric_abbrevs "l2" "ip" "cos" "hamming")
  set(cagra_team_sizes 8 16 32)
  set(cagra_dataset_block_dims 128 256 512)
  set(cagra_pq_bits 8)
  set(cagra_pq_lens 2 4)
  set(cagra_codebook_type "half")
  # CAGRA kernels only use uint32_t as SourceIndexT (matching non-JIT path)
  set(cagra_source_index_types "uint32_t")
  set(cagra_source_index_abbrevs "ui")

  # Generate setup_workspace fragments (one per team_size, dataset_block_dim, data_type, index_type,
  # distance_type, query_type) QueryT can be float (for most metrics) or uint8_t (for BitwiseHamming
  # when DataT=uint8_t)
  foreach(data_idx IN ITEMS 0 1 2 3)
    list(GET cagra_data_types ${data_idx} data_type)
    list(GET cagra_data_type_abbrevs ${data_idx} type_abbrev)
    foreach(team_size IN LISTS cagra_team_sizes)
      foreach(dataset_block_dim IN LISTS cagra_dataset_block_dims)
        # Always generate QueryT=float fragment
        set(kernel_name
            "setup_workspace_standard_t${team_size}_dim${dataset_block_dim}_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}_f"
        )
        set(filename "${generated_kernels_dir}/cagra_device_functions/fatbin_${kernel_name}.cu")
        set(data_type "${data_type}")
        set(index_type "${cagra_index_type}")
        set(distance_type "${cagra_distance_type}")
        set(query_type "float")
        set(query_type_abbrev "f")
        set(query_type_suffix "_f")
        set(query_type_suffix_reg "_f")
        set(pq_bits "0")
        set(pq_len "0")
        set(codebook_type "void")
        set(pq_prefix "_standard")
        set(pq_suffix "")
        set(codebook_tag "")
        set(codebook_tag_comma "")
        set(impl_file "setup_workspace_standard_impl.cuh")
        set(idx_abbrev "${cagra_index_abbrev}")
        set(dist_abbrev "${cagra_distance_abbrev}")
        configure_file(
          "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/setup_workspace.cu.in"
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
        # For uint8_t data type, also generate QueryT=uint8_t fragment (for BitwiseHamming)
        if(data_idx EQUAL 2) # uint8_t
          set(kernel_name
              "setup_workspace_standard_t${team_size}_dim${dataset_block_dim}_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}_uc"
          )
          set(filename "${generated_kernels_dir}/cagra_device_functions/fatbin_${kernel_name}.cu")
          set(data_type "${data_type}")
          set(index_type "${cagra_index_type}")
          set(distance_type "${cagra_distance_type}")
          set(query_type "uint8_t")
          set(query_type_abbrev "uc")
          set(query_type_suffix "_uc")
          set(query_type_suffix_reg "_uc")
          set(pq_bits "0")
          set(pq_len "0")
          set(codebook_type "void")
          set(pq_prefix "_standard")
          set(pq_suffix "")
          set(codebook_tag "")
          set(codebook_tag_comma "")
          set(impl_file "setup_workspace_standard_impl.cuh")
          set(idx_abbrev "${cagra_index_abbrev}")
          set(dist_abbrev "${cagra_distance_abbrev}")
          configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/setup_workspace.cu.in"
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
        endif()
      endforeach()
    endforeach()
  endforeach()

  # Generate compute_distance fragments (without metric - metric is handled via dist_op fragments)
  # QueryT can be float (for most metrics) or uint8_t (for BitwiseHamming when DataT=uint8_t)
  foreach(data_idx IN ITEMS 0 1 2 3)
    list(GET cagra_data_types ${data_idx} data_type)
    list(GET cagra_data_type_abbrevs ${data_idx} type_abbrev)
    foreach(team_size IN LISTS cagra_team_sizes)
      foreach(dataset_block_dim IN LISTS cagra_dataset_block_dims)
        # Always generate QueryT=float fragment
        set(kernel_name
            "compute_distance_standard_t${team_size}_dim${dataset_block_dim}_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}_f"
        )
        set(filename "${generated_kernels_dir}/cagra_device_functions/fatbin_${kernel_name}.cu")
        set(data_type "${data_type}")
        set(index_type "${cagra_index_type}")
        set(distance_type "${cagra_distance_type}")
        set(query_type "float")
        set(query_type_abbrev "f")
        set(query_type_suffix "_f")
        set(query_type_suffix_reg "_f")
        set(pq_bits "0")
        set(pq_len "0")
        set(codebook_type "void")
        set(pq_prefix "_standard")
        set(pq_suffix "")
        set(codebook_tag "")
        set(codebook_tag_comma "")
        set(impl_file "compute_distance_standard_impl.cuh")
        set(idx_abbrev "${cagra_index_abbrev}")
        set(dist_abbrev "${cagra_distance_abbrev}")
        configure_file(
          "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/compute_distance.cu.in"
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
        # For uint8_t data type, also generate QueryT=uint8_t fragment (for BitwiseHamming)
        if(data_idx EQUAL 2) # uint8_t
          set(kernel_name
              "compute_distance_standard_t${team_size}_dim${dataset_block_dim}_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}_uc"
          )
          set(filename "${generated_kernels_dir}/cagra_device_functions/fatbin_${kernel_name}.cu")
          set(data_type "${data_type}")
          set(index_type "${cagra_index_type}")
          set(distance_type "${cagra_distance_type}")
          set(query_type "uint8_t")
          set(query_type_abbrev "uc")
          set(query_type_suffix "_uc")
          set(query_type_suffix_reg "_uc")
          set(pq_bits "0")
          set(pq_len "0")
          set(codebook_type "void")
          set(pq_prefix "_standard")
          set(pq_suffix "")
          set(codebook_tag "")
          set(codebook_tag_comma "")
          set(impl_file "compute_distance_standard_impl.cuh")
          set(idx_abbrev "${cagra_index_abbrev}")
          set(dist_abbrev "${cagra_distance_abbrev}")
          configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/compute_distance.cu.in"
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
        endif()
      endforeach()
    endforeach()
  endforeach()

  # Generate dist_op fragments for each metric QueryT can be float (for most metrics) or uint8_t
  # (for BitwiseHamming) DistanceT is always float Generate dist_op fragments for unique metric
  # tags: l2, inner_product (used by both ip and cos), hamming
  set(dist_op_tags "l2" "inner_product" "hamming")
  foreach(metric_tag IN LISTS dist_op_tags)
    if(metric_tag STREQUAL "hamming")
      set(query_type "uint8_t")
      set(query_type_abbrev "uc")
    else()
      set(query_type "float")
      set(query_type_abbrev "f")
    endif()
    set(kernel_name "dist_op_${metric_tag}_${query_type_abbrev}_${cagra_distance_abbrev}")
    set(filename "${generated_kernels_dir}/cagra_device_functions/fatbin_${kernel_name}.cu")
    set(metric_tag "${metric_tag}")
    set(query_type "${query_type}")
    set(query_type_abbrev "${query_type_abbrev}")
    set(distance_type "${cagra_distance_type}")
    set(dist_abbrev "${cagra_distance_abbrev}")
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/dist_op.cu.in"
      "${filename}" @ONLY
    )
    embed_jit_lto_fatbin(
      FATBIN_TARGET "fatbin_${kernel_name}"
      FATBIN_SOURCE "${filename}"
      EMBEDDED_TARGET "${target}"
      EMBEDDED_HEADER "${generated_kernels_dir}/cagra_device_functions/${kernel_name}.h"
      EMBEDDED_ARRAY "embedded_${kernel_name}"
    )
  endforeach()

  # Generate normalization fragments (no-op and cosine) These are used to normalize distances for
  # CosineExpanded metric
  foreach(data_idx IN ITEMS 0 1 2 3)
    list(GET cagra_data_types ${data_idx} data_type)
    list(GET cagra_data_type_abbrevs ${data_idx} type_abbrev)
    foreach(team_size IN LISTS cagra_team_sizes)
      foreach(dataset_block_dim IN LISTS cagra_dataset_block_dims)
        # No-op normalization fragment (for non-CosineExpanded metrics)
        set(kernel_name
            "apply_normalization_standard_noop_t${team_size}_dim${dataset_block_dim}_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}"
        )
        set(filename "${generated_kernels_dir}/cagra_device_functions/fatbin_${kernel_name}.cu")
        set(data_type "${data_type}")
        set(index_type "${cagra_index_type}")
        set(distance_type "${cagra_distance_type}")
        set(idx_abbrev "${cagra_index_abbrev}")
        set(dist_abbrev "${cagra_distance_abbrev}")
        set(normalization_suffix "_noop")
        configure_file(
          "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/apply_normalization.cu.in"
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

        # Cosine normalization fragment (for CosineExpanded metric)
        set(kernel_name
            "apply_normalization_standard_cosine_t${team_size}_dim${dataset_block_dim}_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}"
        )
        set(filename "${generated_kernels_dir}/cagra_device_functions/fatbin_${kernel_name}.cu")
        set(normalization_suffix "_cosine")
        configure_file(
          "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/apply_normalization.cu.in"
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

  # Generate VPQ descriptor fragments NOTE: VPQ is ONLY supported with L2Expanded metric (not
  # InnerProduct, CosineExpanded, etc.) Generating for all data types: float, half, int8_t, uint8_t
  foreach(data_idx IN ITEMS 0 1 2 3)
    list(GET cagra_data_types ${data_idx} data_type)
    list(GET cagra_data_type_abbrevs ${data_idx} type_abbrev)
    foreach(team_size IN LISTS cagra_team_sizes)
      foreach(dataset_block_dim IN LISTS cagra_dataset_block_dims)
        foreach(pq_len IN LISTS cagra_pq_lens)
          # L2Expanded
          set(kernel_name
              "setup_workspace_vpq_t${team_size}_dim${dataset_block_dim}_${cagra_pq_bits}pq_${pq_len}subd_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}"
          )
          set(filename "${generated_kernels_dir}/cagra_device_functions/fatbin_${kernel_name}.cu")
          set(pq_bits "${cagra_pq_bits}")
          set(pq_len "${pq_len}")
          set(codebook_type "${cagra_codebook_type}")
          set(query_type "half")
          set(query_type_abbrev "h")
          set(query_type_suffix "")
          set(query_type_suffix_reg "")
          set(pq_prefix "_vpq")
          set(pq_suffix "_${cagra_pq_bits}pq_${pq_len}subd")
          set(codebook_tag "tag_codebook_half")
          set(codebook_tag_comma ", ")
          set(impl_file "setup_workspace_vpq_impl.cuh")
          set(data_type "${data_type}")
          set(index_type "${cagra_index_type}")
          set(distance_type "${cagra_distance_type}")
          set(idx_abbrev "${cagra_index_abbrev}")
          set(dist_abbrev "${cagra_distance_abbrev}")
          configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/setup_workspace.cu.in"
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

          # L2Expanded
          set(kernel_name
              "compute_distance_vpq_t${team_size}_dim${dataset_block_dim}_${cagra_pq_bits}pq_${pq_len}subd_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}"
          )
          set(filename "${generated_kernels_dir}/cagra_device_functions/fatbin_${kernel_name}.cu")
          set(pq_bits "${cagra_pq_bits}")
          set(pq_len "${pq_len}")
          set(codebook_type "${cagra_codebook_type}")
          set(query_type "half")
          set(query_type_abbrev "h")
          set(query_type_suffix "")
          set(query_type_suffix_reg "")
          set(pq_prefix "_vpq")
          set(pq_suffix "_${cagra_pq_bits}pq_${pq_len}subd")
          set(codebook_tag "tag_codebook_half")
          set(codebook_tag_comma ", ")
          set(impl_file "compute_distance_vpq_impl.cuh")
          set(idx_abbrev "${cagra_index_abbrev}")
          set(dist_abbrev "${cagra_distance_abbrev}")
          set(data_type "${data_type}")
          set(index_type "${cagra_index_type}")
          set(distance_type "${cagra_distance_type}")
          configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/compute_distance.cu.in"
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

  # Generate CAGRA kernel entrypoint fragments These are the main kernel entrypoints that call the
  # device functions
  set(cagra_topk_by_bitonic_sort_options "true" "false")
  set(cagra_bitonic_sort_and_merge_multi_warps_options "true" "false")
  set(cagra_topk_by_bitonic_sort_str_options "true" "false")
  set(cagra_bitonic_sort_and_merge_multi_warps_str_options "true" "false")

  # For kernel instantiation, we need to provide template parameters The actual
  # metric/team_size/dataset_block_dim used at runtime are determined via device functions We use
  # default values for the template instantiation - these don't affect runtime behavior Note: Metric
  # is no longer in the kernel name - it's linked via dist_op and normalization fragments
  foreach(data_idx IN ITEMS 0 1 2 3)
    list(GET cagra_data_types ${data_idx} data_type)
    list(GET cagra_data_type_abbrevs ${data_idx} type_abbrev)
    foreach(topk_idx IN ITEMS 0 1)
      list(GET cagra_topk_by_bitonic_sort_options ${topk_idx} topk_by_bitonic_sort)
      list(GET cagra_topk_by_bitonic_sort_str_options ${topk_idx} topk_by_bitonic_sort_str)
      foreach(merge_idx IN ITEMS 0 1)
        list(GET cagra_bitonic_sort_and_merge_multi_warps_options ${merge_idx}
             bitonic_sort_and_merge_multi_warps
        )
        list(GET cagra_bitonic_sort_and_merge_multi_warps_str_options ${merge_idx}
             bitonic_sort_and_merge_multi_warps_str
        )
        foreach(team_size IN LISTS cagra_team_sizes)
          foreach(dataset_block_dim IN LISTS cagra_dataset_block_dims)
            # CAGRA only uses uint32_t as SourceIndexT
            set(source_index_type "uint32_t")
            set(src_idx_abbrev "ui")
            # Generate QueryT variants: float for all, uint8_t only when DataT=uint8_t
            set(query_type_variants "float")
            set(query_type_abbrev_variants "f")
            if(data_idx EQUAL 2) # uint8_t
              list(APPEND query_type_variants "uint8_t")
              list(APPEND query_type_abbrev_variants "uc")
            endif()
            foreach(query_idx IN ITEMS 0 1)
              # Skip second iteration if we don't have uint8_t variant
              if(query_idx EQUAL 1 AND NOT data_idx EQUAL 2)
                break()
              endif()
              list(GET query_type_variants ${query_idx} query_type)
              list(GET query_type_abbrev_variants ${query_idx} query_type_abbrev)
              # Regular kernel entrypoint (no metric in name)
              set(kernel_name
                  "search_single_cta_kernel_${topk_by_bitonic_sort_str}_${bitonic_sort_and_merge_multi_warps_str}_t${team_size}_dim${dataset_block_dim}_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}_${query_type_abbrev}_${src_idx_abbrev}"
              )
              set(filename
                  "${generated_kernels_dir}/cagra_kernel_entrypoints/fatbin_${kernel_name}.cu"
              )
              set(team_size "${team_size}")
              set(dataset_block_dim "${dataset_block_dim}")
              set(pq_bits "0")
              set(pq_len "0")
              set(codebook_type "void")
              set(pq_suffix "")
              set(pq_prefix "")
              set(codebook_tag "")
              set(codebook_tag_comma "")
              set(query_type "${query_type}")
              set(query_type_abbrev "${query_type_abbrev}")
              set(index_type "${cagra_index_type}")
              set(distance_type "${cagra_distance_type}")
              set(idx_abbrev "${cagra_index_abbrev}")
              set(dist_abbrev "${cagra_distance_abbrev}")
              configure_file(
                "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/search_single_cta_kernel.cu.in"
                "${filename}"
                @ONLY
              )
              embed_jit_lto_fatbin(
                FATBIN_TARGET "fatbin_${kernel_name}"
                FATBIN_SOURCE "${filename}"
                EMBEDDED_TARGET "${target}"
                EMBEDDED_HEADER "${generated_kernels_dir}/cagra_kernel_entrypoints/${kernel_name}.h"
                EMBEDDED_ARRAY "embedded_${kernel_name}"
              )

              # Persistent kernel entrypoint (no metric in name)
              set(kernel_name
                  "search_single_cta_kernel_p_${topk_by_bitonic_sort_str}_${bitonic_sort_and_merge_multi_warps_str}_t${team_size}_dim${dataset_block_dim}_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}_${query_type_abbrev}_${src_idx_abbrev}"
              )
              set(filename
                  "${generated_kernels_dir}/cagra_kernel_entrypoints/fatbin_${kernel_name}.cu"
              )
              set(team_size "${team_size}")
              set(dataset_block_dim "${dataset_block_dim}")
              set(pq_bits "0")
              set(pq_len "0")
              set(codebook_type "void")
              set(pq_suffix "")
              set(pq_prefix "")
              set(codebook_tag "")
              set(codebook_tag_comma "")
              set(query_type "${query_type}")
              set(query_type_abbrev "${query_type_abbrev}")
              set(index_type "${cagra_index_type}")
              set(distance_type "${cagra_distance_type}")
              set(idx_abbrev "${cagra_index_abbrev}")
              set(dist_abbrev "${cagra_distance_abbrev}")
              configure_file(
                "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/search_single_cta_kernel_p.cu.in"
                "${filename}"
                @ONLY
              )
              embed_jit_lto_fatbin(
                FATBIN_TARGET "fatbin_${kernel_name}"
                FATBIN_SOURCE "${filename}"
                EMBEDDED_TARGET "${target}"
                EMBEDDED_HEADER "${generated_kernels_dir}/cagra_kernel_entrypoints/${kernel_name}.h"
                EMBEDDED_ARRAY "embedded_${kernel_name}"
              )
            endforeach() # query_type variant
          endforeach() # dataset_block_dim
        endforeach() # team_size
      endforeach() # merge_idx
    endforeach() # topk_idx
  endforeach() # data_idx

  # Generate single_cta VPQ kernel entrypoints NOTE: VPQ is ONLY supported with L2Expanded metric
  # VPQ kernels need pq_bits and pq_len in addition to team_size and dataset_block_dim
  foreach(data_idx IN ITEMS 0 1 2 3)
    list(GET cagra_data_types ${data_idx} data_type)
    list(GET cagra_data_type_abbrevs ${data_idx} type_abbrev)
    foreach(topk_idx IN ITEMS 0 1)
      list(GET cagra_topk_by_bitonic_sort_options ${topk_idx} topk_by_bitonic_sort)
      list(GET cagra_topk_by_bitonic_sort_str_options ${topk_idx} topk_by_bitonic_sort_str)
      foreach(merge_idx IN ITEMS 0 1)
        list(GET cagra_bitonic_sort_and_merge_multi_warps_options ${merge_idx}
             bitonic_sort_and_merge_multi_warps
        )
        list(GET cagra_bitonic_sort_and_merge_multi_warps_str_options ${merge_idx}
             bitonic_sort_and_merge_multi_warps_str
        )
        foreach(team_size IN LISTS cagra_team_sizes)
          foreach(dataset_block_dim IN LISTS cagra_dataset_block_dims)
            foreach(pq_len IN LISTS cagra_pq_lens)
              # CAGRA only uses uint32_t as SourceIndexT
              set(source_index_type "uint32_t")
              set(src_idx_abbrev "ui")
              # Regular VPQ kernel entrypoint Note: "vpq" is no longer in the kernel name - PQ
              # parameters distinguish VPQ Metric is no longer in the kernel name - VPQ only
              # supports L2Expanded
              set(kernel_name
                  "search_single_cta_kernel_${topk_by_bitonic_sort_str}_${bitonic_sort_and_merge_multi_warps_str}_t${team_size}_dim${dataset_block_dim}_${cagra_pq_bits}pq_${pq_len}subd_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}_h_${src_idx_abbrev}"
              )
              set(filename
                  "${generated_kernels_dir}/cagra_kernel_entrypoints/fatbin_${kernel_name}.cu"
              )
              # VPQ only supports L2Expanded, but we don't need to pass metric to the template
              # anymore
              set(team_size "${team_size}")
              set(dataset_block_dim "${dataset_block_dim}")
              set(pq_bits "${cagra_pq_bits}")
              set(pq_len "${pq_len}")
              set(codebook_type "${cagra_codebook_type}")
              set(pq_suffix "_${cagra_pq_bits}pq_${pq_len}subd")
              set(pq_prefix "")
              set(codebook_tag "tag_codebook_half")
              set(codebook_tag_comma ", ")
              set(query_type "half")
              set(query_type_abbrev "h")
              set(index_type "${cagra_index_type}")
              set(distance_type "${cagra_distance_type}")
              set(idx_abbrev "${cagra_index_abbrev}")
              set(dist_abbrev "${cagra_distance_abbrev}")
              configure_file(
                "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/search_single_cta_kernel.cu.in"
                "${filename}"
                @ONLY
              )
              embed_jit_lto_fatbin(
                FATBIN_TARGET "fatbin_${kernel_name}"
                FATBIN_SOURCE "${filename}"
                EMBEDDED_TARGET "${target}"
                EMBEDDED_HEADER "${generated_kernels_dir}/cagra_kernel_entrypoints/${kernel_name}.h"
                EMBEDDED_ARRAY "embedded_${kernel_name}"
              )

              # Persistent VPQ kernel entrypoint Note: "vpq" is no longer in the kernel name - PQ
              # parameters distinguish VPQ Metric is no longer in the kernel name - VPQ only
              # supports L2Expanded
              set(kernel_name
                  "search_single_cta_kernel_p_${topk_by_bitonic_sort_str}_${bitonic_sort_and_merge_multi_warps_str}_t${team_size}_dim${dataset_block_dim}_${cagra_pq_bits}pq_${pq_len}subd_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}_h_${src_idx_abbrev}"
              )
              set(filename
                  "${generated_kernels_dir}/cagra_kernel_entrypoints/fatbin_${kernel_name}.cu"
              )
              # VPQ only supports L2Expanded, but we don't need to pass metric to the template
              # anymore
              set(team_size "${team_size}")
              set(dataset_block_dim "${dataset_block_dim}")
              set(pq_bits "${cagra_pq_bits}")
              set(pq_len "${pq_len}")
              set(codebook_type "${cagra_codebook_type}")
              set(pq_suffix "_${cagra_pq_bits}pq_${pq_len}subd")
              set(pq_prefix "")
              set(codebook_tag "tag_codebook_half")
              set(codebook_tag_comma ", ")
              set(query_type "half")
              set(query_type_abbrev "h")
              set(index_type "${cagra_index_type}")
              set(distance_type "${cagra_distance_type}")
              set(idx_abbrev "${cagra_index_abbrev}")
              set(dist_abbrev "${cagra_distance_abbrev}")
              configure_file(
                "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/search_single_cta_kernel_p.cu.in"
                "${filename}"
                @ONLY
              )
              embed_jit_lto_fatbin(
                FATBIN_TARGET "fatbin_${kernel_name}"
                FATBIN_SOURCE "${filename}"
                EMBEDDED_TARGET "${target}"
                EMBEDDED_HEADER "${generated_kernels_dir}/cagra_kernel_entrypoints/${kernel_name}.h"
                EMBEDDED_ARRAY "embedded_${kernel_name}"
              )
            endforeach() # pq_len
          endforeach() # dataset_block_dim
        endforeach() # team_size
      endforeach() # merge_idx
    endforeach() # topk_idx
  endforeach() # data_idx

  # Generate multi_cta kernel entrypoints Multi_cta kernels don't use topk_by_bitonic_sort or
  # bitonic_sort_and_merge_multi_warps as template parameters (those are handled inside the kernel
  # based on max_elements) IMPORTANT: Need to generate kernels for all combinations of team_size and
  # dataset_block_dim because the kernel template uses DescriptorT::kTeamSize and
  # DescriptorT::kDatasetBlockDim as template parameters when calling
  # setup_workspace_standard/compute_distance_standard CAGRA only uses uint32_t as SourceIndexT
  # (matching non-JIT path)
  foreach(data_idx IN ITEMS 0 1 2 3)
    list(GET cagra_data_types ${data_idx} data_type)
    list(GET cagra_data_type_abbrevs ${data_idx} type_abbrev)
    foreach(team_size IN LISTS cagra_team_sizes)
      foreach(dataset_block_dim IN LISTS cagra_dataset_block_dims)
        # CAGRA only uses uint32_t as SourceIndexT
        set(source_index_type "uint32_t")
        set(src_idx_abbrev "ui")
        # Generate QueryT variants: float for all, uint8_t only when DataT=uint8_t
        set(query_type_variants "float")
        set(query_type_abbrev_variants "f")
        if(data_idx EQUAL 2) # uint8_t
          list(APPEND query_type_variants "uint8_t")
          list(APPEND query_type_abbrev_variants "uc")
        endif()
        foreach(query_idx IN ITEMS 0 1)
          # Skip second iteration if we don't have uint8_t variant
          if(query_idx EQUAL 1 AND NOT data_idx EQUAL 2)
            break()
          endif()
          list(GET query_type_variants ${query_idx} query_type)
          list(GET query_type_abbrev_variants ${query_idx} query_type_abbrev)
          # Multi_cta kernel entrypoint (no metric in name)
          set(kernel_name
              "search_multi_cta_kernel_t${team_size}_dim${dataset_block_dim}_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}_${query_type_abbrev}_${src_idx_abbrev}"
          )
          set(filename "${generated_kernels_dir}/cagra_kernel_entrypoints/fatbin_${kernel_name}.cu")
          set(team_size "${team_size}")
          set(dataset_block_dim "${dataset_block_dim}")
          set(pq_bits "0")
          set(pq_len "0")
          set(codebook_type "void")
          set(pq_suffix "")
          set(pq_prefix "")
          set(codebook_tag "")
          set(codebook_tag_comma "")
          set(query_type "${query_type}")
          set(query_type_abbrev "${query_type_abbrev}")
          set(index_type "${cagra_index_type}")
          set(distance_type "${cagra_distance_type}")
          set(idx_abbrev "${cagra_index_abbrev}")
          set(dist_abbrev "${cagra_distance_abbrev}")
          configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/search_multi_cta_kernel.cu.in"
            "${filename}"
            @ONLY
          )
          embed_jit_lto_fatbin(
            FATBIN_TARGET "fatbin_${kernel_name}"
            FATBIN_SOURCE "${filename}"
            EMBEDDED_TARGET "${target}"
            EMBEDDED_HEADER "${generated_kernels_dir}/cagra_kernel_entrypoints/${kernel_name}.h"
            EMBEDDED_ARRAY "embedded_${kernel_name}"
          )
        endforeach() # query_type variant
      endforeach() # dataset_block_dim
    endforeach() # team_size
  endforeach() # data_idx

  # Generate multi_cta VPQ kernel entrypoints NOTE: VPQ is ONLY supported with L2Expanded metric
  # (not InnerProduct, CosineExpanded, etc.) VPQ kernels need pq_bits and pq_len in addition to
  # team_size and dataset_block_dim VPQ is supported for all data types (float, half, int8_t,
  # uint8_t) CAGRA only uses uint32_t as SourceIndexT
  foreach(data_idx IN ITEMS 0 1 2 3)
    list(GET cagra_data_types ${data_idx} data_type)
    list(GET cagra_data_type_abbrevs ${data_idx} type_abbrev)
    foreach(team_size IN LISTS cagra_team_sizes)
      foreach(dataset_block_dim IN LISTS cagra_dataset_block_dims)
        foreach(pq_len IN LISTS cagra_pq_lens)
          # CAGRA only uses uint32_t as SourceIndexT
          set(source_index_type "uint32_t")
          set(src_idx_abbrev "ui")
          # Multi_cta VPQ kernel entrypoint Note: Metric is no longer in the kernel name - VPQ only
          # supports L2Expanded
          set(kernel_name
              "search_multi_cta_kernel_vpq_t${team_size}_dim${dataset_block_dim}_${cagra_pq_bits}pq_${pq_len}subd_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}_h_${src_idx_abbrev}"
          )
          set(filename "${generated_kernels_dir}/cagra_kernel_entrypoints/fatbin_${kernel_name}.cu")
          # VPQ only supports L2Expanded, but we don't need to pass metric to the template anymore
          set(team_size "${team_size}")
          set(dataset_block_dim "${dataset_block_dim}")
          set(pq_bits "${cagra_pq_bits}")
          set(pq_len "${pq_len}")
          set(codebook_type "${cagra_codebook_type}")
          set(pq_suffix "_${cagra_pq_bits}pq_${pq_len}subd")
          set(pq_prefix "_vpq")
          set(codebook_tag "tag_codebook_half")
          set(codebook_tag_comma ", ")
          set(query_type "half")
          set(query_type_abbrev "h")
          set(index_type "${cagra_index_type}")
          set(distance_type "${cagra_distance_type}")
          set(idx_abbrev "${cagra_index_abbrev}")
          set(dist_abbrev "${cagra_distance_abbrev}")
          configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/search_multi_cta_kernel.cu.in"
            "${filename}"
            @ONLY
          )
          embed_jit_lto_fatbin(
            FATBIN_TARGET "fatbin_${kernel_name}"
            FATBIN_SOURCE "${filename}"
            EMBEDDED_TARGET "${target}"
            EMBEDDED_HEADER "${generated_kernels_dir}/cagra_kernel_entrypoints/${kernel_name}.h"
            EMBEDDED_ARRAY "embedded_${kernel_name}"
          )
        endforeach() # pq_len
      endforeach() # dataset_block_dim
    endforeach() # team_size
  endforeach() # data_idx

  # Generate multi_kernel kernel entrypoints Multi_kernel has two separate kernels:
  # random_pickup_kernel and compute_distance_to_child_nodes_kernel
  foreach(data_idx IN ITEMS 0 1 2 3)
    list(GET cagra_data_types ${data_idx} data_type)
    list(GET cagra_data_type_abbrevs ${data_idx} type_abbrev)
    foreach(team_size IN LISTS cagra_team_sizes)
      foreach(dataset_block_dim IN LISTS cagra_dataset_block_dims)
        # Generate QueryT variants: float for all, uint8_t only when DataT=uint8_t
        set(query_type_variants "float")
        set(query_type_abbrev_variants "f")
        if(data_idx EQUAL 2) # uint8_t
          list(APPEND query_type_variants "uint8_t")
          list(APPEND query_type_abbrev_variants "uc")
        endif()
        foreach(query_idx IN ITEMS 0 1)
          # Skip second iteration if we don't have uint8_t variant
          if(query_idx EQUAL 1 AND NOT data_idx EQUAL 2)
            break()
          endif()
          list(GET query_type_variants ${query_idx} query_type)
          list(GET query_type_abbrev_variants ${query_idx} query_type_abbrev)
          # random_pickup_kernel entrypoint (no metric in name) Note: random_pickup_kernel doesn't
          # use SourceIndexT, so no loop needed
          set(kernel_name
              "random_pickup_kernel_t${team_size}_dim${dataset_block_dim}_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}_${query_type_abbrev}"
          )
          set(filename "${generated_kernels_dir}/cagra_kernel_entrypoints/fatbin_${kernel_name}.cu")
          set(team_size "${team_size}")
          set(dataset_block_dim "${dataset_block_dim}")
          set(pq_bits "0")
          set(pq_len "0")
          set(codebook_type "void")
          set(pq_suffix "")
          set(pq_prefix "")
          set(codebook_tag "")
          set(codebook_tag_comma "")
          set(query_type "${query_type}")
          set(query_type_abbrev "${query_type_abbrev}")
          set(index_type "${cagra_index_type}")
          set(distance_type "${cagra_distance_type}")
          set(idx_abbrev "${cagra_index_abbrev}")
          set(dist_abbrev "${cagra_distance_abbrev}")
          configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/random_pickup_kernel.cu.in"
            "${filename}"
            @ONLY
          )
          embed_jit_lto_fatbin(
            FATBIN_TARGET "fatbin_${kernel_name}"
            FATBIN_SOURCE "${filename}"
            EMBEDDED_TARGET "${target}"
            EMBEDDED_HEADER "${generated_kernels_dir}/cagra_kernel_entrypoints/${kernel_name}.h"
            EMBEDDED_ARRAY "embedded_${kernel_name}"
          )

          # CAGRA only uses uint32_t as SourceIndexT
          set(source_index_type "uint32_t")
          set(src_idx_abbrev "ui")
          # compute_distance_to_child_nodes_kernel entrypoint (no metric in name)
          set(kernel_name
              "compute_distance_to_child_nodes_kernel_t${team_size}_dim${dataset_block_dim}_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}_${query_type_abbrev}_${src_idx_abbrev}"
          )
          set(filename "${generated_kernels_dir}/cagra_kernel_entrypoints/fatbin_${kernel_name}.cu")
          set(team_size "${team_size}")
          set(dataset_block_dim "${dataset_block_dim}")
          set(pq_bits "0")
          set(pq_len "0")
          set(codebook_type "void")
          set(pq_suffix "")
          set(pq_prefix "")
          set(codebook_tag "")
          set(codebook_tag_comma "")
          set(query_type "${query_type}")
          set(query_type_abbrev "${query_type_abbrev}")
          set(index_type "${cagra_index_type}")
          set(distance_type "${cagra_distance_type}")
          set(idx_abbrev "${cagra_index_abbrev}")
          set(dist_abbrev "${cagra_distance_abbrev}")
          configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/compute_distance_to_child_nodes_kernel.cu.in"
            "${filename}"
            @ONLY
          )
          embed_jit_lto_fatbin(
            FATBIN_TARGET "fatbin_${kernel_name}"
            FATBIN_SOURCE "${filename}"
            EMBEDDED_TARGET "${target}"
            EMBEDDED_HEADER "${generated_kernels_dir}/cagra_kernel_entrypoints/${kernel_name}.h"
            EMBEDDED_ARRAY "embedded_${kernel_name}"
          )
        endforeach() # query_type variant
      endforeach() # dataset_block_dim
    endforeach() # team_size
  endforeach() # data_idx

  # Generate multi_kernel VPQ kernel entrypoints VPQ kernels need pq_bits and pq_len in addition to
  # team_size and dataset_block_dim VPQ is supported for all data types (float, half, int8_t,
  # uint8_t)
  foreach(data_idx IN ITEMS 0 1 2 3)
    list(GET cagra_data_types ${data_idx} data_type)
    list(GET cagra_data_type_abbrevs ${data_idx} type_abbrev)
    foreach(team_size IN LISTS cagra_team_sizes)
      foreach(dataset_block_dim IN LISTS cagra_dataset_block_dims)
        foreach(pq_len IN LISTS cagra_pq_lens)
          # random_pickup_kernel VPQ entrypoint Note: Metric is no longer in the kernel name - VPQ
          # only supports L2Expanded
          set(kernel_name
              "random_pickup_kernel_vpq_t${team_size}_dim${dataset_block_dim}_${cagra_pq_bits}pq_${pq_len}subd_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}_h"
          )
          set(filename "${generated_kernels_dir}/cagra_kernel_entrypoints/fatbin_${kernel_name}.cu")
          # VPQ only supports L2Expanded, but we don't need to pass metric to the template anymore
          set(team_size "${team_size}")
          set(dataset_block_dim "${dataset_block_dim}")
          set(pq_bits "${cagra_pq_bits}")
          set(pq_len "${pq_len}")
          set(codebook_type "${cagra_codebook_type}")
          set(pq_suffix "_${cagra_pq_bits}pq_${pq_len}subd")
          set(pq_prefix "_vpq")
          set(codebook_tag "tag_codebook_half")
          set(codebook_tag_comma ", ")
          set(query_type "half")
          set(query_type_abbrev "h")
          set(index_type "${cagra_index_type}")
          set(distance_type "${cagra_distance_type}")
          set(idx_abbrev "${cagra_index_abbrev}")
          set(dist_abbrev "${cagra_distance_abbrev}")
          configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/random_pickup_kernel.cu.in"
            "${filename}"
            @ONLY
          )
          embed_jit_lto_fatbin(
            FATBIN_TARGET "fatbin_${kernel_name}"
            FATBIN_SOURCE "${filename}"
            EMBEDDED_TARGET "${target}"
            EMBEDDED_HEADER "${generated_kernels_dir}/cagra_kernel_entrypoints/${kernel_name}.h"
            EMBEDDED_ARRAY "embedded_${kernel_name}"
          )

          # CAGRA only uses uint32_t as SourceIndexT
          set(source_index_type "uint32_t")
          set(src_idx_abbrev "ui")
          # compute_distance_to_child_nodes_kernel VPQ entrypoint Note: Metric is no longer in the
          # kernel name - VPQ only supports L2Expanded
          set(kernel_name
              "compute_distance_to_child_nodes_kernel_vpq_t${team_size}_dim${dataset_block_dim}_${cagra_pq_bits}pq_${pq_len}subd_${type_abbrev}_${cagra_index_abbrev}_${cagra_distance_abbrev}_h_${src_idx_abbrev}"
          )
          set(filename "${generated_kernels_dir}/cagra_kernel_entrypoints/fatbin_${kernel_name}.cu")
          # VPQ only supports L2Expanded, but we don't need to pass metric to the template anymore
          set(team_size "${team_size}")
          set(dataset_block_dim "${dataset_block_dim}")
          set(pq_bits "${cagra_pq_bits}")
          set(pq_len "${pq_len}")
          set(codebook_type "${cagra_codebook_type}")
          set(pq_suffix "_${cagra_pq_bits}pq_${pq_len}subd")
          set(pq_prefix "_vpq")
          set(codebook_tag "tag_codebook_half")
          set(codebook_tag_comma ", ")
          set(query_type "half")
          set(query_type_abbrev "h")
          set(index_type "${cagra_index_type}")
          set(distance_type "${cagra_distance_type}")
          set(idx_abbrev "${cagra_index_abbrev}")
          set(dist_abbrev "${cagra_distance_abbrev}")
          configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/compute_distance_to_child_nodes_kernel.cu.in"
            "${filename}"
            @ONLY
          )
          embed_jit_lto_fatbin(
            FATBIN_TARGET "fatbin_${kernel_name}"
            FATBIN_SOURCE "${filename}"
            EMBEDDED_TARGET "${target}"
            EMBEDDED_HEADER "${generated_kernels_dir}/cagra_kernel_entrypoints/${kernel_name}.h"
            EMBEDDED_ARRAY "embedded_${kernel_name}"
          )
        endforeach() # pq_len
      endforeach() # dataset_block_dim
    endforeach() # team_size
  endforeach() # data_idx

  # Generate apply_filter_kernel entrypoints apply_filter_kernel doesn't use dataset_descriptor, so
  # it only needs index types CAGRA only uses uint32_t as SourceIndexT
  set(source_index_type "uint32_t")
  set(src_idx_abbrev "ui")
  set(kernel_name
      "apply_filter_kernel_${cagra_index_abbrev}_${cagra_distance_abbrev}_${src_idx_abbrev}"
  )
  set(filename "${generated_kernels_dir}/cagra_kernel_entrypoints/fatbin_${kernel_name}.cu")
  set(index_type "${cagra_index_type}")
  set(distance_type "${cagra_distance_type}")
  set(idx_abbrev "${cagra_index_abbrev}")
  set(dist_abbrev "${cagra_distance_abbrev}")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/apply_filter_kernel.cu.in"
    "${filename}"
    @ONLY
  )
  embed_jit_lto_fatbin(
    FATBIN_TARGET "fatbin_${kernel_name}"
    FATBIN_SOURCE "${filename}"
    EMBEDDED_TARGET "${target}"
    EMBEDDED_HEADER "${generated_kernels_dir}/cagra_kernel_entrypoints/${kernel_name}.h"
    EMBEDDED_ARRAY "embedded_${kernel_name}"
  )

  # Generate CAGRA sample filter fragments using shared implementation CAGRA only uses uint32_t as
  # SourceIndexT (matching non-JIT path)
  set(cagra_filter_configs "filter_none" "filter_bitset")
  foreach(filter_name IN LISTS cagra_filter_configs)
    set(source_index_type "uint32_t")
    set(src_idx_abbrev "ui")
    set(header_file "neighbors/detail/jit_lto_kernels/${filter_name}.cuh")
    set(kernel_name "${filter_name}_${src_idx_abbrev}")
    set(filename "${generated_kernels_dir}/cagra_filter_device_functions/fatbin_${kernel_name}.cu")
    set(namespace "cuvs::neighbors::detail")
    # Pass both filter_name (for include) and kernel_name (for registration)
    set(filter_name_var "${filter_name}")
    set(kernel_name_var "${kernel_name}")
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/jit_lto_kernels/filter.cu.in" "${filename}"
      @ONLY
    )
    embed_jit_lto_fatbin(
      FATBIN_TARGET "fatbin_${kernel_name}"
      FATBIN_SOURCE "${filename}"
      EMBEDDED_TARGET "${target}"
      EMBEDDED_HEADER "${generated_kernels_dir}/cagra_filter_device_functions/${kernel_name}.h"
      EMBEDDED_ARRAY "embedded_${kernel_name}"
    )
  endforeach() # filter_name
endfunction()

#=============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

function(find_and_configure_faiss)
    set(oneValueArgs VERSION REPOSITORY PINNED_TAG BUILD_STATIC_LIBS EXCLUDE_FROM_ALL ENABLE_GPU)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

        rapids_find_generate_module(faiss
                HEADER_NAMES  faiss/IndexFlat.h
                LIBRARY_NAMES faiss
                )

        set(BUILD_SHARED_LIBS ON)
        if (PKG_BUILD_STATIC_LIBS)
            set(BUILD_SHARED_LIBS OFF)
            set(CPM_DOWNLOAD_faiss ON)
        endif()

        include(cmake/modules/FindAVX.cmake)

        # Link against AVX CPU lib if it exists
        set(CUVS_FAISS_GLOBAL_TARGETS faiss::faiss)
        set(CUVS_FAISS_EXPORT_GLOBAL_TARGETS faiss)
        set(CUVS_FAISS_OPT_LEVEL "generic")
        if(CXX_AVX_FOUND)
            set(CUVS_FAISS_OPT_LEVEL "avx2")
            list(APPEND CUVS_FAISS_GLOBAL_TARGETS faiss::faiss_avx2)
            list(APPEND CUVS_FAISS_EXPORT_GLOBAL_TARGETS faiss_avx2)
        endif()

        rapids_cpm_find(faiss ${PKG_VERSION}
                GLOBAL_TARGETS ${CUVS_FAISS_GLOBAL_TARGETS}
                CPM_ARGS
                GIT_REPOSITORY   ${PKG_REPOSITORY}
                GIT_TAG          ${PKG_PINNED_TAG}
                EXCLUDE_FROM_ALL ${PKG_EXCLUDE_FROM_ALL}
                OPTIONS
                "FAISS_ENABLE_GPU ${PKG_ENABLE_GPU}"
                "FAISS_ENABLE_PYTHON OFF"
                "FAISS_OPT_LEVEL ${CUVS_FAISS_OPT_LEVEL}"
                "FAISS_USE_CUDA_TOOLKIT_STATIC ${CUDA_STATIC_RUNTIME}"
                "BUILD_TESTING OFF"
                "CMAKE_MESSAGE_LOG_LEVEL VERBOSE"
                )

        if(TARGET faiss AND NOT TARGET faiss::faiss)
            add_library(faiss::faiss ALIAS faiss)
        endif()

    if(CXX_AVX_FOUND)

        if(TARGET faiss_avx2 AND NOT TARGET faiss::faiss_avx2)
            add_library(faiss::faiss_avx2 ALIAS faiss_avx2)
        endif()
    endif()


    if(faiss_ADDED)
            rapids_export(BUILD faiss
                    EXPORT_SET faiss-targets
                    GLOBAL_TARGETS ${CUVS_FAISS_EXPORT_GLOBAL_TARGETS}
                    NAMESPACE faiss::)
        endif()

    # We generate the faiss-config files when we built faiss locally, so always do `find_dependency`
    rapids_export_package(BUILD OpenMP cuvs-ann-bench-exports) # faiss uses openMP but doesn't export a need for it
    rapids_export_package(BUILD faiss cuvs-ann-bench-exports GLOBAL_TARGETS ${CUVS_FAISS_GLOBAL_TARGETS} ${CUVS_FAISS_EXPORT_GLOBAL_TARGETS})
    rapids_export_package(INSTALL faiss cuvs-ann-bench-exports GLOBAL_TARGETS ${CUVS_FAISS_GLOBAL_TARGETS} ${CUVS_FAISS_EXPORT_GLOBAL_TARGETS})

    # Tell cmake where it can find the generated faiss-config.cmake we wrote.
    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(BUILD faiss [=[${CMAKE_CURRENT_LIST_DIR}]=]
                                    EXPORT_SET cuvs-ann-bench-exports)
endfunction()

if(NOT CUVS_FAISS_GIT_TAG)
    # TODO: Remove this once faiss supports FAISS_USE_CUDA_TOOLKIT_STATIC
    # (https://github.com/facebookresearch/faiss/pull/2446)
    set(CUVS_FAISS_GIT_TAG fea/statically-link-ctk)
    # set(CUVS_FAISS_GIT_TAG bde7c0027191f29c9dadafe4f6e68ca0ee31fb30)
endif()

if(NOT CUVS_FAISS_GIT_REPOSITORY)
    # TODO: Remove this once faiss supports FAISS_USE_CUDA_TOOLKIT_STATIC
    # (https://github.com/facebookresearch/faiss/pull/2446)
    set(CUVS_FAISS_GIT_REPOSITORY https://github.com/cjnolet/faiss.git)
    # set(CUVS_FAISS_GIT_REPOSITORY https://github.com/facebookresearch/faiss.git)
endif()

find_and_configure_faiss(VERSION    1.7.4
        REPOSITORY  ${CUVS_FAISS_GIT_REPOSITORY}
        PINNED_TAG  ${CUVS_FAISS_GIT_TAG}
        BUILD_STATIC_LIBS ${CUVS_USE_FAISS_STATIC}
        EXCLUDE_FROM_ALL ${CUVS_EXCLUDE_FAISS_FROM_ALL}
        ENABLE_GPU ${CUVS_FAISS_ENABLE_GPU})

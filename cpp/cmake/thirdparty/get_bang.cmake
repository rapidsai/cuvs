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

function(find_and_configure_bang)
  set(oneValueArgs VERSION REPOSITORY PINNED_TAG BUILD_STATIC_LIBS EXCLUDE_FROM_ALL ENABLE_GPU)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN} )

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(faiss version repository tag shallow exclude)

  rapids_cpm_find(bang ${version}
    GLOBAL_TARGETS bang bang::bang
    CPM_ARGS
    GIT_REPOSITORY ${repository}
    GIT_TAG ${tag}
    GIT_SHALLOW ${shallow}
    EXCLUDE_FROM_ALL ${exclude}
    )

  if(TARGET bang AND NOT TARGET bang::bang)
    add_library(bang::bang ALIAS bang)
  endif()

  if(bang_ADDED)
    rapids_export(BUILD bang
                  EXPORT_SET bang-targets
                  NAMESPACE bang::)
  endif()

endfunction()

find_and_configure_bang()

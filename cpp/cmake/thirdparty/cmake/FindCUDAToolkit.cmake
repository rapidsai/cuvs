# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# cmake-format: on
#
# Backport only the CUDA::bin2c target from Kitware/CMake v4.3.0's
# Modules/FindCUDAToolkit.cmake; delegate toolkit discovery to the installed CMake.
# The search hints also support conda's nvcc symlink layout. Remove this wrapper
# when the minimum CMake version supports both the target and that layout.
#
# Distributed under the OSI-approved BSD 3-Clause License. See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

include("${CMAKE_ROOT}/Modules/FindCUDAToolkit.cmake")

if(CUDAToolkit_FOUND AND NOT TARGET CUDA::bin2c)
  # Conda's targets/<arch>/bin contains only an nvcc symlink; bin2c is beside
  # its resolved executable in the environment's bin directory.
  set(_CUDAToolkit_bin2c_dirs "${CUDAToolkit_BIN_DIR}")
  if(CUDAToolkit_NVCC_EXECUTABLE)
    get_filename_component(_CUDAToolkit_nvcc_realpath "${CUDAToolkit_NVCC_EXECUTABLE}" REALPATH)
    get_filename_component(_CUDAToolkit_nvcc_real_dir "${_CUDAToolkit_nvcc_realpath}" DIRECTORY)
    list(APPEND _CUDAToolkit_bin2c_dirs "${_CUDAToolkit_nvcc_real_dir}")
  endif()
  find_program(CUDA_bin2c_EXECUTABLE
    NAMES bin2c
    HINTS ${_CUDAToolkit_bin2c_dirs}
    NO_DEFAULT_PATH
  )
  if(CUDA_bin2c_EXECUTABLE)
    add_executable(CUDA::bin2c IMPORTED)
    set_property(TARGET CUDA::bin2c PROPERTY IMPORTED_LOCATION "${CUDA_bin2c_EXECUTABLE}")
  endif()
  unset(_CUDAToolkit_bin2c_dirs)
  unset(_CUDAToolkit_nvcc_realpath)
  unset(_CUDAToolkit_nvcc_real_dir)
endif()

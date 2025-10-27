/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

module com.nvidia.cuvs {
  requires java.logging;

  exports com.nvidia.cuvs;
  exports com.nvidia.cuvs.spi;

  uses com.nvidia.cuvs.spi.CuVSServiceProvider;
}

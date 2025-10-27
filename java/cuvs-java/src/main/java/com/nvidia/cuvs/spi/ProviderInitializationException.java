/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.spi;

class ProviderInitializationException extends Exception {
  ProviderInitializationException(String message, Throwable cause) {
    super(message, cause);
  }

  public ProviderInitializationException(String message) {
    super(message);
  }
}

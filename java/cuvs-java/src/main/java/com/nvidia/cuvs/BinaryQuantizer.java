/*
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
package com.nvidia.cuvs;

import com.nvidia.cuvs.spi.CuVSProvider;
import java.util.Objects;

/**
 * {@link BinaryQuantizer} provides binary quantization functionality for datasets.
 *
 * Binary quantization applies a transform that changes any positive values to a
 * bitwise 1 and negative/zero values to 0.
 */
public interface BinaryQuantizer {

  /**
   * Applies binary quantization transform to given dataset.
   *
   * @param dataset a two-dimensional float array to transform
   * @return a two-dimensional byte array containing the binary quantized data
   */
  static byte[][] transform(CuVSResources cuvsResources, float[][] dataset) throws Throwable {
    Objects.requireNonNull(cuvsResources);
    Objects.requireNonNull(dataset);
    return CuVSProvider.provider().binaryQuantizerTransform(cuvsResources, dataset);
  }

  /**
   * Applies binary quantization transform to given dataset.
   *
   * @param dataset a {@link Dataset} object containing the vectors to transform
   * @return a two-dimensional byte array containing the binary quantized data
   */
  static byte[][] transform(CuVSResources cuvsResources, Dataset dataset) throws Throwable {
    Objects.requireNonNull(cuvsResources);
    Objects.requireNonNull(dataset);
    return CuVSProvider.provider().binaryQuantizerTransform(cuvsResources, dataset);
  }
}

/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
  // simple smoke test to make sure that we can compile the cagra.h API
  // using a c compiler. This isn't aiming to be a full test, just checking
  // that the exposed C-API is valid C code and doesn't contain C++ features
  cuvsCagraIndex_t index;
  cuvsCagraIndexCreate(&index);
  cuvsCagraIndexDestroy(index);
  return 0;
}

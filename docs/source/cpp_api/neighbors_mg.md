# Multi-GPU Nearest Neighbors

The Multi-GPU (SNMG - single-node multi-GPUs) nearest neighbors API provides a set of functions to deploy ANN indexes across multiple GPUs for improved performance and scalability.

`#include <cuvs/neighbors/common.hpp>`

namespace *cuvs::neighbors*

## Index build parameters

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_cpp_index_params

</div>

## Search parameters

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_cpp_search_params

</div>

## Index build

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_cpp_index_build

</div>

## Index extend

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_cpp_index_extend

</div>

## Index search

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_cpp_index_search

</div>

## Index serialize

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_cpp_serialize

</div>

## Index deserialize

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_cpp_deserialize

</div>

## Distribute pre-built local index

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_cpp_distribute

</div>

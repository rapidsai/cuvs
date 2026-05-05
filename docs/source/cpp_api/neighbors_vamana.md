# Vamana

Vamana is the graph construction algorithm behind the well-known DiskANN vector search solution. The cuVS implementation of Vamana/DiskANN is a custom GPU-acceleration version of the algorithm that aims to reduce index construction time using NVIDIA GPUs.

`#include <cuvs/neighbors/vamana.hpp>`

namespace *cuvs::neighbors::vamana*

## Index build parameters

<div class="doxygengroup" project="cuvs" members="" content-only="">

vamana_cpp_index_params

</div>

## Index

<div class="doxygengroup" project="cuvs" members="" content-only="">

vamana_cpp_index

</div>

## Index build

<div class="doxygengroup" project="cuvs" members="" content-only="">

vamana_cpp_index_build

</div>

## Index serialize

<div class="doxygengroup" project="cuvs" members="" content-only="">

vamana_cpp_serialize

</div>

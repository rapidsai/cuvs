# Multi-GPU Nearest Neighbors

The Multi-GPU (SNMG - single-node multi-GPUs) C API provides a set of functions to deploy ANN indexes across multiple GPUs for improved performance and scalability.

# Common Types and Enums

Common types and enums used across multi-GPU ANN algorithms.

`#include <cuvs/neighbors/mg_common.h>`

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_c_common_types

</div>

# Multi-GPU IVF-Flat

The Multi-GPU IVF-Flat method extends the IVF-Flat ANN algorithm to work across multiple GPUs. It provides two distribution modes: replicated (for higher throughput) and sharded (for handling larger datasets).

`#include <cuvs/neighbors/mg_ivf_flat.h>`

## IVF-Flat Index Build Parameters

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_ivf_flat_c_index_params

</div>

## IVF-Flat Index Search Parameters

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_ivf_flat_c_search_params

</div>

## IVF-Flat Index

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_ivf_flat_c_index

</div>

## IVF-Flat Index Build

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_ivf_flat_c_index_build

</div>

## IVF-Flat Index Search

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_ivf_flat_c_index_search

</div>

## IVF-Flat Index Extend

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_ivf_flat_c_index_extend

</div>

## IVF-Flat Index Serialize

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_ivf_flat_c_index_serialize

</div>

## IVF-Flat Index Deserialize

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_ivf_flat_c_index_deserialize

</div>

## IVF-Flat Index Distribute

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_ivf_flat_c_index_distribute

</div>

# Multi-GPU IVF-PQ

The Multi-GPU IVF-PQ method extends the IVF-PQ ANN algorithm to work across multiple GPUs. It provides two distribution modes: replicated (for higher throughput) and sharded (for handling larger datasets).

`#include <cuvs/neighbors/mg_ivf_pq.h>`

## IVF-PQ Index Build Parameters

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_ivf_pq_c_index_params

</div>

## IVF-PQ Index Search Parameters

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_ivf_pq_c_search_params

</div>

## IVF-PQ Index

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_ivf_pq_c_index

</div>

## IVF-PQ Index Build

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_ivf_pq_c_index_build

</div>

## IVF-PQ Index Search

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_ivf_pq_c_index_search

</div>

## IVF-PQ Index Extend

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_ivf_pq_c_index_extend

</div>

## IVF-PQ Index Serialize

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_ivf_pq_c_index_serialize

</div>

## IVF-PQ Index Deserialize

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_ivf_pq_c_index_deserialize

</div>

## IVF-PQ Index Distribute

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_ivf_pq_c_index_distribute

</div>

# Multi-GPU CAGRA

The Multi-GPU CAGRA method extends the CAGRA graph-based ANN algorithm to work across multiple GPUs. It provides two distribution modes: replicated (for higher throughput) and sharded (for handling larger datasets).

`#include <cuvs/neighbors/mg_cagra.h>`

## CAGRA Index Build Parameters

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_cagra_c_index_params

</div>

## CAGRA Index Search Parameters

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_cagra_c_search_params

</div>

## CAGRA Index

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_cagra_c_index

</div>

## CAGRA Index Build

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_cagra_c_index_build

</div>

## CAGRA Index Search

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_cagra_c_index_search

</div>

## CAGRA Index Extend

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_cagra_c_index_extend

</div>

## CAGRA Index Serialize

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_cagra_c_index_serialize

</div>

## CAGRA Index Deserialize

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_cagra_c_index_deserialize

</div>

## CAGRA Index Distribute

<div class="doxygengroup" project="cuvs" members="" content-only="">

mg_cagra_c_index_distribute

</div>

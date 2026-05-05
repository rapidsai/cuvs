# Multi-GPU Nearest Neighbors

The Multi-GPU (SNMG - single-node multi-GPUs) C API provides a set of functions to deploy ANN indexes across multiple GPUs for improved performance and scalability.

# Common Types and Enums

Common types and enums used across multi-GPU ANN algorithms.

`#include <cuvs/neighbors/mg_common.h>`

> **Generated API group:** `mg_c_common_types`
>
> project:  cuvs; members; content-only.

# Multi-GPU IVF-Flat

The Multi-GPU IVF-Flat method extends the IVF-Flat ANN algorithm to work across multiple GPUs. It provides two distribution modes: replicated (for higher throughput) and sharded (for handling larger datasets).

`#include <cuvs/neighbors/mg_ivf_flat.h>`

## IVF-Flat Index Build Parameters

> **Generated API group:** `mg_ivf_flat_c_index_params`
>
> project:  cuvs; members; content-only.

## IVF-Flat Index Search Parameters

> **Generated API group:** `mg_ivf_flat_c_search_params`
>
> project:  cuvs; members; content-only.

## IVF-Flat Index

> **Generated API group:** `mg_ivf_flat_c_index`
>
> project:  cuvs; members; content-only.

## IVF-Flat Index Build

> **Generated API group:** `mg_ivf_flat_c_index_build`
>
> project:  cuvs; members; content-only.

## IVF-Flat Index Search

> **Generated API group:** `mg_ivf_flat_c_index_search`
>
> project:  cuvs; members; content-only.

## IVF-Flat Index Extend

> **Generated API group:** `mg_ivf_flat_c_index_extend`
>
> project:  cuvs; members; content-only.

## IVF-Flat Index Serialize

> **Generated API group:** `mg_ivf_flat_c_index_serialize`
>
> project:  cuvs; members; content-only.

## IVF-Flat Index Deserialize

> **Generated API group:** `mg_ivf_flat_c_index_deserialize`
>
> project:  cuvs; members; content-only.

## IVF-Flat Index Distribute

> **Generated API group:** `mg_ivf_flat_c_index_distribute`
>
> project:  cuvs; members; content-only.

# Multi-GPU IVF-PQ

The Multi-GPU IVF-PQ method extends the IVF-PQ ANN algorithm to work across multiple GPUs. It provides two distribution modes: replicated (for higher throughput) and sharded (for handling larger datasets).

`#include <cuvs/neighbors/mg_ivf_pq.h>`

## IVF-PQ Index Build Parameters

> **Generated API group:** `mg_ivf_pq_c_index_params`
>
> project:  cuvs; members; content-only.

## IVF-PQ Index Search Parameters

> **Generated API group:** `mg_ivf_pq_c_search_params`
>
> project:  cuvs; members; content-only.

## IVF-PQ Index

> **Generated API group:** `mg_ivf_pq_c_index`
>
> project:  cuvs; members; content-only.

## IVF-PQ Index Build

> **Generated API group:** `mg_ivf_pq_c_index_build`
>
> project:  cuvs; members; content-only.

## IVF-PQ Index Search

> **Generated API group:** `mg_ivf_pq_c_index_search`
>
> project:  cuvs; members; content-only.

## IVF-PQ Index Extend

> **Generated API group:** `mg_ivf_pq_c_index_extend`
>
> project:  cuvs; members; content-only.

## IVF-PQ Index Serialize

> **Generated API group:** `mg_ivf_pq_c_index_serialize`
>
> project:  cuvs; members; content-only.

## IVF-PQ Index Deserialize

> **Generated API group:** `mg_ivf_pq_c_index_deserialize`
>
> project:  cuvs; members; content-only.

## IVF-PQ Index Distribute

> **Generated API group:** `mg_ivf_pq_c_index_distribute`
>
> project:  cuvs; members; content-only.

# Multi-GPU CAGRA

The Multi-GPU CAGRA method extends the CAGRA graph-based ANN algorithm to work across multiple GPUs. It provides two distribution modes: replicated (for higher throughput) and sharded (for handling larger datasets).

`#include <cuvs/neighbors/mg_cagra.h>`

## CAGRA Index Build Parameters

> **Generated API group:** `mg_cagra_c_index_params`
>
> project:  cuvs; members; content-only.

## CAGRA Index Search Parameters

> **Generated API group:** `mg_cagra_c_search_params`
>
> project:  cuvs; members; content-only.

## CAGRA Index

> **Generated API group:** `mg_cagra_c_index`
>
> project:  cuvs; members; content-only.

## CAGRA Index Build

> **Generated API group:** `mg_cagra_c_index_build`
>
> project:  cuvs; members; content-only.

## CAGRA Index Search

> **Generated API group:** `mg_cagra_c_index_search`
>
> project:  cuvs; members; content-only.

## CAGRA Index Extend

> **Generated API group:** `mg_cagra_c_index_extend`
>
> project:  cuvs; members; content-only.

## CAGRA Index Serialize

> **Generated API group:** `mg_cagra_c_index_serialize`
>
> project:  cuvs; members; content-only.

## CAGRA Index Deserialize

> **Generated API group:** `mg_cagra_c_index_deserialize`
>
> project:  cuvs; members; content-only.

## CAGRA Index Distribute

> **Generated API group:** `mg_cagra_c_index_distribute`
>
> project:  cuvs; members; content-only.

# Multi-GPU IVF-Flat

Multi-GPU IVF-Flat extends the IVF-Flat algorithm to work across multiple GPUs, providing improved scalability and performance for large-scale vector search. It supports both replicated and sharded distribution modes.

> **Note:**
> **IMPORTANT**: Multi-GPU IVF-Flat requires all data (datasets, queries, output arrays) to be in host memory (CPU).
> If using CuPy/device arrays, transfer to host with `array.get()` or `cp.asnumpy(array)` before use.

## Index build parameters

> **Python class:** `cuvs.neighbors.mg.ivf_flat.IndexParams`
>
> members.

## Index search parameters

> **Python class:** `cuvs.neighbors.mg.ivf_flat.SearchParams`
>
> members.

## Index

> **Python class:** `cuvs.neighbors.mg.ivf_flat.Index`
>
> members.

## Index build

> **Python function:** `cuvs.neighbors.mg.ivf_flat.build`

## Index search

> **Python function:** `cuvs.neighbors.mg.ivf_flat.search`

## Index extend

> **Python function:** `cuvs.neighbors.mg.ivf_flat.extend`

## Index save

> **Python function:** `cuvs.neighbors.mg.ivf_flat.save`

## Index load

> **Python function:** `cuvs.neighbors.mg.ivf_flat.load`

## Index distribute

> **Python function:** `cuvs.neighbors.mg.ivf_flat.distribute`

# Multi-GPU CAGRA

Multi-GPU CAGRA extends the graph-based CAGRA algorithm to work across multiple GPUs, providing improved scalability and performance for large-scale vector search. It supports both replicated and sharded distribution modes.

> **Note:**
> **IMPORTANT**: Multi-GPU CAGRA requires all data (datasets, queries, output arrays) to be in host memory (CPU).
> If using CuPy/device arrays, transfer to host with `array.get()` or `cp.asnumpy(array)` before use.

## Index build parameters

> **Python class:** `cuvs.neighbors.mg.cagra.IndexParams`
>
> members.

## Index search parameters

> **Python class:** `cuvs.neighbors.mg.cagra.SearchParams`
>
> members.

## Index

> **Python class:** `cuvs.neighbors.mg.cagra.Index`
>
> members.

## Index build

> **Python function:** `cuvs.neighbors.mg.cagra.build`

## Index search

> **Python function:** `cuvs.neighbors.mg.cagra.search`

## Index save

> **Python function:** `cuvs.neighbors.mg.cagra.save`

## Index load

> **Python function:** `cuvs.neighbors.mg.cagra.load`

## Index distribute

> **Python function:** `cuvs.neighbors.mg.cagra.distribute`

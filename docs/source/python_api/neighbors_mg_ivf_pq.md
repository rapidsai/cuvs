# Multi-GPU IVF-PQ

Multi-GPU IVF-PQ extends the IVF-PQ (Inverted File with Product Quantization) algorithm to work across multiple GPUs, providing improved scalability and performance for large-scale vector search. It supports both replicated and sharded distribution modes.

> [!NOTE]
> **IMPORTANT**: Multi-GPU IVF-PQ requires all data (datasets, queries, output arrays) to be in host memory (CPU). If using CuPy/device arrays, transfer to host with `array.get()` or `cp.asnumpy(array)` before use.

## Index build parameters

<div class="autoclass" members="">

cuvs.neighbors.mg.ivf_pq.IndexParams

</div>

## Index search parameters

<div class="autoclass" members="">

cuvs.neighbors.mg.ivf_pq.SearchParams

</div>

## Index

<div class="autoclass" members="">

cuvs.neighbors.mg.ivf_pq.Index

</div>

## Index build

<div class="autofunction">

cuvs.neighbors.mg.ivf_pq.build

</div>

## Index search

<div class="autofunction">

cuvs.neighbors.mg.ivf_pq.search

</div>

## Index extend

<div class="autofunction">

cuvs.neighbors.mg.ivf_pq.extend

</div>

## Index save

<div class="autofunction">

cuvs.neighbors.mg.ivf_pq.save

</div>

## Index load

<div class="autofunction">

cuvs.neighbors.mg.ivf_pq.load

</div>

## Index distribute

<div class="autofunction">

cuvs.neighbors.mg.ivf_pq.distribute

</div>

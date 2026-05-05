# Multi-GPU IVF-Flat

Multi-GPU IVF-Flat extends the IVF-Flat algorithm to work across multiple GPUs, providing improved scalability and performance for large-scale vector search. It supports both replicated and sharded distribution modes.

> [!NOTE]
> **IMPORTANT**: Multi-GPU IVF-Flat requires all data (datasets, queries, output arrays) to be in host memory (CPU). If using CuPy/device arrays, transfer to host with `array.get()` or `cp.asnumpy(array)` before use.

## Index build parameters

<div class="autoclass" members="">

cuvs.neighbors.mg.ivf_flat.IndexParams

</div>

## Index search parameters

<div class="autoclass" members="">

cuvs.neighbors.mg.ivf_flat.SearchParams

</div>

## Index

<div class="autoclass" members="">

cuvs.neighbors.mg.ivf_flat.Index

</div>

## Index build

<div class="autofunction">

cuvs.neighbors.mg.ivf_flat.build

</div>

## Index search

<div class="autofunction">

cuvs.neighbors.mg.ivf_flat.search

</div>

## Index extend

<div class="autofunction">

cuvs.neighbors.mg.ivf_flat.extend

</div>

## Index save

<div class="autofunction">

cuvs.neighbors.mg.ivf_flat.save

</div>

## Index load

<div class="autofunction">

cuvs.neighbors.mg.ivf_flat.load

</div>

## Index distribute

<div class="autofunction">

cuvs.neighbors.mg.ivf_flat.distribute

</div>

# IVF-PQ

The IVF-PQ method is an ANN algorithm. Like IVF-Flat, IVF-PQ splits the points into a number of clusters (also specified by a parameter called n_lists) and searches the closest clusters to compute the nearest neighbors (also specified by a parameter called n_probes), but it shrinks the sizes of the vectors using a technique called product quantization.

`#include <cuvs/neighbors/ivf_pq.hpp>`

namespace *cuvs::neighbors::ivf_pq*

## Index build parameters

<div class="doxygengroup" project="cuvs" members="" content-only="">

ivf_pq_cpp_index_params

</div>

## Index search parameters

<div class="doxygengroup" project="cuvs" members="" content-only="">

ivf_pq_cpp_search_params

</div>

## Index

<div class="doxygengroup" project="cuvs" members="" content-only="">

ivf_pq_cpp_index

</div>

## Index build

<div class="doxygengroup" project="cuvs" members="" content-only="">

ivf_pq_cpp_index_build

</div>

## Index extend

<div class="doxygengroup" project="cuvs" members="" content-only="">

ivf_pq_cpp_index_extend

</div>

## Index search

<div class="doxygengroup" project="cuvs" members="" content-only="">

ivf_pq_cpp_index_search

</div>

## Index serialize

<div class="doxygengroup" project="cuvs" members="" content-only="">

ivf_pq_cpp_serialize

</div>

## Helper Methods

Additional helper functions for manipulating the underlying data of an IVF-PQ index, unpacking records, and writing PQ codes into an existing IVF list.

namespace *cuvs::neighbors::ivf_pq::helpers*

<div class="doxygengroup" project="cuvs" members="" content-only="">

ivf_pq_cpp_helpers

</div>

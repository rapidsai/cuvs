Brute-force
===========

Brute-force, or flat index, is the most simple index type, as it ultimately boils down to an exhaustive matrix multiplication.

While it scales with :math:`O(N^2*D)`, brute-force can be a great choice when

1. exact nearest neighbors are required, and
2. when the number of vectors is relatively small (a few thousand to a few million)

Brute-force can also be a good choice for heavily filtered queries where other algorithms might struggle returning the expected results. For example,
when filtering out 90%-95% of the vectors from a search, the IVF methods could struggle to return anything at all with smaller number of probes and
graph-based algorithms with limited hash table memory could end up skipping over important unfiltered entries.

[ :doc:`C API <../c_api/neighbors_bruteforce_c>` | :doc:`C++ API <../cpp_api/neighbors_bruteforce>` | :doc:`Python API <../python_api/neighbors_brute_force>` | :doc:`Rust API <../rust_api/index>` ]

Filtering considerations
------------------------

Because it is exhaustive, brute-force can quickly become the slowest, albeit most accurate form of search. However, even
when the number of vectors in an index are very large, brute-force can still be used to search vectors efficiently with a filter.

This is especially true for cases where the filter is excluding 90%-99% of the vectors in the index where the partitioning
inherent in other approximate algorithms would simply not include expected vectors in the results. In the case of pre-filtered
brute-force, the computation is inverted so distances are only computed between vectors that pass the filter, significantly reducing
the amount of computation required.

Configuration parameters
------------------------

Build parameters
~~~~~~~~~~~~~~~~

None

Search Parameters
~~~~~~~~~~~~~~~~~

None


Tuning Considerations
---------------------

Brute-force is exact but that doesn't always mean it's deterministic. For example, when there are many nearest neighbors with
the same distances it's possible they might be ordered differently across different runs. This especially becomes apparent in
cases where there are points with the same distance right near the cutoff of `k`, which can cause the final list of neighbors
to differ from ground truth. This is not often a problem in practice and can usually be mitigated by increasing `k`.


Memory footprint
----------------

:math:`precision` is the number of bytes in each element of each vector (e.g. 32-bit = 4-bytes)


Index footprint
~~~~~~~~~~~~~~~

Raw vectors: :math:`n\_vectors * n\_dimensions * precision`

Vector norms (for distances which require them): :math:`n\_vectors * precision`

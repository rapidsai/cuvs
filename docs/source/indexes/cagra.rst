CAGRA
=====

CAGRA is a graph-based index that is based loosely on the navigable small-world graph (NSG) algorithm, but which has been
built from the ground-up specifically for the GPU. CAGRA constructs a flat graph representation by first building a kNN graph
of the training points and then removing redundant paths between neighbors.

The CAGRA algorithm has two basic steps-
1. Construct a kNN graph
2. Prune redundant routes from the kNN graph.

Brute-force could be used to construct the initial kNN graph. This would yield the most accurate graph but would be very slow and
we find that in practice the kNN graph does not need to be very accurate since the pruning step helps to boost the overall recall of
the index. cuVS provides IVF-PQ and NN-Descent strategies for building the initial kNN graph and these can be selected in index
 params object during index construction.

Interoperability with HNSW
--------------------------

[ C API | C++ API | Python API | Rust API ]


Configuration parameters
------------------------

Tuning Considerations
---------------------

Memory footprint
----------------


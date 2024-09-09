IVF-Flat
========

IVF-Flat is an inverted file index (IVF) algorithm, which in the context of nearest neighbors means that data points are
partitioned into clusters. At search time, brute-force is performed only in a (user-defined) subset of the closest clusters.
In practice, this algorithm can search the index much faster than brute-force and oftem still maintain an acceptable
recall, though this comes with the drawback that the index itself copies the original training vectors into a memory layout
that is optimized for fast memory reads and adds some additional memory storage overheads. Once the index is trained,
this algorithm no longer requires the original raw training vectors.

IVF-Flat tends to be a great choice when

1. like brute-force, there is enough device memory available to fit all of the vectors
in the index, and
2. exact recall is not needed. as with the other index types, the tuning parameters are used to trade-off recall for search latency / throughput.

[ C API | C++ API | Python API | Rust API ]

Configuration parameters
------------------------


Tuning Considerations
---------------------

Memory footprint
----------------


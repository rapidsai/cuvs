IVF-PQ
======

IVF-PQ is an inverted file index (IVF) algorithm, which is an extension to the IVF-Flat algorithm (e.g. data points are first
partitioned into clusters) where product quantization is performed within each cluster in order to shrink the memory footprint
of the index. Product quantization is a lossy compression method and it is capable of storing larger number of vectors
on the GPU by offloading the original vectors to main memory, however higher compression levels often lead to reduced recall.
Often a strategy called refinement reranking is employed to make up for the lost recall by querying the IVF-PQ index for a larger
`k` than desired and performing a reordering and reduction to `k` based on the distances from the unquantized vectors. Unfortunately,
this does mean that the unquantized raw vectors need to be available and often this can be done efficiently using multiple CPU threads.

[ C API | C++ API | Python API | Rust API ]


Configuration parameters
------------------------



Tuning Considerations
---------------------

Memory footprint
----------------


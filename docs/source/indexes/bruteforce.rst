Brute-force
===========

Brute-force, or flat index, is the most simple index type, as it ultimately boils down to an exhaustive matrix multiplication.

While it scales with O(N^2*D), brute-force can be a great choice when

1. exact nearest neighbors are required, and
2. when the number of vectors is relatively small (a few thousand to one million)


[ C API | C++ API | Python API | Rust API ]

Configuration parameters
------------------------



Tuning Considerations
---------------------

Memory footprint
----------------


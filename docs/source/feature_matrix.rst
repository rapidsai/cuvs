Vector Search Index Feature Matrix
===================================

The following table shows which vector search indexes are available in each language API and whether they support multi-GPU execution.

.. list-table::
   :widths: 20 8 8 8 8 8 8 12
   :header-rows: 1

   * - Index
     - C++
     - C
     - Python
     - Rust
     - Java
     - Go
     - Multi-GPU*
   * - All-Neighbors
     - ✅
     - ✅
     - ✅
     -
     -
     -
     - ✅
   * - Brute-Force
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     -
   * - CAGRA
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - HNSW
     - ✅
     - ✅
     - ✅
     -
     - ✅
     -
     -
   * - IVF-Flat
     - ✅
     - ✅
     - ✅
     - ✅
     -
     - ✅
     - ✅
   * - IVF-PQ
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - NN Descent
     - ✅
     -
     - ✅
     -
     -
     -
     -
   * - Vamana
     - ✅
     - ✅
     -
     - ✅
     -
     -
     -

\* Multi-GPU support is available in C++, C, and Python only.

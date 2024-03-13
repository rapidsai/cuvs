Interoperability
================

DLPack (C)
^^^^^^^^^^

Multi-dimensional span (C++)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

cuVS is built on top of the GPU-accelerated machine learning and data mining primitives in the `RAFT <https://github.com/rapidsai/raft>`_ library. Most of the C++ APIs in cuVS accept `mdspan <https://arxiv.org/abs/2010.06474>`_ multi-dimensional array view for representing data in higher dimensions similar to the `ndarray` in the Numpy Python library. RAFT also contains the corresponding owning `mdarray` structure, which simplifies the allocation and management of multi-dimensional data in both host and device (GPU) memory.

The `mdarray` is an owning object that forms a convenience layer over RMM and can be constructed in RAFT using a number of different helper functions:

.. code-block:: c++

    #include <raft/core/device_mdarray.hpp>
    
    int n_rows = 10;
    int n_cols = 10;
    
    auto scalar = raft::make_device_scalar<float>(handle, 1.0);
    auto vector = raft::make_device_vector<float>(handle, n_cols);
    auto matrix = raft::make_device_matrix<float>(handle, n_rows, n_cols);

The `mdspan` is a lightweight non-owning view that can wrap around any pointer, maintaining shape, layout, and indexing information for accessing elements.

We can construct `mdspan` instances directly from the above `mdarray` instances:

.. code-block:: c++

    // Scalar mdspan on device
    auto scalar_view = scalar.view();

    // Vector mdspan on device
    auto vector_view = vector.view();

    // Matrix mdspan on device
    auto matrix_view = matrix.view();

Since the `mdspan` is just a lightweight wrapper, we can also construct it from the underlying data handles in the `mdarray` instances above. We use the extent to get information about the `mdarray` or `mdspan`'s shape.

.. code-block:: c++

    #include <raft/core/device_mdspan.hpp>

    auto scalar_view = raft::make_device_scalar_view(scalar.data_handle());
    auto vector_view = raft::make_device_vector_view(vector.data_handle(), vector.extent(0));
    auto matrix_view = raft::make_device_matrix_view(matrix.data_handle(), matrix.extent(0), matrix.extent(1));

Of course, RAFT's `mdspan`/`mdarray` APIs aren't just limited to the `device`. You can also create `host` variants:

.. code-block:: c++

    #include <raft/core/host_mdarray.hpp>
    #include <raft/core/host_mdspan.hpp>

    int n_rows = 10;
    int n_cols = 10;

    auto scalar = raft::make_host_scalar<float>(handle, 1.0);
    auto vector = raft::make_host_vector<float>(handle, n_cols);
    auto matrix = raft::make_host_matrix<float>(handle, n_rows, n_cols);

    auto scalar_view = raft::make_host_scalar_view(scalar.data_handle());
    auto vector_view = raft::make_host_vector_view(vector.data_handle(), vector.extent(0));
    auto matrix_view = raft::make_host_matrix_view(matrix.data_handle(), matrix.extent(0), matrix.extent(1));

Please refer to RAFT's `mdspan` `documentation <https://docs.rapids.ai/api/raft/stable/cpp_api/mdspan/>`_ to learn more.


CUDA array interface (Python)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

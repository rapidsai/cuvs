Faiss
-----

Faiss v1.10.0 provides a special conda package that enables a cuVS backend for the Flat, IVF-Flat, IVF-PQ and CAGRA indexes on the GPU. Like the classical Faiss GPU indexes, the cuVS backend also enables interoperability between Faiss CPU indexes, allowing an index to be trained on GPU, searched on CPU, and vice versa.

The cuVS backend can be enabled by building Faiss from source with the `FAISS_ENABLE_CUVS` cmake flag enabled and setting the `use_cuvs` configuration option for the cuVS-enabled GPU indexes.

A pre-compiled conda package can also be installed using the following command:


.. code-block:: bash

    conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs=1.10.0

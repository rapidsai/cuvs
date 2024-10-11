FAISS
-----

FAISS v1.8 provides a special conda package that enables a RAFT backend for the Flat, IVF-Flat and IVF-PQ indexes on the GPU. Like the classical FAISS GPU indexes, the RAFT backend also enables interoperability between FAISS CPU indexes, allowing an index to be trained on GPU, searched on CPU, and vice versa.

The RAFT backend can be enabled by building FAISS from source with the `FAISS_USE_RAFT` cmake flag enabled and setting the `use_raft` configuration option for the RAFT-enabled GPU indexes.

A pre-compiled conda package can also be installed using the following command:

.. code-block:: bash

    conda install -c conda-forge -c pytorch -c rapidsai -c nvidia -c "nvidia/label/cuda-11.8.0" faiss-gpu-raft

The next release of FAISS will feature cuVS support as we continue to migrate the vector search algorithms from RAFT to cuVS.

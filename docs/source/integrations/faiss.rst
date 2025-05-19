Faiss
-----

Faiss v1.10.0 and beyond provides a special conda package that enables a cuVS backend for the Flat, IVF-Flat, IVF-PQ and CAGRA indexes on the GPU. Like the classical Faiss GPU indexes, the cuVS backend also enables interoperability between Faiss CPU indexes, allowing an index to be trained on GPU, searched on CPU, and vice versa.

The cuVS backend can be enabled by setting the appropriate cmake flag while building Faiss from source. A pre-compiled conda package can also be installed. Refer to `Faiss installation guidelines <https://github.com/facebookresearch/faiss/blob/main/INSTALL.md>`_ for more information.

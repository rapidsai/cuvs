Getting Started
===============

Welcome to cuVS, the premier library for GPU-accelerated vector search and clustering! cuVS provides several core building blocks for constructing new algorithms, as well as end-to-end vector search and clustering algorithms for use either standalone or through a growing list of :doc:`integrations <integrations>`.

There are several benefits to using cuVS and GPUs for vector search, including

#. Fast index build
#. Latency critical and high throughput search
#. Parameter tuning
#. Cost savings
#. Interoperability (build on GPU, deploy on CPU)
#. Multiple language support
#. Building blocks for composing new or accelerating existing algorithms

New to vector search?
=====================

If you are unfamiliar with the basics of vector search or how vector search differs from vector databases, then :doc:`this primer on vector search guide <vector_databases_vs_vector_search>` should provide some good insight. As outlined in the primer, vector search as used in vector databases is often closer to machine learning than to traditional databases. This means that while traditional databases can often be slow without any performance tuning, they will usually still yield the correct results. Unfortunately, vector search indexes, like other machine learning models, can yield garbage results of not tuned correctly. Fortunately, this opens up the whole world of hyperparamer optimization to improve vector search performance and quality. Please see our :doc:`index tuning guide <tuning_guide>` for more information.

When comparing the performance of vector search indexes, it is important that considerations are made with respect to three main dimensions:

#. Build time
#. Search quality
#. Search performance

Please see the :doc:`primer on comparing vector search index performance <comparing_indexes>`` for more information on methodologies and how to make a fair apples-to-apples comparison during your evaluations.

Using cuVS APIs
===============

cuVS is a C++ library its core, which is wrapped with a C library and exposed further through various different languages. cuVS currently provides APIs and documentation for :doc:`C <c_api>`, :doc:`C++ <cpp_api>`, :doc:`Python <python_api>`, and :doc:`Rust <rust_api/index>` with more languages in the works. our :doc:`API basics <api_basics>` provides some background and context about the important paradigms and vocabulary types you'll encounter when working with cuVS types.

Please refer to the :doc:`guide on API interoperability <api_interoperability>` for more information on how cuVS can work seamlessly with other libraries like numpy, cupy, tensorflow, and pytorch, even without having to copy device memory.


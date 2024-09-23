~~~~~~~~~~~~~~~
Getting Started
~~~~~~~~~~~~~~~

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

If you are unfamiliar with the basics of vector search or how vector search differs from vector databases, then :doc:`this primer on vector search guide <vector_databases_vs_vector_search>` should provide some good insight. As outlined in the primer, vector search as used in vector databases is often closer to machine learning than to traditional databases. This means that while traditional databases can often be slow without any performance tuning, they will usually still yield the correct results. Unfortunately, vector search indexes, like other machine learning models, can yield garbage results of not tuned correctly.

Fortunately, this opens up the whole world of hyperparamer optimization to improve vector search performance and quality. Please see our :doc:`index tuning guide <tuning_guide>` for more information.

When comparing the performance of vector search indexes, it is important that considerations are made with respect to three main dimensions:

#. Build time
#. Search quality
#. Search performance

Please see the :doc:`primer on comparing vector search index performance <comparing_indexes>` for more information on methodologies and how to make a fair apples-to-apples comparison during your evaluations.

Supported indexes
=================

cuVS supports many of the standard index types with the list continuing to grow and stay current with the state-of-the-art. Please refer to our :doc:`vector search index guide <indexes/indexes>` for to learn more about each individual index type, when they can be useful on the GPU, the tuning knobs they offer to trade off performance and quality.

The primary goal of cuVS is to enable speed, scale, and flexibility (in that order)- and one of the important value propositions is to enhance existing software deployments with extensible GPU capabilities to improve pain points while not interrupting parts of the system that work well today with CPU.


Using cuVS APIs
===============

cuVS is a C++ library its core, which is wrapped with a C library and exposed further through various different languages. cuVS currently provides APIs and documentation for :doc:`C <c_api>`, :doc:`C++ <cpp_api>`, :doc:`Python <python_api>`, and :doc:`Rust <rust_api/index>` with more languages in the works. our :doc:`API basics <api_basics>` provides some background and context about the important paradigms and vocabulary types you'll encounter when working with cuVS types.

Please refer to the :doc:`guide on API interoperability <api_interoperability>` for more information on how cuVS can work seamlessly with other libraries like numpy, cupy, tensorflow, and pytorch, even without having to copy device memory.


Where to next?
==============

cuVS is free and open source software, licesed under Apache 2.0 Once you are familiar with and/or have used cuVS, you can access the developer community most easily through :doc:`Github <https://github.com/rapidsai/cuvs>`. Please open Github issues for any bugs, questions or feature requests.

You can also access the RAPIDS community through :doc:`Slack <https://rapids.ai/slack-invite>`, :doc:`Stack Overflow <https://stackoverflow.com/tags/rapids>` and :doc:`X <https://twitter.com/rapidsai>`

We frequently publish blogs on GPU-enabled vector search, which can provide great deep dives into various important topics and breakthroughs:

#.  :doc:`Accelerating Vector Search with cuVS IVF-PQ <https://developer.nvidia.com/blog/accelerating-vector-search-rapids-cuvs-ivf-pq-deep-dive-part-1/>`
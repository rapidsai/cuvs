~~~~~~~~~~~~~~~
Getting Started
~~~~~~~~~~~~~~~

- `New to vector search?`_

  * :doc:`Primer on vector search <choosing_and_configuring_indexes>`

  * :doc:`Vector search indexes vs vector databases <vector_databases_vs_vector_search>`

  * :doc:`Index tuning guide <tuning_guide>`

  * :doc:`Comparing vector search index performance <comparing_indexes>`

- `Supported indexes`_

  * :doc:`Vector search index guide <neighbors/neighbors>`

- `Using cuVS APIs`_

  * :doc:`C API Docs <c_api>`

  * :doc:`C++ API Docs <cpp_api>`

  * :doc:`Python API Docs <python_api>`

  * :doc:`Rust API Docs <rust_api/index>`

  * :doc:`API basics <api_basics>`

  * :doc:`API interoperability <api_interoperability>`

- `Where to next?`_

  * `Social media`_

  * `Blogs`_

  * `Research`_

  * `Get involved`_

New to vector search?
=====================

If you are unfamiliar with the basics of vector search or how vector search differs from vector databases, then :doc:`this primer on vector search guide <choosing_and_configuring_indexes>` should provide some good insight. Another good resource for the uninitiated is our :doc:`vector databases vs vector search <vector_databases_vs_vector_search>` guide. As outlined in the primer, vector search as used in vector databases is often closer to machine learning than to traditional databases. This means that while traditional databases can often be slow without any performance tuning, they will usually still yield the correct results. Unfortunately, vector search indexes, like other machine learning models, can yield garbage results of not tuned correctly.

Fortunately, this opens up the whole world of hyperparamer optimization to improve vector search performance and quality. Please see our :doc:`index tuning guide <tuning_guide>` for more information.

When comparing the performance of vector search indexes, it is important that considerations are made with respect to three main dimensions:

#. Build time
#. Search quality
#. Search performance

Please see the :doc:`primer on comparing vector search index performance <comparing_indexes>` for more information on methodologies and how to make a fair apples-to-apples comparison during your evaluations.

Supported indexes
=================

cuVS supports many of the standard index types with the list continuing to grow and stay current with the state-of-the-art. Please refer to our :doc:`vector search index guide <neighbors/neighbors>` for to learn more about each individual index type, when they can be useful on the GPU, the tuning knobs they offer to trade off performance and quality.

The primary goal of cuVS is to enable speed, scale, and flexibility (in that order)- and one of the important value propositions is to enhance existing software deployments with extensible GPU capabilities to improve pain points while not interrupting parts of the system that work well today with CPU.


Using cuVS APIs
===============

cuVS is a C++ library at its core, which is wrapped with a C library and exposed further through various different languages. cuVS currently provides APIs and documentation for :doc:`C <c_api>`, :doc:`C++ <cpp_api>`, :doc:`Python <python_api>`, and :doc:`Rust <rust_api/index>` with more languages in the works. our :doc:`API basics <api_basics>` provides some background and context about the important paradigms and vocabulary types you'll encounter when working with cuVS types.

Please refer to the :doc:`guide on API interoperability <api_interoperability>` for more information on how cuVS can work seamlessly with other libraries like numpy, cupy, tensorflow, and pytorch, even without having to copy device memory.


Where to next?
==============

cuVS is free and open source software, licesed under Apache 2.0 Once you are familiar with and/or have used cuVS, you can access the developer community most easily through `Github <https://github.com/rapidsai/cuvs>`_. Please open Github issues for any bugs, questions or feature requests.

Social media
------------

You can access the RAPIDS community through `Slack <https://rapids.ai/slack-invite>`_ , `Stack Overflow <https://stackoverflow.com/tags/rapids>`_ and `X <https://twitter.com/rapidsai>`_

Blogs
-----

We frequently publish blogs on GPU-enabled vector search, which can provide great deep dives into various important topics and breakthroughs:

#. `Accelerated Vector Search: Approximating with cuVS IVF-Flat <https://developer.nvidia.com/blog/accelerated-vector-search-approximating-with-rapids-raft-ivf-flat/>`_
#. `Accelerating Vector Search with cuVS IVF-PQ <https://developer.nvidia.com/blog/accelerating-vector-search-rapids-cuvs-ivf-pq-deep-dive-part-1/>`_

Research
--------

For the interested reader, many of the accelerated implementations in cuVS are also based on research papers which can provide a lot more background. We also ask you to please cite the corresponding algorithms by referencing them in your own research.

#. `CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search <https://arxiv.org/abs/2308.15136>`_
#. `Top-K Algorithms on GPU: A Comprehensive Study and New Methods <https://dl.acm.org/doi/10.1145/3581784.3607062>`_
#. `Fast K-NN Graph Construction by GPU Based NN-Descent <https://dl.acm.org/doi/abs/10.1145/3459637.3482344?casa_token=O_nan1B1F5cAAAAA:QHWDEhh0wmd6UUTLY9_Gv6c3XI-5DXM9mXVaUXOYeStlpxTPmV3nKvABRfoivZAaQ3n8FWyrkWw>`_
#. `cuSLINK: Single-linkage Agglomerative Clustering on the GPU <https://arxiv.org/abs/2306.16354>`_
#. `GPU Semiring Primitives for Sparse Neighborhood Methods <https://arxiv.org/abs/2104.06357>`_


Get involved
------------

We always welcome patches for new features and bug fixes. Please read our `contributing guide <contributing.md>`_ for more information on contributing patches to cuVS.



.. toctree::
   :hidden:

   choosing_and_configuring_indexes.rst
   vector_databases_vs_vector_search.rst
   tuning_guide.rst
   comparing_indexes.rst
   neighbors/neighbors.rst
   api_basics.rst
   api_interoperability.rst
   working_with_ann_indexes.rst
   filtering.rst

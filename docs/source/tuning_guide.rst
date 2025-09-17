~~~~~~~~~~~~~~~~~~~~~~
Automated tuning Guide
~~~~~~~~~~~~~~~~~~~~~~

Introduction
============

A Method for tuning and evaluating Vector Search Indexes At Scale in Locally Indexed Vector Databases. For more information on the differences between locally and globally indexed vector databases, please see :doc:`this guide <vector_databases_vs_vector_search>`. The goal of this guide is to give users a scalable and effective approach for tuning a vector search index, no matter how large.  Evaluation of a vector search index “model” that measures recall in proportion to build time so that it penalizes the recall when the build time is really high (should ultimately optimize for finding a lower build time and higher recall).

For more information on the various different types of vector search indexes, please see our :doc:`guide to choosing vector search indexes <choosing_and_configuring_indexes>`

Why automated tuning?
=====================

As much as 75% of users have told us they will not be able to tune a vector database beyond one or two simple knobs and we suggest that an ideal “knob” would be to balance training time and search time with search quality. The more time, the higher the quality, and the more needed to find an acceptable search performance. Even the 25% of users that want to tune are still asking for simple tools for doing so. These users also ask for some simple guidelines for setting tuning parameters, like :doc:`this guide <neighbors/neighbors>`.

Since vector search indexes are more closely related to machine learning models than traditional databases indexes, one option for easing the parameter tuning burden is to use hyper-parameter optimization tools like `Ray Tune <https://medium.com/rapids-ai/30x-faster-hyperparameter-search-with-raytune-and-rapids-403013fbefc5>`_ and `Optuna <https://docs.rapids.ai/deployment/stable/examples/rapids-optuna-hpo/notebook/>`_. to verify this.

How to tune?
============

But how would this work when we have an index that's massively large- like 1TB?

One benefit to locally indexed vector databases is that they often scale by breaking the larger set of vectors down into a smaller set by uniformly random subsampling and training smaller vector search index models on the sub-samples. Most often, the same set of tuning parameters are applied to all of the smaller sub-index models, rather than trying to set them individually for each one. During search, the query vectors are often sent to all of the sub-indexes and the resulting neighbors list reduced down to `k` based on the closest distances (or similarities).

Because many databases use this sub-sampling trick, it's possible to perform an automated parameter tuning on the larger index just by randomly samplnig some number of vectors from it, splitting them into disjoint train/test/eval datasets, computing ground truth with brute-force, and then performing a hyper-parameter optimization on it. This procedure can also be repeated multiple times to simulate a monte-carlo cross validation.

GPUs are naturally great at performing massively parallel tasks, especially when they are largely independent tasks, such as training and evaluating models with different hyper-parameter settings in parallel. Hyper-parameter optimization also lends itself well to distributed processing, such as multi-node multi-GPU operation.

Steps to achieve automated tuning
=================================

More formally, an automated parameter tuning workflow with monte-carlo cross-validaton looks likes something like this:

#. Ingest a large dataset into the vector database of your choice

#. Choose an index size based on number of vectors. This should usually align with the average number of vectors the database will end up putting in a single ANN sub-index model.

#. Uniformly random sample the number of vectors specified above from the database for a training set. This is often accomplished by generating some number of random (unique) numbers up to the dataset size.

#. Uniformly sample some number of vectors for a test set and do this again for an evaluation set. 1-10% of the vectors in the training set.

#. Use the test set to compute ground truth on the vectors from prior step against all vectors in the training set.

#. Start the HPO tuning process for the training set, using the test vectors for the query set. It's important to make sure your HPO is multi-objective and optimizes for: a) low build time, b) high throughput or low latency sarch (depending on needs), and c) acceptable recall.

#. Use the evaluation dataset to test that the optimal hyper-parameters generalize to unseen points that were not used in the optimization process.

#. Optionally, the above steps multiple times on different uniform sub-samplings. Optimal parameters can then be combined over the multiple monte-carlo optimization iterations. For example, many hyper-parameters can simply be averaged but care might need to be taken for other parameters.

#. Create a new index in the database using the ideal params from above that meet the target constraints (e.g. build vs search vs quality)

Conclusion
==========

By the end of this process, you should have a set of parameters that meet your target constraints while demonstrating how well the optimal hyper-parameters generalize across the dataset. The major benefit to this approach is that it breaks a potentially unbounded dataset size down into manageable chunks and accelerates tuning on those chunks. We see this process as a major value add for vector search on the GPU.

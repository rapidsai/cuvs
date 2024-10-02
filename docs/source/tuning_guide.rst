.. _tuning_guide:

~~~~~~~~~~~~
Tuning Guide
~~~~~~~~~~~~

A Method for tuning and evaluating Vector Search Indexes At Scale in Locally Indexed Vector Databases. For more information on the differences between locally and globally indexed vector databases, please see :doc:`this guide <vector_databases_vs_vector_search>`. The goal of this guide is to give users a scalable and effective approach for tuning a vector search index, no matter how large.  Evaluation of a vector search index “model” that measures recall in proportion to build time so that it penalizes the recall when the build time is really high (should ultimately optimize for finding a lower build time and higher recall).

For more information on the various different types of vector search indexes, please see our :doc:`guide to choosing vector search indexes <choosing_and_configuring_indexes>`

As much as 75% of users have told us they will not be able to tune a vector database beyond one or two simple knobs and we suggest that an ideal “knob” would be to balance training time and search time with search quality. The more time, the higher the quality, and the more needed to find an acceptable search performance. Even the 25% of users that want to tune are still asking for simple tools for doing so. These users also ask for some simple guidelines for setting tuning parameters, like :doc:`this guide <indexes/indexes>`.

Since vector search indexes are more closely related to machine learning models than traditional databases indexes, one option for easing the parameter tuning burden is to use hyper-parameter optimization tools like `Ray Tune <https://medium.com/rapids-ai/30x-faster-hyperparameter-search-with-raytune-and-rapids-403013fbefc5>`_ and `Optuna <https://docs.rapids.ai/deployment/stable/examples/rapids-optuna-hpo/notebook/>`_. to verify this.

But how would this work when we have an index that's massively large- like 1TB?

One benefit to locally indexed vector databases is that they tend to scale by breaking a larger set of vectors down into smaller small subsampling from a dataset and does a parameter search on it. Then we actually evaluate random queries against the ground truth to test that the index params actually generalized well (I'm confident they will).

Getting Started with Optuna and RAPIDS for HPO — RAPIDS Deployment Documentation documentation

Ray tune / Optune should allow us to plug in cuvs' Python API trivially and then we just specify a bunch of params to tune and let it go to town- this would ideally be done on a multi-node multi-GPU setup where we can try 10's of combinations at once, starting with "empirical heuristics" as defaults and iterate through something like a bayesian optimizer to find the best params.

#. Generate a dataset with a reasonable number of vectors (say 10Mx768)
#. Subsample from the population uniformly, let's say 10% of that (1M vectors)
#. Subsample from the population uniformly, let's say 1% of the 1M vectors from the prior step, this is a validation set.
#. Compute ground truth on the vectors from prior step against all 10M vectors
#. Start tuning process for the 1M vectors from step 2 using the vectors from step 3 as the query set
#. Using the ideal params that provide the target objective (e.g. build vs quality), ingest all 10M vectors into the database and create an index.
#. Query the vectors from the database and calculate the recall. Verify it's close to the recall from the model params chosen in 5 (within some small epsilon). .


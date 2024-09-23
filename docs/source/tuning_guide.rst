.. _tuning_guide:

~~~~~~~~~~~~
Tuning Guide
~~~~~~~~~~~~

A Method for tuning and evaluating Vector Search Indexes At Scale in Locally Indexed Vector Databases

Objective
=========

Give uswrs an approach for tuning a vector search index. Evaluation of a vector search index “model” that measures recall in proportion to build time so that it penalizes the recall when the build time is really high (should ultimately optimize for finding a lower build time and higher recall).

Output
======
An example notebook which can be released in cuVS as an example for tuning an index, especially for CAGRA.

Background
==========

Vector databases 101: Configuring Vector Search Indexes

Many customers (Specifically AWS and Google) have told us that >75% of their users will not be able to tune a vector database beyond one or two simple knobs. They suggest that an ideal “knob” would be to balance training time with search quality. The more time, the higher the quality. For the <25% that wants to tune, they’ve asked for simple tools for tuning. They also ask for some simple guidelines for setting tuning parameters.
Strategy
Ray-tune and our Python APIs could be an option to verify this. We could write a notebook that takes some small subsampling from a dataset and does a parameter search on it. Then we actually evaluate random queries against the ground truth to test that the index params actually generalized well (I'm confident they will).

Getting Started with Optuna and RAPIDS for HPO — RAPIDS Deployment Documentation documentation

Ray tune / Optune should allow us to plug in cuvs' Python API trivially and then we just specify a bunch of params to tune and let it go to town- this would ideally be done on a multi-node multi-GPU setup where we can try 10's of combinations at once, starting with "empirical heuristics" as defaults and iterate through something like a bayesian optimizer to find the best params.

#. Generate a dataset with a reasonable number of vectors (say 10Mx768)
#. Subsample from the population uniformly, let's say 10% of that (1M vectors)
#. Subsample from the population uniformly, let's say 1% of the 1M vectors from the prior step, this is a validation set.
#. Compute ground truth on the vectors from prior step against all 10M vectors
#. Start tuning process for the 1M vectors from step 2 using the vectors from step 3 as the query set
#. Using the ideal params that provide the target objective (e.g. build vs quality), ingest all 10M vectors into the database and create an index.
#. Query the vectors from the database and calculate the recall. Verify it's close to the recall from the model params chosen in 5 (within some small epsilon). .


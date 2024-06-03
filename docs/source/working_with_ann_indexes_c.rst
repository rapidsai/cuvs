Working with ANN Indexes in C
=============================

- `Building an index`_
- `Searching an index`_
- `CPU/GPU Interoperability`_
- `Serializing an index`_

Building an index
-----------------

.. code-block:: c

    #include <cuvs/neighbors/cagra.h>

    cuvsResources_t res;
    cuvsCagraIndexParams_t index_params;
    cuvsCagraIndex_t index;

    DLManagedTensor *dataset;

    // populate tensor with data
    load_dataset(dataset);

    cuvsResourcesCreate(&res);
    cuvsCagraIndexParamsCreate(&index_params);
    cuvsCagraIndexCreate(&index);

    cuvsCagraBuild(res, index_params, dataset, index);

    cuvsCagraIndexDestroy(index);
    cuvsCagraIndexParamsDestroy(index_params);
    cuvsResourcesDestroy(res);


Searching an index
------------------

.. code-block:: c

    #include <cuvs/neighbors/cagra.h>

    cuvsResources_t res;
    cuvsCagraSearchParams_t search_params;
    cuvsCagraIndex_t index;

    // ... build index ...

    DLManagedTensor *queries;

    DLManagedTensor *neighbors;
    DLManagedTensor *distances;

    // populate tensor with data
    load_queries(queries);

    cuvsResourcesCreate(&res);
    cuvsCagraSearchParamsCreate(&index_params);

    cuvsCagraSearch(res, search_params, index, queries, neighbors, distances);

    cuvsCagraIndexDestroy(index);
    cuvsCagraIndexParamsDestroy(index_params);
    cuvsResourcesDestroy(res);

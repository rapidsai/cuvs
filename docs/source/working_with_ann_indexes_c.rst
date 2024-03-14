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


Serializing an index
--------------------
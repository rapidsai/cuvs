Working with ANN Indexes
========================

- `Building an index`_
- `Searching an index`_
- `CPU/GPU Interoperability`_
- `Serializing an index`_

Building an index
-----------------

C
^

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

C++
^^^

.. code-block:: c++

    #include <cuvs/neighbors/cagra.hpp>

    using namespace cuvs::neighbors;

    raft::device_matrix_view<float> dataset = load_dataset();
    raft::device_resources res;

    cagra::index_params index_params;

    auto index = cagra::build(res, index_params, dataset);


Python
^^^^^^

.. code-block:: python

    from cuvs.neighbors import cagra

    dataset = load_data()
    index_params = cagra.IndexParams()

    index = cagra.build_index(build_params, dataset)

Rust
^^^^

.. code-block:: rust

    use cuvs::cagra::{Index, IndexParams};
    use cuvs::{Resources, Result};

    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    /// Example showing how to index and search data with CAGRA
    fn cagra_example() -> Result<()> {
        let res = Resources::new()?;

        // Create a new random dataset to index
        let n_datapoints = 65536;
        let n_features = 512;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

        // build the cagra index
        let build_params = IndexParams::new()?;
        let index = Index::build(&res, &build_params, &dataset)?;

        Ok(())
    }


Searching an index
------------------


CPU/GPU interoperability
------------------------

Serializing an index
--------------------
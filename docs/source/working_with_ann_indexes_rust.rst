Working with ANN Indexes in Rust
================================

- `Building an index`_
- `Searching an index`_
- `CPU/GPU Interoperability`_
- `Serializing an index`_

Building an index
-----------------

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

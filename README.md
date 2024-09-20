# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;cuVS: Vector Search and Clustering on the GPU</div>

> [!note]
> cuVS is a new library mostly derived from the approximate nearest neighbors and clustering algorithms in the [RAPIDS RAFT](https://github.com/rapidsai/raft) library of data mining primitives. RAPIDS RAFT currently contains the most fully-featured versions of the approximate nearest neighbors and clustering algorithms in cuVS. We are in the process of migrating the algorithms from RAFT to cuVS, but if you are unsure of which to use, please consider the following:
> 1. RAFT contains C++ and Python APIs for all of the approximate nearest neighbors and clustering algorithms.
> 2. cuVS contains a growing support for different languages, including C, C++, Python, and Rust. We will be adding more language support to cuVS in the future but will not be improving the language support for RAFT.
> 3. Once all of RAFT's approximate nearest neighbors and clustering algorithms are moved to cuVS, the RAFT APIs will be deprecated and eventually removed altogether. Once removed, RAFT will become a lightweight header-only library. In the meantime, there's no harm in using RAFT if support for additional languages is not needed.


## Contents

1. [Useful Resources](#useful-resources)
2. [What is cuVS?](#what-is-cuvs)
3. [Installing cuVS](#installing-cuvs)
4. [Getting Started](#getting-started)
5. [Contributing](#contributing)
6. [References](#references)

## Useful Resources

- [Code Examples](https://github.com/rapidsai/cuvs/tree/HEAD/examples): Self-contained Code Examples.
- [API Reference Documentation](https://docs.rapids.ai/api/cuvs/nightly/api_docs): API Documentation.
- [Getting Started Guide](https://docs.rapids.ai/api/cuvs/nightly/getting_started): Getting started with RAFT.
- [Build and Install Guide](https://docs.rapids.ai/api/cuvs/nightly/build): Instructions for installing and building cuVS.
- [RAPIDS Community](https://rapids.ai/community.html): Get help, contribute, and collaborate.
- [GitHub repository](https://github.com/rapidsai/cuvs): Download the cuVS source code.
- [Issue tracker](https://github.com/rapidsai/cuvs/issues): Report issues or request features.

## What is cuVS?

cuVS contains state-of-the-art implementations of several algorithms for running approximate nearest neighbors and clustering on the GPU. It can be used directly or through the various databases and other libraries that have integrated it. The primary goal of cuVS is to simplify the use of GPUs for vector similarity search and clustering.

## Installing cuVS

cuVS comes with pre-built packages that can be installed through [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-python). Different packages are available for the different languages supported by cuVS:

| Python | C/C++                       |
|--------|-----------------------------|
| `cuvs` | `libcuvs`, `libcuvs-static` |

### Stable release

It is recommended to use [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) to install the desired packages. The following command will install the Python package. You can substitute `cuvs` for any of the packages in the table above:

```bash
mamba install -c conda-forge -c nvidia -c rapidsai cuvs
```

### Nightlies
If installing a version that has not yet been released, the `rapidsai` channel can be replaced with `rapidsai-nightly`:

```bash
mamba install -c conda-forge -c nvidia -c rapidsai-nightly cuvs=24.10
```

Please see the [Build and Install Guide](https://docs.rapids.ai/api/cuvs/stable/build/) for more information on installing cuVS and building from source.

## Getting Started

The following code snippets train an approximate nearest neighbors index for the CAGRA algorithm.

### Python API

```python
from cuvs.neighbors import cagra

dataset = load_data()
index_params = cagra.IndexParams()

index = cagra.build(build_params, dataset)
```

### C++ API

```c++
#include <cuvs/neighbors/cagra.hpp>

using namespace cuvs::neighbors;

raft::device_matrix_view<float> dataset = load_dataset();
raft::device_resources res;

cagra::index_params index_params;

auto index = cagra::build(res, index_params, dataset);
```

For more examples of the C++ APIs, refer to the [examples](https://github.com/rapidsai/cuvs/tree/HEAD/examples) directory in the codebase.

### C API

```c
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
```

### Rust API

```rust
use cuvs::cagra::{Index, IndexParams, SearchParams};
use cuvs::{ManagedTensor, Resources, Result};

use ndarray::s;
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
    println!(
        "Indexed {}x{} datapoints into cagra index",
        n_datapoints, n_features
    );

    // use the first 4 points from the dataset as queries : will test that we get them back
    // as their own nearest neighbor
    let n_queries = 4;
    let queries = dataset.slice(s![0..n_queries, ..]);

    let k = 10;

    // CAGRA search API requires queries and outputs to be on device memory
    // copy query data over, and allocate new device memory for the distances/ neighbors
    // outputs
    let queries = ManagedTensor::from(&queries).to_device(&res)?;
    let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
    let neighbors = ManagedTensor::from(&neighbors_host).to_device(&res)?;

    let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
    let distances = ManagedTensor::from(&distances_host).to_device(&res)?;

    let search_params = SearchParams::new()?;

    index.search(&res, &search_params, &queries, &neighbors, &distances)?;

    // Copy back to host memory
    distances.to_host(&res, &mut distances_host)?;
    neighbors.to_host(&res, &mut neighbors_host)?;

    // nearest neighbors should be themselves, since queries are from the
    // dataset
    println!("Neighbors {:?}", neighbors_host);
    println!("Distances {:?}", distances_host);
    Ok(())
}
```


## Contributing

If you are interested in contributing to the cuVS library, please read our [Contributing guidelines](docs/source/contributing.md). Refer to the [Developer Guide](docs/source/developer_guide.md) for details on the developer guidelines, workflows, and principles.

## References

When citing cuVS generally, please consider referencing this Github repository.
```bibtex
@misc{rapidsai,
  title={Rapidsai/cuVS: Vector Search and Clustering on the GPU.},
  url={https://github.com/rapidsai/cuvs},
  journal={GitHub},
  publisher={Nvidia RAPIDS},
  author={Rapidsai},
  year={2024}
}
```

If citing CAGRA, please consider the following bibtex:
```bibtex
@misc{ootomo2023cagra,
      title={CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs},
      author={Hiroyuki Ootomo and Akira Naruse and Corey Nolet and Ray Wang and Tamas Feher and Yong Wang},
      year={2023},
      eprint={2308.15136},
      archivePrefix={arXiv},
      primaryClass={cs.DS}
}
```

If citing the k-selection routines, please consider the following bibtex:
```bibtex
@proceedings{10.1145/3581784,
    title = {SC '23: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
    year = {2023},
    isbn = {9798400701092},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    abstract = {Started in 1988, the SC Conference has become the annual nexus for researchers and practitioners from academia, industry and government to share information and foster collaborations to advance the state of the art in High Performance Computing (HPC), Networking, Storage, and Analysis.},
    location = {, Denver, CO, USA, }
}
```

If citing the nearest neighbors descent API, please consider the following bibtex:
```bibtex
@inproceedings{10.1145/3459637.3482344,
    author = {Wang, Hui and Zhao, Wan-Lei and Zeng, Xiangxiang and Yang, Jianye},
    title = {Fast K-NN Graph Construction by GPU Based NN-Descent},
    year = {2021},
    isbn = {9781450384469},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3459637.3482344},
    doi = {10.1145/3459637.3482344},
    abstract = {NN-Descent is a classic k-NN graph construction approach. It is still widely employed in machine learning, computer vision, and information retrieval tasks due to its efficiency and genericness. However, the current design only works well on CPU. In this paper, NN-Descent has been redesigned to adapt to the GPU architecture. A new graph update strategy called selective update is proposed. It reduces the data exchange between GPU cores and GPU global memory significantly, which is the processing bottleneck under GPU computation architecture. This redesign leads to full exploitation of the parallelism of the GPU hardware. In the meantime, the genericness, as well as the simplicity of NN-Descent, are well-preserved. Moreover, a procedure that allows to k-NN graph to be merged efficiently on GPU is proposed. It makes the construction of high-quality k-NN graphs for out-of-GPU-memory datasets tractable. Our approach is 100-250\texttimes{} faster than the single-thread NN-Descent and is 2.5-5\texttimes{} faster than the existing GPU-based approaches as we tested on million as well as billion scale datasets.},
    booktitle = {Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
    pages = {1929–1938},
    numpages = {10},
    keywords = {high-dimensional, nn-descent, gpu, k-nearest neighbor graph},
    location = {Virtual Event, Queensland, Australia},
    series = {CIKM '21}
}
```

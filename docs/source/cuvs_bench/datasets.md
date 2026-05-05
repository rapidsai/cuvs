# cuVS Bench Datasets

A dataset usually has 4 binary files containing database vectors, query vectors, ground truth neighbors and their corresponding distances. For example, Glove-100 dataset has files <span class="title-ref">base.fbin</span> (database vectors), <span class="title-ref">query.fbin</span> (query vectors), <span class="title-ref">groundtruth.neighbors.ibin</span> (ground truth neighbors), and <span class="title-ref">groundtruth.distances.fbin</span> (ground truth distances). The first two files are for index building and searching, while the other two are associated with a particular distance and are used for evaluation.

The file suffixes <span class="title-ref">.fbin</span>, <span class="title-ref">.f16bin</span>, <span class="title-ref">.ibin</span>, <span class="title-ref">.u8bin</span>, and <span class="title-ref">.i8bin</span> denote that the data type of vectors stored in the file are <span class="title-ref">float32</span>, <span class="title-ref">float16</span>(a.k.a <span class="title-ref">half</span>), <span class="title-ref">int</span>, <span class="title-ref">uint8</span>, and <span class="title-ref">int8</span>, respectively. These binary files are little-endian and the format is: the first 8 bytes are <span class="title-ref">num_vectors</span> (<span class="title-ref">uint32_t</span>) and <span class="title-ref">num_dimensions</span> (<span class="title-ref">uint32_t</span>), and the following <span class="title-ref">num_vectors \* num_dimensions \* sizeof(type)</span> bytes are vectors stored in row-major order.

Some implementation can take <span class="title-ref">float16</span> database and query vectors as inputs and will have better performance. Use <span class="title-ref">python/cuvs_bench/cuvs_bench/get_dataset/fbin_to_f16bin.py</span> to transform dataset from <span class="title-ref">float32</span> to <span class="title-ref">float16</span> type.

Commonly used datasets can be downloaded from two websites: \#. Million-scale datasets can be found at the [Data sets](https://github.com/erikbern/ann-benchmarks#data-sets) section of [ann-benchmarks](https://github.com/erikbern/ann-benchmarks).

> However, these datasets are in HDF5 format. Use <span class="title-ref">python/cuvs_bench/cuvs_bench/get_dataset/hdf5_to_fbin.py</span> to transform the format. The usage of this script is:
>
> ``` bash
> $ python/cuvs_bench/cuvs_bench/get_dataset/hdf5_to_fbin.py
> usage: hdf5_to_fbin.py [-n] <input>.hdf5
>    -n: normalize base/query set
>  outputs: <input>.base.fbin
>           <input>.query.fbin
>           <input>.groundtruth.neighbors.ibin
>           <input>.groundtruth.distances.fbin
> ```
>
> So for an input <span class="title-ref">.hdf5</span> file, four output binary files will be produced. See previous section for an example of prepossessing GloVe dataset.
>
> Most datasets provided by <span class="title-ref">ann-benchmarks</span> use <span class="title-ref">Angular</span> or <span class="title-ref">Euclidean</span> distance. <span class="title-ref">Angular</span> denotes cosine distance. However, computing cosine distance reduces to computing inner product by normalizing vectors beforehand. In practice, we can always do the normalization to decrease computation cost, so it's better to measure the performance of inner product rather than cosine distance. The <span class="title-ref">-n</span> option of <span class="title-ref">hdf5_to_fbin.py</span> can be used to normalize the dataset.

1.  Billion-scale datasets can be found at [big-ann-benchmarks](http://big-ann-benchmarks.com). The ground truth file contains both neighbors and distances, thus should be split. A script is provided for this:

    > Take Deep-1B dataset as an example:
    >
    > ``` bash
    > mkdir -p data/deep-1B && cd data/deep-1B
    >
    > # download manually "Ground Truth" file of "Yandex DEEP"
    > # suppose the file name is deep_new_groundtruth.public.10K.bin
    > python -m cuvs_bench.split_groundtruth deep_new_groundtruth.public.10K.bin groundtruth
    >
    > # two files 'groundtruth.neighbors.ibin' and 'groundtruth.distances.fbin' should be produced
    > ```
    >
    > Besides ground truth files for the whole billion-scale datasets, this site also provides ground truth files for the first 10M or 100M vectors of the base sets. This mean we can use these billion-scale datasets as million-scale datasets. To facilitate this, an optional parameter <span class="title-ref">subset_size</span> for dataset can be used. See the next step for further explanation.

## Generate ground truth

If you have a dataset, but no corresponding ground truth file, then you can generate ground trunth using the <span class="title-ref">generate_groundtruth</span> utility. Example usage:

``` bash
# With existing query file
python -m cuvs_bench.generate_groundtruth --dataset /dataset/base.fbin --output=groundtruth_dir --queries=/dataset/query.public.10K.fbin

# With randomly generated queries
python -m cuvs_bench.generate_groundtruth --dataset /dataset/base.fbin --output=groundtruth_dir --queries=random --n_queries=10000

# Using only a subset of the dataset. Define queries by randomly
# selecting vectors from the (subset of the) dataset.
python -m cuvs_bench.generate_groundtruth --dataset /dataset/base.fbin --nrows=2000000 --output=groundtruth_dir --queries=random-choice --n_queries=10000
```

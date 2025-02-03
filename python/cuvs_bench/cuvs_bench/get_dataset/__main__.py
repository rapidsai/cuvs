#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import ast
import h5py
import numpy as np
import os
import subprocess
import sys
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist
import requests


def get_dataset_path(name, ann_bench_data_path):
    if not os.path.exists(ann_bench_data_path):
        os.mkdir(ann_bench_data_path)
    return os.path.join(ann_bench_data_path, f"{name}.hdf5")


def download_dataset(url, path):
    if not os.path.exists(path):
        print(f"downloading {url} -> {path}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)


def convert_hdf5_to_fbin(path, normalize):
    scripts_path = os.path.dirname(os.path.realpath(__file__))
    ann_bench_scripts_path = os.path.join(scripts_path, "hdf5_to_fbin.py")
    print(f"calling script {ann_bench_scripts_path}")
    if normalize and "angular" in path:
        subprocess.run(
            ["python", ann_bench_scripts_path, "-n", "%s" % path], check=True
        )
    else:
        subprocess.run(
            ["python", ann_bench_scripts_path, "%s" % path], check=True
        )


def move(name, ann_bench_data_path):
    if "angular" in name:
        new_name = name.replace("angular", "inner")
    else:
        new_name = name
    new_path = os.path.join(ann_bench_data_path, new_name)
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    for bin_name in [
        "base.fbin",
        "query.fbin",
        "groundtruth.neighbors.ibin",
        "groundtruth.distances.fbin",
    ]:
        os.rename(
            f"{ann_bench_data_path}/{name}.{bin_name}",
            f"{new_path}/{bin_name}",
        )


def download(name, normalize, ann_bench_data_path):
    path = get_dataset_path(name, ann_bench_data_path)
    try:
        url = f"http://ann-benchmarks.com/{name}.hdf5"
        download_dataset(url, path)

        convert_hdf5_to_fbin(path, normalize)

        move(name, ann_bench_data_path)
    except Exception:
        print(f"Cannot download {url}")
        raise


def generate_ann_benchmark_like_data(
    output_file='ann_benchmarks_like.hdf5',
    n_train=1000,
    n_test=100,
    d=32,
    centers=3,
    k=100,
    metric='euclidean',
    dataset_path='test-data/'
):
    """
    Generate a synthetic dataset in HDF5 format with a structure
    similar to ann-benchmarks datasets. By default, ground truth
    is computed for the top-100 nearest neighbors.
    """

    train_data, _ = make_blobs(
        n_samples=n_train,
        n_features=d,
        centers=centers,
        random_state=42
    )

    test_data, _ = make_blobs(
        n_samples=n_test,
        n_features=d,
        centers=centers,
        random_state=84
    )

    test_data = test_data.astype(np.float32)
    train_data = train_data.astype(np.float32)

    dist_matrix = cdist(test_data, train_data, metric=metric)

    actual_k = min(k, n_train)
    neighbors = np.argsort(dist_matrix, axis=1)[:, :actual_k].astype(np.int32)
    distances = np.take_along_axis(dist_matrix, neighbors, axis=1).astype(np.float32)

    full_path = os.path.join(dataset_path, "test-data")
    os.makedirs(full_path, exist_ok=True)
    full_path = os.path.join(full_path, output_file)

    with h5py.File(full_path, 'w') as f:
        # Datasets
        f.create_dataset('train', data=train_data)
        f.create_dataset('test', data=test_data)
        f.create_dataset('neighbors', data=neighbors)
        f.create_dataset('distances', data=distances)

        f.attrs['distance'] = metric

    convert_hdf5_to_fbin(full_path, normalize=True)

    print(f"Created {full_path} with:")
    print(f" - train shape = {train_data.shape}")
    print(f" - test shape = {test_data.shape}")
    print(f" - neighbors shape = {neighbors.shape}")
    print(f" - distances shape = {distances.shape}")
    print(f" - metric = {metric}")
    print(f" - neighbors per test sample = {actual_k}")



def main():
    call_path = os.getcwd()
    if "RAPIDS_DATASET_ROOT_DIR" in os.environ:
        default_dataset_path = os.getenv("RAPIDS_DATASET_ROOT_DIR")
    else:
        default_dataset_path = os.path.join(call_path, "datasets/")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset", help="dataset to download", default="glove-100-angular"
    )
    parser.add_argument(
        "--test-data-n-train", help="dataset to download", default=10000
    )
    parser.add_argument(
        "--test-data-n-test", help="dataset to download", default=1000
    )
    parser.add_argument(
        "--test-data-dims", help="dataset to download", default=32
    )
    parser.add_argument(
        "--test-data-k", help="dataset to download", default=100
    )
    parser.add_argument(
        "--test-data-output-file", help="dataset to download", default="ann_benchmarks_like.hdf5"
    )
    parser.add_argument(
        "--dataset-path",
        help="path to download dataset",
        default=default_dataset_path,
    )
    parser.add_argument(
        "--normalize",
        help="normalize cosine distance to inner product",
        action="store_true",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    if args.dataset == "test-data":
        generate_ann_benchmark_like_data(
            output_file=args.test_data_output_file,
            n_train=args.test_data_n_train,
            n_test=args.test_data_n_test,
            d=args.test_data_dims,
            centers=3,
            k=args.test_data_k,
            metric='euclidean',
            dataset_path=args.dataset_path
        )
    else:
        download(args.dataset, args.normalize, args.dataset_path)


if __name__ == "__main__":
    main()

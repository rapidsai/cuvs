#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import gzip
import os
import shutil
import struct
import subprocess
import tarfile

import click
import h5py
import numpy as np
import requests
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs


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


def download_with_wget(url, path):
    """Download file using wget (better for large FTP files)."""
    if not os.path.exists(path):
        print(f"downloading {url} -> {path}...")
        subprocess.run(["wget", "-O", path, url], check=True)


def read_bvecs(filename, n_vectors=None):
    """
    Read .bvecs file format from TEXMEX.
    Format: each vector is [dim (4 bytes int)] [dim bytes uint8 data]
    Returns float32 array.
    """
    print(f"Reading {filename}...")
    with open(filename, "rb") as f:
        # Read dimension from first vector
        dim = struct.unpack("i", f.read(4))[0]
        f.seek(0)

        # Calculate number of vectors
        f.seek(0, 2)  # Seek to end
        file_size = f.tell()
        record_size = 4 + dim
        total_vectors = file_size // record_size
        f.seek(0)

        if n_vectors is None:
            n_vectors = total_vectors
        else:
            n_vectors = min(n_vectors, total_vectors)

        print(f"  Reading {n_vectors:,} vectors of dimension {dim}")

        data = np.zeros((n_vectors, dim), dtype=np.float32)
        for i in range(n_vectors):
            f.read(4)  # Skip dimension
            vec = np.frombuffer(f.read(dim), dtype=np.uint8)
            data[i] = vec.astype(np.float32)

            if (i + 1) % 10_000_000 == 0:
                print(f"    Loaded {i + 1:,} / {n_vectors:,} vectors...")

    return data


def read_ivecs(filename, n_vectors=None):
    """
    Read .ivecs file format (groundtruth neighbors).
    Format: each vector is [dim (4 bytes int)] [dim int32 values]
    """
    print(f"Reading {filename}...")
    with open(filename, "rb") as f:
        dim = struct.unpack("i", f.read(4))[0]
        f.seek(0)

        f.seek(0, 2)
        file_size = f.tell()
        record_size = 4 + dim * 4
        total_vectors = file_size // record_size
        f.seek(0)

        if n_vectors is None:
            n_vectors = total_vectors
        else:
            n_vectors = min(n_vectors, total_vectors)

        print(f"  Reading {n_vectors:,} vectors of dimension {dim}")

        data = np.zeros((n_vectors, dim), dtype=np.int32)
        for i in range(n_vectors):
            d = struct.unpack("i", f.read(4))[0]
            data[i] = np.frombuffer(f.read(dim * 4), dtype=np.int32)

    return data


def read_fvecs(filename, n_vectors=None):
    """
    Read .fvecs file format.
    Format: each vector is [dim (4 bytes int)] [dim float32 values]
    """
    print(f"Reading {filename}...")
    with open(filename, "rb") as f:
        dim = struct.unpack("i", f.read(4))[0]
        f.seek(0)

        f.seek(0, 2)
        file_size = f.tell()
        record_size = 4 + dim * 4
        total_vectors = file_size // record_size
        f.seek(0)

        if n_vectors is None:
            n_vectors = total_vectors
        else:
            n_vectors = min(n_vectors, total_vectors)

        print(f"  Reading {n_vectors:,} vectors of dimension {dim}")

        data = np.zeros((n_vectors, dim), dtype=np.float32)
        for i in range(n_vectors):
            f.read(4)  # Skip dimension
            data[i] = np.frombuffer(f.read(dim * 4), dtype=np.float32)

            if (i + 1) % 10_000_000 == 0:
                print(f"    Loaded {i + 1:,} / {n_vectors:,} vectors...")

    return data


def write_fbin(filename, data):
    """Write data in .fbin format (used by cuVS benchmarks)."""
    print(f"Writing {filename}...")
    n, d = data.shape
    with open(filename, "wb") as f:
        f.write(struct.pack("i", n))
        f.write(struct.pack("i", d))
        data.astype(np.float32).tofile(f)
    print(f"  Wrote {n:,} x {d} float32 array")


def write_ibin(filename, data):
    """Write data in .ibin format (used by cuVS benchmarks for groundtruth)."""
    print(f"Writing {filename}...")
    n, d = data.shape
    with open(filename, "wb") as f:
        f.write(struct.pack("i", n))
        f.write(struct.pack("i", d))
        data.astype(np.int32).tofile(f)
    print(f"  Wrote {n:,} x {d} int32 array")


def download_sift1b(ann_bench_data_path, n_base_vectors=None):
    """
    Download and convert SIFT1B dataset from TEXMEX corpus.
    http://corpus-texmex.irisa.fr/

    The dataset contains:
    - bigann_base.bvecs: 1B base vectors (128-dim uint8)
    - bigann_query.bvecs: 10K query vectors
    - bigann_learn.bvecs: 100M learning vectors
    - bigann_gnd.tar.gz: Groundtruth for various subset sizes
    """
    base_url = "ftp://ftp.irisa.fr/local/texmex/corpus"

    # Create output directory
    if n_base_vectors is None:
        output_dir = os.path.join(ann_bench_data_path, "sift-1B")
    else:
        size_suffix = f"{n_base_vectors // 1_000_000}M"
        output_dir = os.path.join(ann_bench_data_path, f"sift-{size_suffix}")

    os.makedirs(output_dir, exist_ok=True)

    # Temporary directory for downloads
    tmp_dir = os.path.join(ann_bench_data_path, "tmp_sift1b")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        # Download and process base vectors
        base_gz = os.path.join(tmp_dir, "bigann_base.bvecs.gz")
        base_file = os.path.join(tmp_dir, "bigann_base.bvecs")

        if not os.path.exists(os.path.join(output_dir, "base.fbin")):
            if not os.path.exists(base_file):
                if not os.path.exists(base_gz):
                    download_with_wget(
                        f"{base_url}/bigann_base.bvecs.gz", base_gz
                    )
                print(f"Decompressing {base_gz}...")
                with gzip.open(base_gz, "rb") as f_in:
                    with open(base_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

            base_data = read_bvecs(base_file, n_base_vectors)
            write_fbin(os.path.join(output_dir, "base.fbin"), base_data)
            del base_data

        # Download and process query vectors
        query_gz = os.path.join(tmp_dir, "bigann_query.bvecs.gz")
        query_file = os.path.join(tmp_dir, "bigann_query.bvecs")

        if not os.path.exists(os.path.join(output_dir, "query.fbin")):
            if not os.path.exists(query_file):
                if not os.path.exists(query_gz):
                    download_with_wget(
                        f"{base_url}/bigann_query.bvecs.gz", query_gz
                    )
                print(f"Decompressing {query_gz}...")
                with gzip.open(query_gz, "rb") as f_in:
                    with open(query_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

            query_data = read_bvecs(query_file)
            write_fbin(os.path.join(output_dir, "query.fbin"), query_data)
            del query_data

        # Download and process groundtruth
        gnd_tar = os.path.join(tmp_dir, "bigann_gnd.tar.gz")

        if not os.path.exists(
            os.path.join(output_dir, "groundtruth.neighbors.ibin")
        ):
            if not os.path.exists(gnd_tar):
                download_with_wget(f"{base_url}/bigann_gnd.tar.gz", gnd_tar)

            print(f"Extracting {gnd_tar}...")
            with tarfile.open(gnd_tar, "r:gz") as tar:
                tar.extractall(tmp_dir)

            # Choose appropriate groundtruth file based on n_base_vectors
            if n_base_vectors is None:
                gnd_file = os.path.join(tmp_dir, "gnd", "idx_1000M.ivecs")
            else:
                # Find closest available groundtruth
                sizes = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
                target_m = n_base_vectors // 1_000_000
                closest = min(
                    sizes,
                    key=lambda x: abs(x - target_m)
                    if x >= target_m
                    else float("inf"),
                )
                if closest > target_m:
                    closest = max(
                        [s for s in sizes if s <= target_m], default=sizes[0]
                    )
                gnd_file = os.path.join(
                    tmp_dir, "gnd", f"idx_{closest}M.ivecs"
                )

            if os.path.exists(gnd_file):
                gnd_data = read_ivecs(gnd_file)
                write_ibin(
                    os.path.join(output_dir, "groundtruth.neighbors.ibin"),
                    gnd_data,
                )
            else:
                print(f"Warning: Groundtruth file {gnd_file} not found")
                # List available files
                gnd_dir = os.path.join(tmp_dir, "gnd")
                if os.path.exists(gnd_dir):
                    print(
                        f"Available groundtruth files: {os.listdir(gnd_dir)}"
                    )

        print(f"\nSIFT1B dataset prepared in: {output_dir}")
        print("Files:")
        for f in os.listdir(output_dir):
            fpath = os.path.join(output_dir, f)
            size_mb = os.path.getsize(fpath) / 1e6
            print(f"  {f}: {size_mb:.1f} MB")

    finally:
        # Optionally clean up temp files
        # shutil.rmtree(tmp_dir)
        print(f"\nTemp files kept in: {tmp_dir}")
        print("You can delete them manually after verifying the dataset.")


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
    output_file="ann_benchmarks_like.hdf5",
    n_train=1000,
    n_test=100,
    d=32,
    centers=3,
    k=100,
    metric="euclidean",
    dataset_path="test-data/",
):
    """
    Generate a synthetic dataset in HDF5 format with a structure
    similar to ann-benchmarks datasets. By default, ground truth
    is computed for the top-100 nearest neighbors.
    """

    train_data, _ = make_blobs(
        n_samples=n_train, n_features=d, centers=centers, random_state=42
    )

    test_data, _ = make_blobs(
        n_samples=n_test, n_features=d, centers=centers, random_state=84
    )

    test_data = test_data.astype(np.float32)
    train_data = train_data.astype(np.float32)

    dist_matrix = cdist(test_data, train_data, metric=metric)

    actual_k = min(k, n_train)
    neighbors = np.argsort(dist_matrix, axis=1)[:, :actual_k].astype(np.int32)
    distances = np.take_along_axis(dist_matrix, neighbors, axis=1).astype(
        np.float32
    )

    full_path = os.path.join(dataset_path, "test-data")
    os.makedirs(full_path, exist_ok=True)
    full_path = os.path.join(full_path, output_file)

    with h5py.File(full_path, "w") as f:
        # Datasets
        f.create_dataset("train", data=train_data)
        f.create_dataset("test", data=test_data)
        f.create_dataset("neighbors", data=neighbors)
        f.create_dataset("distances", data=distances)

        f.attrs["distance"] = metric

    convert_hdf5_to_fbin(full_path, normalize=True)

    print(f"Created {full_path} with:")
    print(f" - train shape = {train_data.shape}")
    print(f" - test shape = {test_data.shape}")
    print(f" - neighbors shape = {neighbors.shape}")
    print(f" - distances shape = {distances.shape}")
    print(f" - metric = {metric}")
    print(f" - neighbors per test sample = {actual_k}")


def get_default_dataset_path():
    return os.getenv(
        "RAPIDS_DATASET_ROOT_DIR", os.path.join(os.getcwd(), "datasets")
    )


@click.command()
@click.option(
    "--dataset",
    default="glove-100-angular",
    help="Dataset to download. Use 'sift-1B' for TEXMEX SIFT1B dataset, "
    "or 'sift-100M', 'sift-10M' etc for subsets.",
)
@click.option(
    "--test-data-n-train",
    default=10000,
    help="Number of training examples for the test data.",
)
@click.option(
    "--test-data-n-test",
    default=1000,
    help="Number of test examples for the test data.",
)
@click.option(
    "--test-data-dims",
    default=32,
    help="Dimensionality for the test data.",
)
@click.option(
    "--test-data-k",
    default=100,
    help="K value for the test data.",
)
@click.option(
    "--test-data-output-file",
    default="ann_benchmarks_like.hdf5",
    help="Output file name for the test data.",
)
@click.option(
    "--dataset-path",
    default=None,
    help="Path to download the dataset. If not provided, defaults to "
    "the value of RAPIDS_DATASET_ROOT_DIR or '<cwd>/datasets'.",
)
@click.option(
    "--normalize",
    is_flag=True,
    help="Normalize cosine distance to inner product.",
)
@click.option(
    "--n-vectors",
    default=None,
    type=int,
    help="Number of base vectors to use (for sift-1B dataset). "
    "E.g., 300000000 for 300M vectors.",
)
def main(
    dataset,
    test_data_n_train,
    test_data_n_test,
    test_data_dims,
    test_data_k,
    test_data_output_file,
    dataset_path,
    normalize,
    n_vectors,
):
    # Compute default dataset_path if not provided.
    if dataset_path is None:
        dataset_path = get_default_dataset_path()

    if dataset == "test-data":
        generate_ann_benchmark_like_data(
            output_file=test_data_output_file,
            n_train=test_data_n_train,
            n_test=test_data_n_test,
            d=test_data_dims,
            centers=3,
            k=test_data_k,
            metric="euclidean",
            dataset_path=dataset_path,
        )
    elif dataset.startswith("sift-"):
        # Handle SIFT1B and subsets from TEXMEX
        # Parse dataset name for size hint (e.g., sift-100M, sift-1B)
        size_str = dataset.split("-")[1].upper()
        if n_vectors is None:
            if size_str == "1B":
                n_vectors = None  # Use all 1B vectors
            elif size_str.endswith("M"):
                n_vectors = int(size_str[:-1]) * 1_000_000
            elif size_str.endswith("K"):
                n_vectors = int(size_str[:-1]) * 1_000
            else:
                try:
                    n_vectors = int(size_str)
                except ValueError:
                    n_vectors = None

        download_sift1b(dataset_path, n_vectors)
    else:
        download(dataset, normalize, dataset_path)


if __name__ == "__main__":
    main()

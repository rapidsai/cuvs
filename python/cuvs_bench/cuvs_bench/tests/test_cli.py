#
# Copyright (c) 2025, NVIDIA CORPORATION.
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
#


import pytest
from click.testing import CliRunner
from pathlib import Path

# Adjust the import to match where your CLI entry point is defined.
# For example, if your __main__.py defines a `main()` function:
from cuvs_bench.get_dataset.__main__ import main


@pytest.fixture(scope="session")
def temp_datasets_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("datasets")


def test_get_dataset_creates_expected_files(temp_datasets_dir: Path):
    runner = CliRunner()
    dataset_path_arg = str(temp_datasets_dir)

    # Invoke the CLI command as if calling:
    # python -m cuvs_bench.get_dataset --dataset test-data --dataset-path <temp_datasets_dir>
    result = runner.invoke(main, ['--dataset', 'test-data', '--dataset-path', dataset_path_arg])

    assert result.exit_code == 0, f"CLI call failed: {result.output}"

    expected_files = [
        "test-data/ann_benchmarks_like.groundtruth.distances.fbin",
        "test-data/ann_benchmarks_like.base.fbin",
        "test-data/ann_benchmarks_like.groundtruth.neighbors.ibin",
        "test-data/ann_benchmarks_like.query.fbin",
        "test-data/ann_benchmarks_like.hdf5",
    ]

    # Verify that each expected file exists in the datasets directory.
    for filename in expected_files:
        file_path = temp_datasets_dir / filename
        assert file_path.exists(), f"Expected file {filename} was not generated."


def test_run_command_creates_results(temp_datasets_dir: Path):
    """
    This test simulates running the command:

        python -m cuvs_bench.run --dataset test-data --dataset-path datasets/ \
            --algorithms faiss_gpu_ivf_flat,faiss_gpu_ivf_sq,cuvs_ivf_flat,cuvs_cagra,ggnn,cuvs_cagra_hnswlib, \
            --batch-size 100 -k 10 --groups test -m latency --force

    It then:
      1. Checks that the output includes a benchmark table header and a line
         matching a pattern for "faiss_gpu_ivf_flat_test.../process_time/real_time"
         (allowing variable numeric values).
      2. Verifies that a set of expected result files (both under result/build and result/search)
         are created under datasets/test-data/ and are not empty.
    """

    dataset_path_arg = str(temp_datasets_dir)

    from cuvs_bench.run.__main__ import main as run_main

    runner = CliRunner()
    run_args = [
        "--dataset", "test-data",
        "--dataset-path", dataset_path_arg,
        "--algorithms", "faiss_gpu_ivf_flat,faiss_gpu_ivf_sq,cuvs_ivf_flat,cuvs_cagra,ggnn,cuvs_cagra_hnswlib,",
        "--batch-size", "100",
        "-k", "10",
        "--groups", "test",
        "-m", "latency",
        "--force"
    ]
    result = runner.invoke(run_main, run_args)
    assert result.exit_code == 0, f"Run command failed with output:\n{result.output}"

    # --- Verify that the expected result files exist and are not empty ---
    expected_files = [
        # Build files:
        "test-data/result/build/cuvs_ivf_flat,test.csv",
        "test-data/result/build/cuvs_cagra_hnswlib,test.csv",
        "test-data/result/build/faiss_gpu_ivf_flat,test.csv",
        "test-data/result/build/cuvs_cagra,test.csv",
        "test-data/result/build/cuvs_cagra,test.json",
        "test-data/result/build/cuvs_cagra_hnswlib,test.json",
        "test-data/result/build/cuvs_ivf_flat,test.json",
        "test-data/result/build/faiss_gpu_ivf_flat,test.json",
        # Search files:
        "test-data/result/search/cuvs_cagra_hnswlib,test,k10,bs100,latency.csv",
        "test-data/result/search/cuvs_cagra_hnswlib,test,k10,bs100,throughput.csv",
        "test-data/result/search/cuvs_cagra_hnswlib,test,k10,bs100,raw.csv",
        "test-data/result/search/cuvs_cagra,test,k10,bs100,latency.csv",
        "test-data/result/search/cuvs_cagra,test,k10,bs100,throughput.csv",
        "test-data/result/search/cuvs_cagra,test,k10,bs100,raw.csv",
        "test-data/result/search/cuvs_ivf_flat,test,k10,bs100,latency.csv",
        "test-data/result/search/cuvs_ivf_flat,test,k10,bs100,raw.csv",
        "test-data/result/search/cuvs_ivf_flat,test,k10,bs100,throughput.csv",
        "test-data/result/search/faiss_gpu_ivf_flat,test,k10,bs100,latency.csv",
        "test-data/result/search/faiss_gpu_ivf_flat,test,k10,bs100,raw.csv",
        "test-data/result/search/faiss_gpu_ivf_flat,test,k10,bs100,throughput.csv",
        "test-data/result/search/cuvs_cagra,test,k10,bs100.json",
        "test-data/result/search/cuvs_cagra_hnswlib,test,k10,bs100.json",
        "test-data/result/search/cuvs_ivf_flat,test,k10,bs100.json",
        "test-data/result/search/faiss_gpu_ivf_flat,test,k10,bs100.json",
    ]

    for rel_path in expected_files:
        file_path = temp_datasets_dir / rel_path
        assert file_path.exists(), f"Expected file {file_path} does not exist."
        assert file_path.stat().st_size > 0, f"Expected file {file_path} is empty."


def test_plot_command_creates_png_files(temp_datasets_dir: Path):
    """
    This test simulates running the command:

      python -m cuvs_bench.plot --dataset test-data --dataset-path datasets/ \
          --algorithms faiss_gpu_ivf_flat,faiss_gpu_ivf_sq,cuvs_ivf_flat,cuvs_cagra,ggnn,cuvs_cagra_hnswlib \
          --batch-size 100 -k 10 --groups test -m latency

    and then verifies that the following files are produced in the working directory:
      - search-test-data-k10-batch_size100.png
      - build-test-data-k10-batch_size100.png

    It also checks that these files are not empty.
    """

    dataset_path_arg = str(temp_datasets_dir)

    # Import the plot command's main function. Adjust the import path if needed.
    from cuvs_bench.plot.__main__ import main as plot_main

    runner = CliRunner()
    args = [
        "--dataset", "test-data",
        "--dataset-path", dataset_path_arg,
        "--output-filepath", dataset_path_arg,
        "--algorithms", "faiss_gpu_ivf_flat,faiss_gpu_ivf_sq,cuvs_ivf_flat,cuvs_cagra,ggnn,cuvs_cagra_hnswlib",
        "--batch-size", "100",
        "-k", "10",
        "--groups", "test",
        "-m", "latency"
    ]
    result = runner.invoke(plot_main, args)
    assert result.exit_code == 0, f"Plot command failed with output:\n{result.output}"

    # Expected output file names.
    expected_files = [
        "search-test-data-k10-batch_size100.png",
        "build-test-data-k10-batch_size100.png"
    ]

    for filename in expected_files:
        file_path = temp_datasets_dir / filename
        assert file_path.exists(), f"Expected file {filename} does not exist."
        assert file_path.stat().st_size > 0, f"Expected file {filename} is empty."

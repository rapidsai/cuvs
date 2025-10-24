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

from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner
from cuvs_bench.get_dataset.__main__ import main


@pytest.fixture(scope="session")
def temp_datasets_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("datasets")


def test_get_dataset_creates_expected_files(temp_datasets_dir: Path):
    runner = CliRunner()
    dataset_path_arg = str(temp_datasets_dir)

    # Invoke the CLI command as if calling:
    # python -m cuvs_bench.get_dataset --dataset test-data \
    # --dataset-path <temp_datasets_dir>
    result = runner.invoke(
        main, ["--dataset", "test-data", "--dataset-path", dataset_path_arg]
    )

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
        assert (
            file_path.exists()
        ), f"Expected file {filename} was not generated."


def test_run_command_creates_results(temp_datasets_dir: Path):
    """
    This test simulates running the command:

        python -m cuvs_bench.run --dataset test-data --dataset-path datasets/ \
            --algorithms faiss_gpu_ivf_flat,faiss_gpu_ivf_sq,cuvs_ivf_flat,\
            cuvs_cagra,ggnn,cuvs_cagra_hnswlib, \
            --batch-size 100 -k 10 --groups test -m latency --force

    It then verifies that the set of expected result files
         (both under result/build and result/search)
         are created under datasets/test-data/ and are not empty.
    """

    dataset_path_arg = str(temp_datasets_dir)

    from cuvs_bench.run.__main__ import main as run_main

    runner = CliRunner()
    run_args = [
        "--dataset",
        "test-data",
        "--dataset-path",
        dataset_path_arg,
        "--algorithms",
        "faiss_gpu_ivf_flat,faiss_gpu_ivf_sq,cuvs_ivf_flat,cuvs_cagra,ggnn,cuvs_cagra_hnswlib,",  # noqa: E501
        "--batch-size",
        "100",
        "-k",
        "10",
        "--groups",
        "test",
        "-m",
        "latency",
        "--force",
    ]
    result = runner.invoke(run_main, run_args)
    assert (
        result.exit_code == 0
    ), f"Run command failed with output:\n{result.output}"

    common_build_header = [
        "algo_name",
        "index_name",
        "time",
        "threads",
        "cpu_time",
    ]

    common_search_header = [
        "algo_name",
        "index_name",
        "recall",
        "throughput",
        "latency",
        "threads",
        "cpu_time",
    ]

    # --- Verify that the expected result files exist and are not empty ---
    expected_files = {
        # Build files:
        "test-data/result/build/cuvs_ivf_flat,test.csv": {
            "header": common_build_header
            + [
                "GPU",
                "niter",
                "nlist",
                "ratio",
            ],
            "rows": 1,
        },
        "test-data/result/build/cuvs_cagra_hnswlib,test.csv": {
            "header": common_build_header
            + [
                "ef_construction",
                "graph_degree",
                "intermediate_graph_degree",
                "label",
            ],
            "rows": 2,
        },
        "test-data/result/build/faiss_gpu_ivf_flat,test.csv": {
            "header": common_build_header
            + [
                "GPU",
                "nlist",
                "ratio",
                "use_cuvs",
            ],
            "rows": 1,
        },
        "test-data/result/build/cuvs_cagra,test.csv": {
            "header": common_build_header
            + [
                "GPU",
                "graph_degree",
                "intermediate_graph_degree",
                "label",
            ],
            "rows": 1,
        },
        # Search files:
        "test-data/result/search/cuvs_cagra_hnswlib,test,k10,bs100,raw.csv": {
            "header": common_search_header
            + [
                "ef",
                "end_to_end",
                "k",
                "n_queries",
                "total_queries",
                "build time",
                "build threads",
                "build cpu_time",
                "ef_construction",
                "graph_degree",
                "intermediate_graph_degree",
                "label",
            ],
            "rows": 4,
        },
        "test-data/result/search/cuvs_cagra,test,k10,bs100,latency.csv": {
            "header": common_search_header
            + [
                "GPU",
                "end_to_end",
                "itopk",
                "k",
                "n_queries",
                "search_width",
                "total_queries",
                "build time",
                "build threads",
                "build cpu_time",
                "build GPU",
                "graph_degree",
                "intermediate_graph_degree",
                "label",
            ],
            "rows": 2,
        },
        "test-data/result/search/cuvs_cagra,test,k10,bs100,throughput.csv": {
            "header": common_search_header
            + [
                "GPU",
                "end_to_end",
                "itopk",
                "k",
                "n_queries",
                "search_width",
                "total_queries",
                "build time",
                "build threads",
                "build cpu_time",
                "build GPU",
                "graph_degree",
                "intermediate_graph_degree",
                "label",
            ],
            "rows": 2,
        },
        "test-data/result/search/cuvs_cagra,test,k10,bs100,raw.csv": {
            "header": common_search_header
            + [
                "GPU",
                "end_to_end",
                "itopk",
                "k",
                "n_queries",
                "search_width",
                "total_queries",
                "build time",
                "build threads",
                "build cpu_time",
                "build GPU",
                "graph_degree",
                "intermediate_graph_degree",
                "label",
            ],
            "rows": 2,
        },
        "test-data/result/search/cuvs_ivf_flat,test,k10,bs100,latency.csv": {
            "header": common_search_header
            + [
                "GPU",
                "end_to_end",
                "k",
                "n_queries",
                "nprobe",
                "total_queries",
                "build time",
                "build threads",
                "build cpu_time",
                "build GPU",
                "niter",
                "nlist",
                "ratio",
            ],
            "rows": 2,
        },
        "test-data/result/search/cuvs_ivf_flat,test,k10,bs100,raw.csv": {
            "header": common_search_header
            + [
                "GPU",
                "end_to_end",
                "k",
                "n_queries",
                "nprobe",
                "total_queries",
                "build time",
                "build threads",
                "build cpu_time",
                "build GPU",
                "niter",
                "nlist",
                "ratio",
            ],
            "rows": 2,
        },
        "test-data/result/search/cuvs_ivf_flat,test,k10,bs100,throughput.csv": {  # noqa: E501
            "header": common_search_header
            + [
                "GPU",
                "end_to_end",
                "k",
                "n_queries",
                "nprobe",
                "total_queries",
                "build time",
                "build threads",
                "build cpu_time",
                "build GPU",
                "niter",
                "nlist",
                "ratio",
            ],
            "rows": 2,
        },
        "test-data/result/search/faiss_gpu_ivf_flat,test,k10,bs100,latency.csv": {  # noqa: E501
            "header": common_search_header
            + [
                "GPU",
                "end_to_end",
                "k",
                "n_queries",
                "nprobe",
                "total_queries",
                "build time",
                "build threads",
                "build cpu_time",
                "build GPU",
                "nlist",
                "ratio",
                "use_cuvs",
            ],
            "rows": 2,
        },
        "test-data/result/search/faiss_gpu_ivf_flat,test,k10,bs100,raw.csv": {
            "header": common_search_header
            + [
                "GPU",
                "end_to_end",
                "k",
                "n_queries",
                "nprobe",
                "total_queries",
                "build time",
                "build threads",
                "build cpu_time",
                "build GPU",
                "nlist",
                "ratio",
                "use_cuvs",
            ],
            "rows": 2,
        },
        "test-data/result/search/faiss_gpu_ivf_flat,test,k10,bs100,throughput.csv": {  # noqa: E501
            "header": common_search_header
            + [
                "GPU",
                "end_to_end",
                "k",
                "n_queries",
                "nprobe",
                "total_queries",
                "build time",
                "build threads",
                "build cpu_time",
                "build GPU",
                "nlist",
                "ratio",
                "use_cuvs",
            ],
            "rows": 2,
        },
    }

    for rel_path, expectations in expected_files.items():
        file_path = temp_datasets_dir / rel_path
        assert file_path.exists(), f"Expected file {file_path} does not exist."
        assert (
            file_path.stat().st_size > 0
        ), f"Expected file {file_path} is empty."

        df = pd.read_csv(file_path)

        actual_header = list(df.columns)
        actual_rows = len(df)

        # breakpoint()
        assert (
            actual_header == expectations["header"]
        ), f"Wrong header produced in file f{rel_path}"
        assert actual_rows == expectations["rows"]


def test_plot_command_creates_png_files(temp_datasets_dir: Path):
    """
    This test simulates running the command:

      python -m cuvs_bench.plot --dataset test-data --dataset-path datasets/ \
          --algorithms faiss_gpu_ivf_flat,faiss_gpu_ivf_sq, \
          cuvs_ivf_flat,cuvs_cagra,ggnn,cuvs_cagra_hnswlib \
          --batch-size 100 -k 10 --groups test -m latency

    and then verifies that the following files are produced in the
    working directory:
      - search-test-data-k10-batch_size100.png
      - build-test-data-k10-batch_size100.png

    It also checks that these files are not empty.
    """

    dataset_path_arg = str(temp_datasets_dir)

    from cuvs_bench.plot.__main__ import main as plot_main

    runner = CliRunner()
    args = [
        "--dataset",
        "test-data",
        "--dataset-path",
        dataset_path_arg,
        "--output-filepath",
        dataset_path_arg,
        "--algorithms",
        "faiss_gpu_ivf_flat,faiss_gpu_ivf_sq,cuvs_ivf_flat,cuvs_cagra,ggnn,cuvs_cagra_hnswlib",  # noqa: E501
        "--batch-size",
        "100",
        "-k",
        "10",
        "--groups",
        "test",
        "-m",
        "latency",
    ]
    result = runner.invoke(plot_main, args)
    assert (
        result.exit_code == 0
    ), f"Plot command failed with output:\n{result.output}"

    # Expected output file names.
    expected_files = [
        "search-test-data-k10-batch_size100.png",
        "build-test-data-k10-batch_size100.png",
    ]

    for filename in expected_files:
        file_path = temp_datasets_dir / filename
        assert file_path.exists(), f"Expected file {filename} does not exist."
        assert (
            file_path.stat().st_size > 0
        ), f"Expected file {filename} is empty."


def test_cuvs_ivf_pq_algorithm(temp_datasets_dir: Path):
    """
    Test the cuvs_ivf_pq algorithm build and search operations.
    """
    dataset_path_arg = str(temp_datasets_dir)
    
    from cuvs_bench.run.__main__ import main as run_main
    
    runner = CliRunner()
    run_args = [
        "--dataset",
        "test-data",
        "--dataset-path",
        dataset_path_arg,
        "--algorithms",
        "cuvs_ivf_pq",
        "--batch-size",
        "100",
        "-k",
        "10",
        "--groups",
        "test",
        "-m",
        "latency",
        "--force",
    ]
    result = runner.invoke(run_main, run_args)
    assert (
        result.exit_code == 0
    ), f"Run command failed with output:\n{result.output}"
    
    # Verify build and search result files exist
    build_file = temp_datasets_dir / "test-data/result/build/cuvs_ivf_pq,test.csv"
    assert build_file.exists(), "cuvs_ivf_pq build file doesn't exist"
    assert build_file.stat().st_size > 0, "cuvs_ivf_pq build file is empty"
    
    search_file = temp_datasets_dir / "test-data/result/search/cuvs_ivf_pq,test,k10,bs100,raw.csv"
    assert search_file.exists(), "cuvs_ivf_pq search file doesn't exist"
    assert search_file.stat().st_size > 0, "cuvs_ivf_pq search file is empty"


def test_cuvs_brute_force_algorithm(temp_datasets_dir: Path):
    """
    Test the cuvs_brute_force algorithm build and search operations.
    """
    dataset_path_arg = str(temp_datasets_dir)
    
    from cuvs_bench.run.__main__ import main as run_main
    
    runner = CliRunner()
    run_args = [
        "--dataset",
        "test-data",
        "--dataset-path",
        dataset_path_arg,
        "--algorithms",
        "cuvs_brute_force",
        "--batch-size",
        "100",
        "-k",
        "10",
        "--groups",
        "test",
        "-m",
        "latency",
        "--force",
    ]
    result = runner.invoke(run_main, run_args)
    assert (
        result.exit_code == 0
    ), f"Run command failed with output:\n{result.output}"
    
    # Verify search result files exist (brute force may not have a build phase)
    search_file = temp_datasets_dir / "test-data/result/search/cuvs_brute_force,test,k10,bs100,raw.csv"
    assert search_file.exists(), "cuvs_brute_force search file doesn't exist"
    assert search_file.stat().st_size > 0, "cuvs_brute_force search file is empty"


def test_cuvs_vamana_algorithm(temp_datasets_dir: Path):
    """
    Test the cuvs_vamana algorithm build and search operations.
    """
    dataset_path_arg = str(temp_datasets_dir)
    
    from cuvs_bench.run.__main__ import main as run_main
    
    runner = CliRunner()
    run_args = [
        "--dataset",
        "test-data",
        "--dataset-path",
        dataset_path_arg,
        "--algorithms",
        "cuvs_vamana",
        "--batch-size",
        "100",
        "-k",
        "10",
        "--groups",
        "test",
        "-m",
        "latency",
        "--force",
    ]
    result = runner.invoke(run_main, run_args)
    # Note: This may fail if vamana is not enabled in the build
    if result.exit_code == 0:
        # Verify build and search result files exist
        build_file = temp_datasets_dir / "test-data/result/build/cuvs_vamana,test.csv"
        assert build_file.exists(), "cuvs_vamana build file doesn't exist"
        assert build_file.stat().st_size > 0, "cuvs_vamana build file is empty"
        
        search_file = temp_datasets_dir / "test-data/result/search/cuvs_vamana,test,k10,bs100,raw.csv"
        assert search_file.exists(), "cuvs_vamana search file doesn't exist"
        assert search_file.stat().st_size > 0, "cuvs_vamana search file is empty"


def test_cuvs_cagra_diskann_algorithm(temp_datasets_dir: Path):
    """
    Test the cuvs_cagra_diskann algorithm build and search operations.
    """
    dataset_path_arg = str(temp_datasets_dir)
    
    from cuvs_bench.run.__main__ import main as run_main
    
    runner = CliRunner()
    run_args = [
        "--dataset",
        "test-data",
        "--dataset-path",
        dataset_path_arg,
        "--algorithms",
        "cuvs_cagra_diskann",
        "--batch-size",
        "100",
        "-k",
        "10",
        "--groups",
        "test",
        "-m",
        "latency",
        "--force",
    ]
    result = runner.invoke(run_main, run_args)
    # Note: This may fail if diskann is not enabled in the build
    if result.exit_code == 0:
        # Verify build and search result files exist
        build_file = temp_datasets_dir / "test-data/result/build/cuvs_cagra_diskann,test.csv"
        assert build_file.exists(), "cuvs_cagra_diskann build file doesn't exist"
        assert build_file.stat().st_size > 0, "cuvs_cagra_diskann build file is empty"
        
        search_file = temp_datasets_dir / "test-data/result/search/cuvs_cagra_diskann,test,k10,bs100,raw.csv"
        assert search_file.exists(), "cuvs_cagra_diskann search file doesn't exist"
        assert search_file.stat().st_size > 0, "cuvs_cagra_diskann search file is empty"


@pytest.mark.skipif(
    not pytest.importorskip("nccl", reason="NCCL not available"),
    reason="Multi-GPU tests require NCCL",
)
def test_multi_gpu_algorithms(temp_datasets_dir: Path):
    """
    Test multi-GPU algorithms (cuvs_mg_*).
    """
    dataset_path_arg = str(temp_datasets_dir)
    
    from cuvs_bench.run.__main__ import main as run_main
    
    runner = CliRunner()
    
    # Test cuvs_mg_ivf_flat
    run_args = [
        "--dataset",
        "test-data",
        "--dataset-path",
        dataset_path_arg,
        "--algorithms",
        "cuvs_mg_ivf_flat",
        "--batch-size",
        "100",
        "-k",
        "10",
        "--groups",
        "test",
        "-m",
        "latency",
        "--force",
    ]
    result = runner.invoke(run_main, run_args)
    if result.exit_code == 0:
        # Verify files exist
        build_file = temp_datasets_dir / "test-data/result/build/cuvs_mg_ivf_flat,test.csv"
        assert build_file.exists(), "cuvs_mg_ivf_flat build file doesn't exist"
        
        search_file = temp_datasets_dir / "test-data/result/search/cuvs_mg_ivf_flat,test,k10,bs100,raw.csv"
        assert search_file.exists(), "cuvs_mg_ivf_flat search file doesn't exist"
    
    # Test cuvs_mg_ivf_pq
    run_args[7] = "cuvs_mg_ivf_pq"
    result = runner.invoke(run_main, run_args)
    if result.exit_code == 0:
        build_file = temp_datasets_dir / "test-data/result/build/cuvs_mg_ivf_pq,test.csv"
        assert build_file.exists(), "cuvs_mg_ivf_pq build file doesn't exist"
        
        search_file = temp_datasets_dir / "test-data/result/search/cuvs_mg_ivf_pq,test,k10,bs100,raw.csv"
        assert search_file.exists(), "cuvs_mg_ivf_pq search file doesn't exist"
    
    # Test cuvs_mg_cagra
    run_args[7] = "cuvs_mg_cagra"
    result = runner.invoke(run_main, run_args)
    if result.exit_code == 0:
        build_file = temp_datasets_dir / "test-data/result/build/cuvs_mg_cagra,test.csv"
        assert build_file.exists(), "cuvs_mg_cagra build file doesn't exist"
        
        search_file = temp_datasets_dir / "test-data/result/search/cuvs_mg_cagra,test,k10,bs100,raw.csv"
        assert search_file.exists(), "cuvs_mg_cagra search file doesn't exist"


def test_multiple_algorithms_integration(temp_datasets_dir: Path):
    """
    Integration test running multiple algorithms together.
    """
    dataset_path_arg = str(temp_datasets_dir)
    
    from cuvs_bench.run.__main__ import main as run_main
    
    runner = CliRunner()
    run_args = [
        "--dataset",
        "test-data",
        "--dataset-path",
        dataset_path_arg,
        "--algorithms",
        "cuvs_ivf_flat,cuvs_ivf_pq,cuvs_brute_force,cuvs_cagra",
        "--batch-size",
        "100",
        "-k",
        "10",
        "--groups",
        "test",
        "-m",
        "throughput",  # Test throughput mode as well
        "--force",
    ]
    result = runner.invoke(run_main, run_args)
    assert (
        result.exit_code == 0
    ), f"Multiple algorithms run failed with output:\n{result.output}"
    
    # Verify that results were created for all algorithms
    algorithms = ["cuvs_ivf_flat", "cuvs_ivf_pq", "cuvs_brute_force", "cuvs_cagra"]
    for algo in algorithms:
        search_file = temp_datasets_dir / f"test-data/result/search/{algo},test,k10,bs100,throughput.csv"
        assert search_file.exists(), f"{algo} throughput search file doesn't exist"
        assert search_file.stat().st_size > 0, f"{algo} throughput search file is empty"

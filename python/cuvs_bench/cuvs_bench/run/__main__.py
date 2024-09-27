#
# Copyright (c) 2024, NVIDIA CORPORATION.
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

import os
from pathlib import Path
from typing import Optional

import click
from run import run_benchmark


@click.command()
@click.option(
    "--subset-size",
    type=click.IntRange(min=1),
    help="The number of subset rows of the dataset to build the index",
)
@click.option(
    "-k",
    "--count",
    default=10,
    show_default=True,
    type=click.IntRange(min=1),
    prompt="Enter the number of  neighbors to search for",
    help="The number of nearest neighbors to search for",
)
@click.option(
    "-bs",
    "--batch-size",
    default=10000,
    show_default=True,
    type=click.IntRange(min=1),
    prompt="Enter the batch size",
    help="Number of query vectors to use in each query trial",
)
@click.option(
    "--dataset-configuration",
    default=None,
    show_default=True,
    help="Path to YAML configuration file for datasets",
)
@click.option(
    "--configuration",
    help="Path to YAML configuration file or directory for algorithms. "
    "Any run groups found in the specified file/directory will "
    "automatically override groups of the same name present in the "
    "default configurations, including `base`.",
)
@click.option(
    "--dataset",
    default="glove-100-inner",
    show_default=True,
    prompt="Enter the name of dataset",
    help="Name of dataset",
)
@click.option(
    "--dataset-path",
    default=lambda: os.environ.get(
        "RAPIDS_DATASET_ROOT_DIR",
        os.path.join(Path(__file__).parent, "datasets/"),
    ),
    show_default=True,
    prompt="Enter the path to dataset folder",
    help="Path to dataset folder, by default will look in "
    "RAPIDS_DATASET_ROOT_DIR if defined, otherwise a datasets "
    "subdirectory from the calling directory.",
)
@click.option("--build", is_flag=True, help="Build the index")
@click.option("--search", is_flag=True, help="Perform the search")
@click.option(
    "--algorithms",
    default="cuvs_cagra",
    show_default=True,
    prompt="Enter the comma separated list of named algorithms to run",
    help="Run only comma separated list of named algorithms. If parameters "
    "`groups` and `algo-groups` are both undefined, then group `base` "
    "is run by default.",
)
@click.option(
    "--groups",
    default="base",
    show_default=True,
    prompt="Enter the comma separated groups of parameters",
    help="Run only comma separated groups of parameters",
)
@click.option(
    "--algo-groups",
    help="Add comma separated <algorithm>.<group> to run. Example usage: "
    ' "--algo-groups=cuvs_cagra.large,hnswlib.large".',
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Re-run algorithms even if their results already exist",
)
@click.option(
    "-m",
    "--search-mode",
    default="latency",
    show_default=True,
    prompt='Enter the search mode ("latency" or "throughput")',
    help="Run search in 'latency' (measure individual batches) or "
    "'throughput' (pipeline batches and measure end-to-end) mode.",
)
@click.option(
    "-t",
    "--search-threads",
    default=None,
    show_default=True,
    help="Specify the number threads to use for throughput benchmark. "
    "Single value or a pair of min and max separated by ':'. "
    "Example: --search-threads=1:4. Power of 2 values between 'min' "
    "and 'max' will be used. If only 'min' is specified, then a single "
    "test is run with 'min' threads. By default min=1, "
    "max=<num hyper threads>.",
)
@click.option(
    "-r",
    "--dry-run",
    is_flag=True,
    help="Dry-run mode will convert the yaml config for the specified "
    "algorithms and datasets to the json format that’s consumed "
    "by the lower-level c++ binaries and then print the command to "
    "run execute the benchmarks but will not actually execute "
    "the command.",
)
@click.option(
    "--raft-log-level",
    default="info",
    show_default=True,
    prompt="Enter the log level",
    help="Log level, possible values are [off, error, warn, info, debug, "
    "trace]. Default: 'info'. Note that 'debug' or more detailed "
    "logging level requires that the library is compiled with "
    "-DRAFT_ACTIVE_LEVEL=<L> where <L> >= <requested log level>.",
)
def main(
    subset_size: Optional[int],
    count: int,
    batch_size: int,
    dataset_configuration: Optional[str],
    configuration: Optional[str],
    dataset: str,
    dataset_path: str,
    build: bool,
    search: bool,
    algorithms: Optional[str],
    groups: str,
    algo_groups: Optional[str],
    force: bool,
    search_mode: str,
    search_threads: Optional[str],
    dry_run: bool,
    raft_log_level: str,
) -> None:
    """
    Main function to run the benchmark with the provided options.

    Parameters
    ----------
    subset_size : Optional[int]
        The number of subset rows of the dataset to build the index.
    count : int
        The number of nearest neighbors to search for.
    batch_size : int
        Number of query vectors to use in each query trial.
    dataset_configuration : Optional[str]
        Path to YAML configuration file for datasets.
    configuration : Optional[str]
        Path to YAML configuration file or directory for algorithms.
    dataset : str
        Name of the dataset to use.
    dataset_path : str
        Path to the dataset folder.
    build : bool
        Whether to build the indices.
    search : bool
        Whether to perform the search.
    algorithms : Optional[str]
        Comma-separated list of algorithm names to use.
    groups : str
        Comma-separated list of groups to consider.
    algo_groups : Optional[str]
        Comma-separated list of algorithm groups to consider.
    force : bool
        Whether to force the execution regardless of warnings.
    search_mode : str
        The mode of search to perform ('latency' or 'throughput').
    search_threads : Optional[str]
        The number of threads to use for throughput benchmark.
    dry_run : bool
        Whether to perform a dry run without actual execution.
    raft_log_level : str
        The logging level for the RAFT library.

    """

    run_benchmark(**locals())


if __name__ == "__main__":
    main()

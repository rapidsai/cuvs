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


import itertools
from unittest.mock import MagicMock, mock_open, patch

import pytest
from benchmark import (
    find_executable,
    gather_algorithm_configs,
    get_dataset_configuration,
    load_algorithms_conf,
    load_yaml_file,
    prepare_conf_file,
    prepare_executables,
    prepare_indexes,
    rmm_present,
    validate_algorithm,
    validate_constraints,
    validate_search_params,
)


def test_load_yaml_file():
    yaml_content = """
    key: value
    """
    with patch("builtins.open", mock_open(read_data=yaml_content)):
        result = load_yaml_file("dummy_path.yaml")
        assert result == {"key": "value"}


def test_get_dataset_configuration():
    dataset_conf_all = [{"name": "dataset1"}, {"name": "dataset2"}]
    result = get_dataset_configuration("dataset1", dataset_conf_all)
    assert result == {"name": "dataset1"}
    with pytest.raises(ValueError):
        get_dataset_configuration("non_existent_dataset", dataset_conf_all)


def test_prepare_conf_file():
    dataset_conf = {"name": "dataset1"}
    result = prepare_conf_file(dataset_conf, 1000, 10, 128)
    expected_result = {
        "dataset": {"name": "dataset1", "subset_size": 1000},
        "search_basic_param": {"k": 10, "batch_size": 128},
    }
    assert result == expected_result
    result_no_subset = prepare_conf_file(dataset_conf, None, 10, 128)
    assert result_no_subset["dataset"].get("subset_size") is None


def test_gather_algorithm_configs(tmpdir):
    scripts_path = tmpdir.mkdir("scripts")
    algos_path = scripts_path.mkdir("algos")
    algos_path.join("algo1.yaml").write("key: value")
    algos_path.join("algo2.yaml").write("key: value")
    result = gather_algorithm_configs(str(scripts_path), None)
    assert len(result) == 2

    custom_conf_dir = tmpdir.mkdir("custom_conf")
    custom_conf_dir.join("custom_algo.yaml").write("key: value")
    result = gather_algorithm_configs(str(scripts_path), str(custom_conf_dir))
    assert len(result) == 3

    custom_conf_file = custom_conf_dir.join("custom_algo_file.yaml")
    custom_conf_file.write("key: value")
    result = gather_algorithm_configs(str(scripts_path), str(custom_conf_file))
    assert len(result) == 4


def test_load_algorithms_conf():
    algos_conf_fs = ["path/to/algo1.yaml", "path/to/algo2.yaml"]
    yaml_content = """
    name: algo1
    groups:
      group1: {}
    """
    with patch("builtins.open", mock_open(read_data=yaml_content)):
        result = load_algorithms_conf(algos_conf_fs, None, None)
        assert "algo1" in result

    with patch("builtins.open", mock_open(read_data=yaml_content)):
        result = load_algorithms_conf(algos_conf_fs, ["algo1"], None)
        assert "algo1" in result
        result = load_algorithms_conf(algos_conf_fs, ["algo2"], None)
        assert "algo1" not in result


@patch(
    "benchmark.find_executable",
    return_value=("executable", "path", "filename"),
)
@patch("benchmark.validate_algorithm", return_value=True)
@patch(
    "benchmark.prepare_indexes", return_value=[{"index_key": "index_value"}]
)
def test_prepare_executables(
    mock_prepare_indexes, mock_validate_algorithm, mock_find_executable
):
    algos_conf = {"algo1": {"groups": {"group1": {"build": {}, "search": {}}}}}
    algos_yaml = {"algo1": {}}
    gpu_present = True
    conf_file = {}
    dataset_path = "dataset_path"
    dataset = "dataset"
    count = 10
    batch_size = 128
    result = prepare_executables(
        algos_conf,
        algos_yaml,
        gpu_present,
        conf_file,
        dataset_path,
        dataset,
        count,
        batch_size,
    )
    assert "executable" in result
    assert len(result["executable"]["index"]) == 1


def test_prepare_indexes():
    group_conf = {"build": {"param1": [1, 2]}, "search": {"param2": [3, 4]}}
    conf_file = {"dataset": {"dims": 128}}
    result = prepare_indexes(
        group_conf,
        "algo",
        "group",
        conf_file,
        "dataset_path",
        "dataset",
        10,
        128,
    )
    assert len(result) == 2
    assert "param1" in result[0]["build_param"]


def test_validate_search_params():
    all_search_params = itertools.product([1, 2], [3, 4])
    search_param_names = ["param1", "param2"]
    group_conf = {}
    conf_file = {"dataset": {"dims": 128}}
    result = validate_search_params(
        all_search_params,
        search_param_names,
        "algo",
        group_conf,
        conf_file,
        10,
        128,
    )
    assert len(result) == 4


def test_rmm_present():
    with patch.dict("sys.modules", {"rmm": MagicMock()}):
        assert rmm_present() is True
    with patch.dict("sys.modules", {"rmm": None}):
        assert rmm_present() is False


@patch("benchmark.get_build_path", return_value="build_path")
def test_find_executable(mock_get_build_path):
    algos_conf = {"algo1": {"executable": "executable1"}}
    result = find_executable(algos_conf, "algo1", "group1", 10, 128)
    assert result == (
        "executable1",
        "build_path",
        ("algo1,group1", "algo1,group1,k10,bs128"),
    )
    mock_get_build_path.return_value = None
    with pytest.raises(FileNotFoundError):
        find_executable(algos_conf, "algo1", "group1", 10, 128)


def test_validate_algorithm():
    algos_conf = {"algo1": {"requires_gpu": False}}
    result = validate_algorithm(algos_conf, "algo1", gpu_present=True)
    assert result is True
    result = validate_algorithm(algos_conf, "algo1", gpu_present=False)
    assert result is True
    algos_conf["algo1"]["requires_gpu"] = True
    result = validate_algorithm(algos_conf, "algo1", gpu_present=False)
    assert result is False


@patch("benchmark.import_module")
def test_validate_constraints(mock_import_module):
    mock_validator = MagicMock()
    mock_import_module.return_value = mock_validator
    mock_validator.constraint_func.return_value = True
    algos_conf = {
        "algo1": {"constraints": {"build": "module.constraint_func"}}
    }
    result = validate_constraints(
        algos_conf, "algo1", "build", {"param1": "value1"}, 128, None, None
    )
    assert result is True

    algos_conf = {"algo1": {"constraints": {}}}
    result = validate_constraints(
        algos_conf, "algo1", "build", {"param1": "value1"}, 128, None, None
    )
    assert result is True

    mock_validator.constraint_func.return_value = False
    algos_conf["algo1"]["constraints"]["build"] = "module.constraint_func"
    result = validate_constraints(
        algos_conf, "algo1", "build", {"param1": "value1"}, 128, None, None
    )
    assert result is False

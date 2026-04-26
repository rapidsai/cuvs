#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for shared backend utilities and Dataset lazy loading.
"""

import numpy as np
import pytest
import yaml

from cuvs_bench.backends import Dataset
from cuvs_bench.backends.utils import dtype_from_filename, load_vectors
from cuvs_bench.orchestrator.config_loaders import CppGBenchConfigLoader


def _write_test_bin(path, data):
    """Write a numpy array in big-ann-bench binary format."""
    with open(path, "wb") as f:
        np.asarray(data.shape, dtype=np.uint32).tofile(f)
        data.tofile(f)


class TestDtypeFromFilename:
    """Tests for dtype_from_filename."""

    def test_fbin(self):
        """Test .fbin maps to float32."""
        assert dtype_from_filename("vectors.fbin") == np.float32

    def test_f16bin(self):
        """Test .f16bin maps to float16."""
        assert dtype_from_filename("vectors.f16bin") == np.float16

    def test_ibin(self):
        """Test .ibin maps to int32."""
        assert dtype_from_filename("groundtruth.ibin") == np.int32

    def test_u8bin(self):
        """Test .u8bin maps to uint8."""
        assert dtype_from_filename("vectors.u8bin") == np.ubyte

    def test_i8bin(self):
        """Test .i8bin maps to int8."""
        assert dtype_from_filename("vectors.i8bin") == np.byte

    def test_unsupported_extension(self):
        """Test that unsupported extensions raise RuntimeError."""
        with pytest.raises(RuntimeError, match="Unsupported file extension"):
            dtype_from_filename("vectors.txt")

    def test_full_path(self):
        """Test that full paths are handled correctly."""
        assert dtype_from_filename("/data/datasets/sift/base.fbin") == np.float32


class TestLoadVectors:
    """Tests for load_vectors."""

    def test_load_float32(self, tmp_path):
        """Test loading float32 vectors."""
        data = np.random.rand(100, 128).astype(np.float32)
        path = str(tmp_path / "test.fbin")
        _write_test_bin(path, data)

        loaded = load_vectors(path)
        np.testing.assert_array_equal(loaded, data)

    def test_load_int32(self, tmp_path):
        """Test loading int32 vectors."""
        data = np.random.randint(0, 1000, size=(50, 10)).astype(np.int32)
        path = str(tmp_path / "test.ibin")
        _write_test_bin(path, data)

        loaded = load_vectors(path)
        np.testing.assert_array_equal(loaded, data)

    def test_load_uint8(self, tmp_path):
        """Test loading uint8 vectors."""
        data = np.random.randint(0, 255, size=(30, 64)).astype(np.uint8)
        path = str(tmp_path / "test.u8bin")
        _write_test_bin(path, data)

        loaded = load_vectors(path)
        np.testing.assert_array_equal(loaded, data)

    def test_load_int8(self, tmp_path):
        """Test loading int8 vectors."""
        data = np.random.randint(-128, 127, size=(30, 64)).astype(np.int8)
        path = str(tmp_path / "test.i8bin")
        _write_test_bin(path, data)

        loaded = load_vectors(path)
        np.testing.assert_array_equal(loaded, data)

    def test_subset_size(self, tmp_path):
        """Test that subset_size limits the number of rows loaded."""
        data = np.random.rand(100, 128).astype(np.float32)
        path = str(tmp_path / "test.fbin")
        _write_test_bin(path, data)

        loaded = load_vectors(path, subset_size=10)
        assert loaded.shape == (10, 128)
        np.testing.assert_array_equal(loaded, data[:10])

    def test_subset_size_larger_than_file(self, tmp_path):
        """Test that subset_size larger than file returns all rows."""
        data = np.random.rand(5, 16).astype(np.float32)
        path = str(tmp_path / "test.fbin")
        _write_test_bin(path, data)

        loaded = load_vectors(path, subset_size=100)
        assert loaded.shape == (5, 16)
        np.testing.assert_array_equal(loaded, data)

    def test_negative_subset_size(self, tmp_path):
        """Test that negative subset_size raises ValueError."""
        data = np.random.rand(10, 4).astype(np.float32)
        path = str(tmp_path / "test.fbin")
        _write_test_bin(path, data)

        with pytest.raises(ValueError, match="subset_size must be a positive integer"):
            load_vectors(path, subset_size=-1)

    def test_zero_subset_size(self, tmp_path):
        """Test that zero subset_size raises ValueError."""
        data = np.random.rand(10, 4).astype(np.float32)
        path = str(tmp_path / "test.fbin")
        _write_test_bin(path, data)

        with pytest.raises(ValueError, match="subset_size must be a positive integer"):
            load_vectors(path, subset_size=0)

    def test_unsupported_extension(self, tmp_path):
        """Test that unsupported file extensions raise RuntimeError."""
        path = str(tmp_path / "test.txt")
        with open(path, "wb") as f:
            f.write(b"dummy")

        with pytest.raises(RuntimeError, match="Unsupported file extension"):
            load_vectors(path)

    def test_truncated_header(self, tmp_path):
        """Test that truncated headers raise ValueError."""
        path = str(tmp_path / "test.fbin")
        with open(path, "wb") as f:
            f.write(b"\x00\x00")

        with pytest.raises(ValueError, match="File too small to contain a valid header"):
            load_vectors(path)

    def test_truncated_data(self, tmp_path):
        """Test that truncated data raises ValueError."""
        path = str(tmp_path / "test.fbin")
        with open(path, "wb") as f:
            np.array([10, 4], dtype=np.uint32).tofile(f)
            np.random.rand(5, 4).astype(np.float32).tofile(f)

        with pytest.raises(ValueError, match="File is truncated"):
            load_vectors(path)

    def test_file_not_found(self):
        """Test that missing files raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_vectors("/nonexistent/path/vectors.fbin")


class TestDatasetLazyLoading:
    """Tests for Dataset transparent vector loading."""

    def test_lazy_load_base_vectors(self, tmp_path):
        """Test that base vectors are loaded from file on first access."""
        data = np.random.rand(50, 32).astype(np.float32)
        path = str(tmp_path / "base.fbin")
        _write_test_bin(path, data)

        dataset = Dataset(name="test", base_file=path)
        np.testing.assert_array_equal(dataset.base_vectors, data)

    def test_lazy_load_query_vectors(self, tmp_path):
        """Test that query vectors are loaded from file on first access."""
        data = np.random.rand(10, 32).astype(np.float32)
        path = str(tmp_path / "query.fbin")
        _write_test_bin(path, data)

        dataset = Dataset(name="test", query_file=path)
        np.testing.assert_array_equal(dataset.query_vectors, data)

    def test_lazy_load_groundtruth(self, tmp_path):
        """Test that ground truth is loaded from file on first access."""
        data = np.random.randint(0, 100, size=(10, 5)).astype(np.int32)
        path = str(tmp_path / "gt.ibin")
        _write_test_bin(path, data)

        dataset = Dataset(name="test", groundtruth_neighbors_file=path)
        np.testing.assert_array_equal(dataset.groundtruth_neighbors, data)

    def test_preloaded_vectors_returned_directly(self):
        """Test that pre-loaded vectors are returned without file access."""
        base = np.random.rand(20, 16).astype(np.float32)
        query = np.random.rand(5, 16).astype(np.float32)

        dataset = Dataset(
            name="test",
            base_vectors=base,
            query_vectors=query,
        )
        np.testing.assert_array_equal(dataset.base_vectors, base)
        np.testing.assert_array_equal(dataset.query_vectors, query)

    def test_no_file_returns_empty(self):
        """Test that Dataset without files returns empty arrays."""
        dataset = Dataset(name="test")
        assert dataset.base_vectors.size == 0
        assert dataset.query_vectors.size == 0
        assert dataset.groundtruth_neighbors is None

    def test_lazy_load_with_subset_size(self, tmp_path):
        """Test that subset_size is respected during lazy loading."""
        data = np.random.rand(100, 32).astype(np.float32)
        path = str(tmp_path / "base.fbin")
        _write_test_bin(path, data)

        dataset = Dataset(
            name="test",
            base_file=path,
            metadata={"subset_size": 10},
        )
        assert dataset.base_vectors.shape == (10, 32)
        np.testing.assert_array_equal(dataset.base_vectors, data[:10])

    def test_file_path_still_accessible(self):
        """Test that file paths are accessible without triggering loading."""
        dataset = Dataset(
            name="test",
            base_file="/path/to/base.fbin",
            query_file="/path/to/query.fbin",
        )
        assert dataset.base_file == "/path/to/base.fbin"
        assert dataset.query_file == "/path/to/query.fbin"
        assert dataset.name == "test"

    def test_file_path_access_does_not_trigger_loading(self, tmp_path):
        """Test that accessing file paths does not load vectors into memory."""
        data = np.random.rand(50, 32).astype(np.float32)
        path = str(tmp_path / "base.fbin")
        _write_test_bin(path, data)

        dataset = Dataset(name="test", base_file=path)

        _ = dataset.base_file
        _ = dataset.name
        _ = dataset.distance_metric

        assert dataset._base_vectors.size == 0

    def test_dims_and_counts(self, tmp_path):
        """Test dims, n_base, and n_queries properties."""
        data = np.random.rand(50, 32).astype(np.float32)
        path = str(tmp_path / "base.fbin")
        _write_test_bin(path, data)

        queries = np.random.rand(10, 32).astype(np.float32)
        qpath = str(tmp_path / "query.fbin")
        _write_test_bin(qpath, queries)

        dataset = Dataset(name="test", base_file=path, query_file=qpath)
        assert dataset.dims == 32
        assert dataset.n_base == 50
        assert dataset.n_queries == 10

    def test_caching(self, tmp_path):
        """Test that vectors are loaded once and cached."""
        data = np.random.rand(10, 4).astype(np.float32)
        path = str(tmp_path / "base.fbin")
        _write_test_bin(path, data)

        dataset = Dataset(name="test", base_file=path)
        first_access = dataset.base_vectors
        second_access = dataset.base_vectors
        assert first_access is second_access


class TestConfigLoaderMethods:
    """Tests for base ConfigLoader inherited methods."""

    def test_load_yaml_file(self, tmp_path):
        """Test loading a YAML file."""
        yaml_path = str(tmp_path / "test.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump({"name": "test", "value": 42}, f)

        loader = CppGBenchConfigLoader(config_path=str(tmp_path))
        result = loader.load_yaml_file(yaml_path)
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_get_dataset_configuration(self):
        """Test finding a dataset by name."""
        loader = CppGBenchConfigLoader()
        datasets = [
            {"name": "sift-128", "dims": 128},
            {"name": "glove-100", "dims": 100},
        ]

        result = loader.get_dataset_configuration("glove-100", datasets)
        assert result["name"] == "glove-100"
        assert result["dims"] == 100

    def test_get_dataset_configuration_not_found(self):
        """Test that missing datasets raise ValueError."""
        loader = CppGBenchConfigLoader()
        datasets = [{"name": "sift-128"}]

        with pytest.raises(ValueError, match="Could not find a dataset configuration"):
            loader.get_dataset_configuration("nonexistent", datasets)

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


_ROOT = Path(__file__).parents[1]
_BUILD_WORKFLOW = _ROOT / ".github" / "workflows" / "build.yaml"
_MATRIX_WORKFLOW = _ROOT / ".github" / "workflows" / "matrix-job.yaml"
_MATRIX_FIXTURE = _ROOT / "tests" / "fixtures" / "conda-cpp-build-matrix.yaml"


def _selected_candidate_rows(workflow, job_name):
    jq = shutil.which("jq")
    if jq is None:
        pytest.skip(
            "jq is required to evaluate the shared-workflow matrix filter"
        )

    expression = workflow["jobs"][job_name]["with"]["matrix_filter"]
    match = re.search(r"&& '(.*)' \|\|", expression)
    assert match is not None
    candidate_filter = match.group(1)

    matrix = yaml.safe_load(_MATRIX_FIXTURE.read_text(encoding="utf-8"))
    result = subprocess.run(
        [jq, "-c", candidate_filter],
        input=json.dumps(matrix),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("job_name", ["java-build", "lucene-build"])
def test_candidate_maven_matrix_has_one_row_per_architecture_and_cuda_major(
    job_name,
):
    workflow = yaml.safe_load(_BUILD_WORKFLOW.read_text(encoding="utf-8"))
    selected = _selected_candidate_rows(workflow, job_name)

    identities = [
        (row["ARCH"], row["CUDA_VER"].split(".")[0]) for row in selected
    ]
    assert identities == [
        ("amd64", "12"),
        ("amd64", "13"),
        ("arm64", "12"),
        ("arm64", "13"),
    ]
    assert len(set(identities)) == len(identities)
    assert {row["PY_VER"] for row in selected} == {"3.12"}


@pytest.mark.parametrize("job_name", ["java-build", "lucene-build"])
def test_candidate_maven_matrix_filter_preserves_production_arch_filter(
    job_name,
):
    workflow = yaml.safe_load(_BUILD_WORKFLOW.read_text(encoding="utf-8"))
    expression = workflow["jobs"][job_name]["with"]["matrix_filter"]
    assert "|| 'map(select(.ARCH == \"amd64\"))'" in expression


@pytest.mark.parametrize(
    ("job_name", "artifact_base"),
    [("java-build", "cuvs-java"), ("lucene-build", "cuvs-lucene")],
)
def test_candidate_maven_artifact_names_are_unique_and_matrix_scoped(
    job_name, artifact_base
):
    caller = yaml.safe_load(_BUILD_WORKFLOW.read_text(encoding="utf-8"))
    callee = yaml.safe_load(_MATRIX_WORKFLOW.read_text(encoding="utf-8"))
    caller_inputs = caller["jobs"][job_name]["with"]
    artifact_name = caller_inputs["candidate-artifact-name"]
    assert "matrix." not in artifact_name
    assert f"'{artifact_base}'" in artifact_name
    assert caller_inputs["artifact-name-prefix"] == artifact_base

    run = callee["jobs"]["run"]["with"]
    candidate_name_expression = run["candidate-artifact-name"]
    assert "inputs.candidate-artifact-name" in candidate_name_expression
    assert "matrix.ARCH" in candidate_name_expression
    assert "matrix.CUDA_VER" in candidate_name_expression
    assert "fromJSON" not in candidate_name_expression
    assert (
        "custom-job.yaml@9ac257258ee48137e797fc3e122f992f9ab17b9c"
        in callee["jobs"]["run"]["uses"]
    )

    rows = _selected_candidate_rows(caller, job_name)
    artifact_names = {
        f"{artifact_base}-{row['ARCH']}-cuda{row['CUDA_VER']}" for row in rows
    }
    assert artifact_names == {
        f"{artifact_base}-amd64-cuda12.9.1",
        f"{artifact_base}-amd64-cuda13.0.2",
        f"{artifact_base}-arm64-cuda12.9.1",
        f"{artifact_base}-arm64-cuda13.0.2",
    }

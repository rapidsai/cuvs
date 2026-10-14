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
_MATRIX_FIXTURE = _ROOT / "tests" / "fixtures" / "conda-cpp-build-matrix.yaml"


@pytest.mark.parametrize("job_name", ["java-build", "lucene-build"])
def test_candidate_maven_matrix_has_one_row_per_architecture_and_cuda_major(
    job_name,
):
    jq = shutil.which("jq")
    if jq is None:
        pytest.skip(
            "jq is required to evaluate the shared-workflow matrix filter"
        )

    workflow = yaml.safe_load(_BUILD_WORKFLOW.read_text(encoding="utf-8"))
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
    selected = json.loads(result.stdout)

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

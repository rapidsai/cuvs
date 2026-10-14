# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parents[1] / "ci" / "package_candidate_maven.sh"
_GNU_TAR = shutil.which("gtar") or shutil.which("tar")


@pytest.mark.parametrize(
    ("artifact", "group_path", "identity_name"),
    [
        ("cuvs-java", "com/nvidia/cuvs", "com.nvidia.cuvs:cuvs-java"),
        (
            "cuvs-lucene",
            "com/nvidia/cuvs/lucene",
            "com.nvidia.cuvs.lucene:cuvs-lucene",
        ),
    ],
)
def test_candidate_maven_archive_uses_frozen_identity_and_maven_layout(
    tmp_path, artifact, group_path, identity_name
):
    if _GNU_TAR is None:
        pytest.skip("tar is unavailable")

    repository = tmp_path / "repo"
    script_directory = repository / "ci"
    script_directory.mkdir(parents=True)
    script = script_directory / "package_candidate_maven.sh"
    shutil.copyfile(_SCRIPT, script)
    script.chmod(0o755)

    module = repository / "java" / artifact
    target = module / "target"
    target.mkdir(parents=True)
    version = "26.12.0"
    (module / "pom.xml").write_text("<project/>\n", encoding="utf-8")
    (target / f"{artifact}-{version}.jar").write_bytes(b"primary jar")
    if artifact == "cuvs-java":
        (target / f"{artifact}-{version}-x86_64-cuda12.jar").write_bytes(
            b"native jar"
        )
    (target / f"{artifact}-{version}-tests.jar").write_bytes(b"test jar")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_maven = fake_bin / "mvn"
    fake_maven.write_text(
        "#!/bin/sh\nprintf '%s\\n' '26.12.0'\n", encoding="utf-8"
    )
    fake_maven.chmod(0o755)
    fake_uname = fake_bin / "uname"
    fake_uname.write_text(
        "#!/bin/sh\nprintf '%s\\n' 'x86_64'\n", encoding="utf-8"
    )
    fake_uname.chmod(0o755)
    gnu_tar_link = fake_bin / "tar"
    gnu_tar_link.symlink_to(_GNU_TAR)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "RAPIDS_RELEASE_CANDIDATE": "true",
        "RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS": json.dumps(
            {
                "unit_package_versions": {f"maven:{artifact}": version},
                "package_versions": {"maven": "26.10.0"},
            }
        ),
        "RAPIDS_CONDA_ARCH": "amd64",
        "RAPIDS_CUDA_VERSION": "12.2",
    }
    subprocess.run(["/bin/bash", str(script), artifact], check=True, env=env)  # noqa: S603

    output = repository / ".release-candidate-artifacts" / "maven"
    archive = output / f"{artifact}.tar.gz"
    identity = json.loads(
        (output / f"{artifact}.identity.json").read_text(encoding="utf-8")
    )
    assert identity == {
        "ecosystem": "maven",
        "name": identity_name,
        "platform": "amd64-cuda12",
        "version": version,
    }  # noqa: S101
    members = subprocess.run(
        [_GNU_TAR, "-tzf", str(archive)],
        check=True,
        capture_output=True,
        text=True,
    )  # noqa: S603
    base = f"{group_path}/{artifact}/{version}"
    assert f"{base}/{artifact}-{version}.jar" in members.stdout.splitlines()  # noqa: S101
    assert f"{base}/{artifact}-{version}.pom" in members.stdout.splitlines()  # noqa: S101
    assert f"{base}/{artifact}-{version}-tests.jar" not in members.stdout  # noqa: S101
    if artifact == "cuvs-java":
        assert (
            f"{base}/{artifact}-{version}-x86_64-cuda12.jar"
            in members.stdout.splitlines()
        )  # noqa: S101


def test_candidate_cuvs_java_archive_requires_current_matrix_classifier(
    tmp_path,
):
    repository = tmp_path / "repo"
    script_directory = repository / "ci"
    script_directory.mkdir(parents=True)
    script = script_directory / "package_candidate_maven.sh"
    shutil.copyfile(_SCRIPT, script)
    script.chmod(0o755)
    module = repository / "java" / "cuvs-java"
    target = module / "target"
    target.mkdir(parents=True)
    (module / "pom.xml").write_text("<project/>\n", encoding="utf-8")
    (target / "cuvs-java-26.12.0.jar").write_bytes(b"primary jar")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_maven = fake_bin / "mvn"
    fake_maven.write_text(
        "#!/bin/sh\nprintf '%s\\n' '26.12.0'\n", encoding="utf-8"
    )
    fake_maven.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "RAPIDS_RELEASE_CANDIDATE": "true",
        "RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS": json.dumps(
            {"unit_package_versions": {"maven:cuvs-java": "26.12.0"}}
        ),
        "RAPIDS_CONDA_ARCH": "amd64",
        "RAPIDS_CUDA_VERSION": "12.2",
    }
    result = subprocess.run(
        ["/bin/bash", str(script), "cuvs-java"],
        env=env,
        capture_output=True,
        text=True,
    )  # noqa: S603
    assert result.returncode != 0  # noqa: S101
    assert "missing cuVS Java classifier JAR" in result.stderr  # noqa: S101

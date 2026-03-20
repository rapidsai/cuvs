#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Pytest fixtures for integration tests.

Requires: pip install cuvs-bench[elastic,integration]
Requires: Docker running locally
"""

import pytest


def _testcontainers_available():
    try:
        from testcontainers.elasticsearch import ElasticSearchContainer
        return True
    except ImportError:
        return False


def _elasticsearch_installed():
    try:
        import elasticsearch  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture(scope="module")
def elasticsearch_container():
    """Start an Elasticsearch container for the duration of the test module.

    Yields a dict with host, port, and get_url() for connecting.
    Skips the test module if testcontainers or elasticsearch is not installed.
    """
    if not _testcontainers_available():
        pytest.skip(
            "Requires testcontainers[elasticsearch]. "
            "Install with: pip install cuvs-bench[integration]"
        )
    if not _elasticsearch_installed():
        pytest.skip(
            "Requires elasticsearch. Install with: pip install cuvs-bench[elastic]"
        )

    from testcontainers.elasticsearch import ElasticSearchContainer

    # Use standard ES OSS image (no GPU in OSS; use use_gpu=False in tests)
    with ElasticSearchContainer(
        image="elasticsearch:8.15.0",
        mem_limit="2g",
    ) as container:
        url = container.get_url()
        # Parse host:port from URL (e.g. http://localhost:32768)
        if url.startswith("http://"):
            rest = url[7:]
        elif url.startswith("https://"):
            rest = url[8:]
        else:
            rest = url
        if "/" in rest:
            host_port = rest.split("/")[0]
        else:
            host_port = rest
        if ":" in host_port:
            host, port_str = host_port.rsplit(":", 1)
            port = int(port_str)
        else:
            host = host_port
            port = 9200

        yield {"host": host, "port": port, "url": url}

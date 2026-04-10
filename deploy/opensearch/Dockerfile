# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM opensearchproject/opensearch:latest

# The repository-s3 plugin is not bundled in the default OpenSearch image but
# is required for the remote vector index build feature: OpenSearch uses it to
# upload raw vectors and download GPU-built indexes via the MinIO bucket.
RUN /usr/share/opensearch/bin/opensearch-plugin install --batch repository-s3

# Pre-populate the keystore with MinIO credentials. The keystore lives in
# /usr/share/opensearch/config (not the data volume) so this survives restarts.
# The repository-s3 plugin reads credentials exclusively from the keystore.
RUN /usr/share/opensearch/bin/opensearch-keystore create && \
    echo "minioadmin" | /usr/share/opensearch/bin/opensearch-keystore add --stdin s3.client.default.access_key && \
    echo "minioadmin" | /usr/share/opensearch/bin/opensearch-keystore add --stdin s3.client.default.secret_key

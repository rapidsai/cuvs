#!/bin/bash
#
# Populate the OpenSearch keystore with S3 credentials from environment variables,
# then start OpenSearch. Doing this at runtime (not image build time) avoids
# baking credentials into image layers.
#
# Required environment variables:
#   AWS_ACCESS_KEY_ID       AWS access key ID
#   AWS_SECRET_ACCESS_KEY   AWS secret access key
#
# Optional environment variables:
#   AWS_SESSION_TOKEN       STS session token (required for temporary credentials)
#
set -e

# The repository-s3 plugin reads credentials exclusively from the keystore.
# If credentials are not set, skip keystore setup — S3 and remote index build
# will be unavailable, but OpenSearch itself will start normally.
if [ -n "${AWS_ACCESS_KEY_ID}" ] && [ -n "${AWS_SECRET_ACCESS_KEY}" ]; then
    rm -f /usr/share/opensearch/config/opensearch.keystore
    /usr/share/opensearch/bin/opensearch-keystore create
    printf '%s' "${AWS_ACCESS_KEY_ID}"       | /usr/share/opensearch/bin/opensearch-keystore add --stdin s3.client.default.access_key
    printf '%s' "${AWS_SECRET_ACCESS_KEY}"   | /usr/share/opensearch/bin/opensearch-keystore add --stdin s3.client.default.secret_key
    if [ -n "${AWS_SESSION_TOKEN}" ]; then
        printf '%s' "${AWS_SESSION_TOKEN}"   | /usr/share/opensearch/bin/opensearch-keystore add --stdin s3.client.default.session_token
    fi
else
    echo "Warning: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set — S3 repository and remote index build will not be available" >&2
fi

exec /usr/share/opensearch/bin/opensearch

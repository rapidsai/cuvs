FROM opensearchproject/opensearch:3.6.0

# The repository-s3 plugin is not bundled in the default OpenSearch image but
# is required for the remote vector index build feature: OpenSearch uses it to
# upload raw vectors and download GPU-built indexes via S3.
RUN /usr/share/opensearch/bin/opensearch-plugin install --batch repository-s3

# entrypoint.sh populates the keystore from S3_ACCESS_KEY / S3_SECRET_KEY
# environment variables at container startup, then execs opensearch.
COPY --chmod=755 entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Build only after inspecting the existing model image and approving build/storage costs.
# Both arguments MUST be immutable registry image@sha256:... references.
ARG OPENSEARCH_IMAGE
ARG MODEL_IMAGE
FROM ${OPENSEARCH_IMAGE} AS engine
FROM ${MODEL_IMAGE}
ARG VALIDATION_INPUT_DIR
USER root
COPY --from=engine /usr/share/opensearch /opt/opensearch
# Omit the optional performance-analyzer agent/listener entirely.
RUN rm -rf /opt/opensearch/plugins/opensearch-performance-analyzer \
    && rm -rf /opt/opensearch/logs /opt/opensearch/data \
    && ln -s /validation-work/opensearch-logs /opt/opensearch/logs \
    && ln -s /validation-work/opensearch-data /opt/opensearch/data \
    && mkdir -p /validation-work \
    && chown -R 10001:10001 /opt/opensearch /validation-work
COPY deploy/cloud-run-validation/opensearch.yml /opt/opensearch/config/opensearch.yml
COPY search_agent /validation-app/search_agent
COPY ${VALIDATION_INPUT_DIR}/ /validation-input/
WORKDIR /validation-app
ENV PYTHONPATH=/validation-app PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    OPENSEARCH_JAVA_HOME=/opt/opensearch/jdk \
    OPENSEARCH_JAVA_OPTS="-Xms512m -Xmx512m -XX:ActiveProcessorCount=2"
USER 10001:10001
ENTRYPOINT ["python", "-m", "search_agent.validation_job"]

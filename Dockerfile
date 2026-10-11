# Review/build template only. Pin the tested base digest and frozen dependencies before deployment.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY search_agent /app/search_agent
COPY data /app/data
RUN pip install --no-cache-dir '.[models]' \
    && useradd --create-home --uid 10001 search \
    && mkdir -p /app/.cache /app/.media \
    && chown -R search:search /app
USER search
ENV PYTHONUNBUFFERED=1
CMD ["sh", "-c", "exec python -m uvicorn search_agent.app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]

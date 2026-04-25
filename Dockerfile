FROM python:3.11-slim

# Install uv — dramatically faster than pip
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# HF Spaces runs as non-root
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy only dependency manifests first (maximises cache hit on rebuilds)
COPY pyproject.toml uv.lock ./

# Install runtime dependencies into the system Python (no venv).
# `pyproject.toml` is not a requirements file; install the explicit server deps
# here, then install the local package after the source has been copied.
RUN uv pip install --system \
    "openenv-core>=0.2.3" \
    "pydantic>=2.0" \
    "httpx>=0.20.0" \
    "openai>=1.0.0" \
    "python-dotenv>=1.0.1" \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.30.0" \
    "websockets>=12.0" \
    "json-repair>=0.28"

# Copy application code
COPY godel_engine/ ./godel_engine/
COPY server/ ./server/
COPY dashboard/ ./dashboard/
COPY inference.py baseline.py ./

# Install the local package itself
RUN uv pip install --system --no-deps -e .

# Fix ownership for HF non-root user
RUN chown -R user:user /app
USER user

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

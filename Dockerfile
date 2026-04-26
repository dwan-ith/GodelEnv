FROM python:3.11-slim

# Install uv — faster than pip
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# HF Spaces runs as non-root
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy only dependency manifests first (maximises cache hit on rebuilds)
COPY pyproject.toml requirements.txt uv.lock ./

# Install runtime dependencies into the system Python (no venv).
RUN uv pip install --system -r requirements.txt

# Copy application code
COPY godel_engine/ ./godel_engine/
COPY server/ ./server/
COPY dashboard/ ./dashboard/
COPY app.py inference.py baseline.py ./

# Install the local package itself
RUN uv pip install --system --no-deps -e .

# Fix ownership for HF non-root user
RUN chown -R user:user /app
USER user

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

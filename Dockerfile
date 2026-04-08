FROM python:3.11-slim

# HF Spaces runs as non-root
RUN useradd -m -u 1000 user
WORKDIR /app

# Install dependencies first (cache layer)
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install the godel_engine package
COPY pyproject.toml ./
COPY godel_engine/ ./godel_engine/
RUN pip install --no-cache-dir -e .

# Copy the rest of the application
COPY . .

# Ensure the non-root user owns the files
RUN chown -R user:user /app
USER user

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

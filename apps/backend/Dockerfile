# Build from the REPO ROOT so the uv workspace resolves:
#   docker build -f apps/backend/Dockerfile -t zo-backend .
FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY . /app

# Install the workspace (no dev tools, no GPU extras — control plane only).
RUN uv sync --no-dev

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "zo_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

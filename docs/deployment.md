# Deployment Guide

This document outlines instructions for deploying the **IntelliDoc** service in production environments.

## System Requirements

- **OS:** Linux (Ubuntu 22.04 LTS recommended) or Windows Server
- **GPU:** NVIDIA GPU with CUDA support (Compute Capability >= 8.0, e.g., RTX 4090, A10G, or T4)
- **RAM:** Minimum 16 GB RAM
- **Storage:** 20 GB SSD storage (for Hugging Face models and PyTorch runtime caching)

---

## Architecture Diagram (Production Deployment)

```
                       +-------------------+
                       |  Nginx / Ingress  |
                       +---------+---------+
                                 |
                     +-----------v-----------+
                     |  FastAPI Application  |
                     +----+-------------+----+
                          |             |
            +-------------v-----+ +-----v-------------+
            |  LangGraph Agent  | |  Redis Cache Layer |
            +-------------+-----+ +-------------------+
                          |
      +-------------------+-------------------+
      |                                       |
+-----v-------------+                   +-----v-------------+
| Transformers (GPU)|                   |   ColBERT Server  |
+-------------------+                   +-------------------+
```

---

## Step-by-Step Production Setup

### 1. Docker Deployment (Recommended)

Create a `docker-compose.yml` file to run the FastAPI app and Redis cache together.

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: always

  intellidoc:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - redis
    restart: always

volumes:
  redis_data:
```

### 2. Environment Variables

Configure the following environment variables in a `.env` file (do not commit this to version control):

| Variable | Description | Default |
| :--- | :--- | :--- |
| `REDIS_URL` | Connection string for Redis cache | `redis://localhost:6379/0` |
| `HF_HOME` | Path to download HF model weights | `~/.cache/huggingface` |
| `PORT` | API server listening port | `8000` |
| `LOG_LEVEL` | Log verbosity | `INFO` |

### 3. Monitoring & Health Checks

The FastAPI app exposes a `/health` endpoint to monitor pipeline performance:

```bash
curl -X GET http://localhost:8000/health
```

Expected Response:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "redis_connected": true,
  "loaded_models": ["roberta-base-squad2"]
}
```

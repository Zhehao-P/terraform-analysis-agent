# Running with Docker

This document explains how to run the Terraform Analysis Agent using Docker with an existing Qdrant instance.

## Prerequisites

- Docker installed on your system
- Qdrant instance already running

## Configuration

1. Create a `.env` file from the example:

```bash
cp .env.example .env
```

2. Update the `.env` file with your API credentials and Qdrant connection details:

```
# API Key for embeddings
LLM_API_KEY=your_api_key_here

# API Base URL
LLM_BASE_URL=https://api.openai.com/v1

# Embedding model to use
EMBEDDING_MODEL_CHOICE=text-embedding-3-small

# Qdrant connection
QDRANT_HOST=your_qdrant_host  # e.g., localhost or container name
QDRANT_PORT=6333
```

## Building and Running

1. Build the Docker image:
```bash
docker build -t mcp/terraform-analysis-agent .
```

2. Run the container:
```bash
docker run -d \
  --name terraform-agent \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  --restart unless-stopped \
  mcp/terraform-analysis-agent
```

## Managing the Container

```bash
# View logs
docker logs -f terraform-agent

# Stop container
docker stop terraform-agent

# Start container
docker start terraform-agent

# Remove container
docker rm terraform-agent
```

## Connecting to Qdrant

If your Qdrant is running in a different Docker container:

1. Find Qdrant's network:
```bash
# List all networks
docker network ls

# Inspect Qdrant container to find its network
docker inspect qdrant | grep NetworkMode
```

2. Connect to the same network:
```bash
docker network connect <network_name> terraform-agent
```

If your Qdrant is running on the host machine:
- Use `host.docker.internal` as QDRANT_HOST in your .env file
- Or use `--network host` to share the host's network namespace

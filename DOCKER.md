# Running with Docker

This document explains how to run the Terraform Analysis Agent using Docker.

## Prerequisites

- Docker installed on your system

## Configuration

1. Create a `.env` file from the example:

```bash
cp .env.example .env
```

2. Update the `.env` file with your API credentials (all required):

```
# API Key for embeddings
LLM_API_KEY=your_api_key_here

# API Base URL
LLM_BASE_URL=https://api.openai.com/v1

# Embedding model to use
EMBEDDING_MODEL_CHOICE=text-embedding-3-small
```

## Running the Container

```bash
# Create data directories (first time only)
mkdir -p data/chromadb data/cache data/github

# Build the image
docker build -t terraform-analysis-agent .

# Run with volume mapping for data persistence
docker run -p 8050:8050 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  terraform-analysis-agent
```

## Running in the Background

For production use:

```bash
docker run -d \
  --name terraform-agent \
  -p 8050:8050 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  --restart unless-stopped \
  terraform-analysis-agent
```

## Managing the Container

```bash
# View logs
docker logs -f terraform-agent

# Stop the container
docker stop terraform-agent

# Start an existing container
docker start terraform-agent

# Remove the container
docker rm terraform-agent
```

## Customizing Port

To use a different port:

```bash
docker run -p 9000:9000 \
  --env PORT=9000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  terraform-analysis-agent
```

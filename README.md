# Terraform Analysis Agent

An MCP server implementation that analyzes Terraform configurations and repositories to help debug Terraform errors.

## Overview

This project provides an MCP (Model Context Protocol) server that can:
1. Ingest Terraform repositories for analysis
2. Search for relevant code when debugging Terraform errors
3. Integrate with any MCP-compatible client (Claude, Windsurf, etc.)

The server uses ChromaDB for vector storage and efficient semantic search.

## Features

The server provides two essential tools:

1. **`ingest_github_repo`**: Clone and index Terraform repositories for analysis
2. **`analyze_error`**: Find relevant code snippets based on Terraform error messages

## Prerequisites

- Python 3.12+
- uv package manager
- Docker (recommended for containerized deployment)
- API key for embeddings (OpenAI or other compatible provider)

## Installation

### Direct Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/terraform-analysis-agent.git
   cd terraform-analysis-agent
   ```

2. Create and activate a virtual environment with uv:
   ```bash
   # Create virtual environment with Python 3.12
   uv venv --python=3.12
   
   # Activate virtual environment
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. Install dependencies with uv:
   ```bash
   uv pip install -e .
   ```

4. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```

5. Configure your API keys in the `.env` file (see Configuration section)

### Using Docker (Recommended)

See the [Docker documentation](DOCKER.md) for detailed instructions on running with Docker.

## Configuration

The following environment variables must be configured in your `.env` file:

| Variable | Description | Example |
|----------|-------------|----------|
| `TRANSPORT` | Transport protocol (sse or stdio) | `sse` |
| `HOST` | Host to bind to when using SSE transport | `0.0.0.0` |
| `PORT` | Port to listen on when using SSE transport | `8050` |
| `LLM_API_KEY` | API key for embeddings | `sk-...` |
| `LLM_BASE_URL` | Base URL for the API | `https://api.openai.com/v1` |
| `EMBEDDING_MODEL_CHOICE` | Embedding model to use | `text-embedding-3-small` |

## Running the Server

### Using uv

#### SSE Transport

```bash
# Set TRANSPORT=sse in .env then:
uv run -m src.main
```

The MCP server will run as an API endpoint that you can then connect to with the configuration shown below.

#### Stdio Transport

With stdio, the MCP client itself can spin up the MCP server, so nothing to run at this point.

### Using Docker

#### SSE Transport

```bash
docker run --env-file .env -p 8050:8050 mcp/terraform-analysis-agent
```

The MCP server will run as an API endpoint within the container that you can then connect to with the configuration shown below.

#### Stdio Transport

With stdio, the MCP client itself can spin up the MCP server container, so nothing to run at this point.

## Integration with MCP Clients

### SSE Configuration

Once you have the server running with SSE transport, you can connect to it using this configuration:

```json
{
  "mcpServers": {
    "terraform-analysis": {
      "transport": "sse",
      "url": "http://localhost:8050/sse"
    }
  }
}
```

> **Note for Windsurf users**: Use `serverUrl` instead of `url` in your configuration:
> ```json
> {
>   "mcpServers": {
>     "terraform-analysis": {
>       "transport": "sse",
>       "serverUrl": "http://localhost:8050/sse"
>     }
>   }
> }
> ```

> **Note for n8n users**: Use host.docker.internal instead of localhost since n8n has to reach outside of its own container to the host machine:
>
> So the full URL in the MCP node would be: http://host.docker.internal:8050/sse

Make sure to update the port if you are using a value other than the default 8050.

### Python with Stdio Configuration

Add this server to your MCP configuration for Claude Desktop, Windsurf, or any other MCP client:

```json
{
  "mcpServers": {
    "terraform-analysis": {
      "command": "your/path/to/terraform-analysis-agent/.venv/Scripts/python.exe",
      "args": ["your/path/to/terraform-analysis-agent/src/main.py"],
      "env": {
        "TRANSPORT": "stdio",
        "LLM_BASE_URL": "https://api.openai.com/v1",
        "LLM_API_KEY": "YOUR-API-KEY",
        "EMBEDDING_MODEL_CHOICE": "text-embedding-3-small"
      }
    }
  }
}
```

### Docker with Stdio Configuration

```json
{
  "mcpServers": {
    "terraform-analysis": {
      "command": "docker",
      "args": ["run", "--rm", "-i",
               "-e", "TRANSPORT",
               "-e", "LLM_BASE_URL",
               "-e", "LLM_API_KEY",
               "-e", "EMBEDDING_MODEL_CHOICE",
               "mcp/terraform-analysis-agent"],
      "env": {
        "TRANSPORT": "stdio",
        "LLM_BASE_URL": "https://api.openai.com/v1",
        "LLM_API_KEY": "YOUR-API-KEY",
        "EMBEDDING_MODEL_CHOICE": "text-embedding-3-small"
      }
    }
  }
}
```

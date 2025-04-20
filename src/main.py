from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
import chromadb
import os
import asyncio
from pathlib import Path
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText
from dotenv import load_dotenv
from common.utils import (
    PayloadField,
    QdrantDB,
    FileType,
    PointStruct,
    setup_logging,
    get_embedding_function,
)

logger = setup_logging(__name__)

load_dotenv()

# Set up logging
logger = setup_logging()

GITHUB_DIR = os.getenv("GITHUB_DIR", "data/github")

@dataclass
class RepoContext:
    """Context for managing repository content storage and retrieval."""
    db_client: QdrantDB


@asynccontextmanager
async def mcp_lifespan(server: FastMCP) -> AsyncIterator[RepoContext]:
    """Manages the MCP agent lifecycle including qdrant persistence.

    This function:
    1. Uses the configured qdrant directory
    2. Initializes qdrant with persistent storage
    3. Creates or loads the collection
    4. Handles cleanup on shutdown
    """
    logger.info("Initializing qdrant client")
    embed_function = get_embedding_function()
    qdrant_db = QdrantDB(embed_fn=embed_function)

    try:
        yield RepoContext(db_client=qdrant_db)
    finally:
        # No explicit persistence needed - PersistentClient handles this automatically
        logger.info("Shutting down qdrant client")

# Initialize FastMCP server
mcp = FastMCP(
    "repo-analysis-agent",
    description="MCP server for repository content analysis using RAG",
    lifespan=mcp_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8000"),
)

@mcp.tool()
async def analyze_terraform_resource(
    ctx: Context, resource_name: str, n_results: int = 3, exclude_file_paths: list[str] = []
) -> str:
    """Find examples and documentation for Terraform resources from indexed repositories.

    This tool performs semantic search against previously ingested Terraform code to find
    examples of how specific resources are used in real-world configurations.

    First ingest one or more repositories using the ingest_github_repo tool, then use this tool
    to search for specific Terraform resources (e.g., "aws_s3_bucket", "azurerm_virtual_network").

    Args:
        ctx: The MCP server context (automatically provided, no need to specify)
        resource_name: The Terraform resource type to find examples for (e.g., "aws_s3_bucket")
        n_results: Number of example files to return (default: 3)

    Returns:
        Example code snippets and configurations for the specified Terraform resource.
    """
    try:
        db_client = ctx.request_context.lifespan_context.db_client

        # Log the search query
        logger.info(
            f"Searching for Terraform resource: {resource_name}, requesting {n_results} results"
        )

        file_paths=[]
        while len(file_paths) < n_results:
            metadata_filter = Filter(
                must=[
                    FieldCondition(
                        key=PayloadField.CONTENT.field_name,
                        match=MatchText(text=resource_name),
                    ),
                ],
                must_not=[
                    FieldCondition(
                        key=PayloadField.FILE_PATH.field_name,
                        match=MatchText(value=file_path),
                    ) for file_path in exclude_file_paths
                ]
            )
            if file_path := db_client.search_vectors(metadata_filter=metadata_filter):
                file_paths.append(file_path)
            else:
                break

        # Check if we found any results
        if len(file_paths) == 0:
            logger.info(f"No examples found for resource: {resource_name}")
            return f"No examples found for Terraform resource '{resource_name}'. Try use another keyword."

        response = f"Key words found in following files: {','.join(file_paths)}\n"
        # Format the response
        for file_path in file_paths:
            response += f"### {file_path} content starts\n"
            path_to_file = os.path.join(GITHUB_DIR, file_path)
            with open(path_to_file, "r", encoding="utf-8") as f:
                response += f.read()
            response += f"### {file_path} content ends\n"

        return response
    except Exception as e:
        logger.error(f"Error searching for Terraform resource: {str(e)}", exc_info=True)
        return f"Error searching for Terraform resource: {str(e)}"


async def main():
    transport = os.getenv("TRANSPORT", "sse")

    if transport == "sse":
        print("🚀 MCP server starting using SSE transport")
        await mcp.run_sse_async()
    elif transport == "stdio":
        print("🚀 MCP server starting using STDIO transport")
        await mcp.run_stdio_async()
    else:
        print(f"❌ Error: TRANSPORT must be either 'sse' or 'stdio', got '{transport}'")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())

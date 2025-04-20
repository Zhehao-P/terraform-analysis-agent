"""
This module contains the main entry point for the MCP server.
It defines the MCP server and the tools it can use.
"""

import os
import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP, Context
from qdrant_client.models import Filter, FieldCondition, MatchText
from common.utils import (
    PayloadField,
    QdrantDB,
    setup_logging,
    get_embedding_function,
)

logger = setup_logging(__name__)

load_dotenv()

# Set up logging
logger = setup_logging()

# Base directory
BASE_DIR = Path(__file__).parent.parent
GITHUB_DIR = os.getenv("GITHUB_DIR", "data/github")
GITHUB_DIR = str(BASE_DIR / GITHUB_DIR)


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
    host=os.getenv("HOST") or "0.0.0.0",
    port=int(os.getenv("PORT") or "8000"),
)


@mcp.tool()
async def analyze_terraform_resource(
    ctx: Context,
    keyword: str,
    n_results: int = 3,
    exclude_file_paths: Optional[list[str]] = None,
    file_type: str | None = None,
) -> str:
    """Search for keyword occurrences in indexed Terraform files and documentation.

    This tool performs a keyword-based search against previously ingested code and
    documentation. It returns the entire content of files containing the keyword,
    with the keyword highlighted. The keyword can be either a Terraform resource
    name (e.g., "aviatrix_account") or an error message to find relevant examples
    and solutions.

    First ingest one or more repositories using the ingest_github_repo tool, then
    use this tool to search for specific keywords (e.g., "aviatrix_account",
    "Invalid provider configuration").

    Args:
        ctx: The MCP server context (automatically provided, no need to specify)
        keyword: The keyword to search for in files and documentation (can be a
                resource name or error message)
        n_results: Maximum number of files to return (default: 3)
        exclude_file_paths: List of file paths to exclude from search results
                          (default: None)
        file_type: Optional filter for file type ("code" or "documentation")
                 (default: None)

    Returns:
        A formatted string containing:
        - List of files where the keyword was found
        - Full content of each matching file with keyword highlighted
        - Error message if no results found or if an error occurred

    Raises:
        Exception: If there is an error during the search or file reading process
    """
    try:
        db_client = ctx.request_context.lifespan_context.db_client
        exclude_file_paths = exclude_file_paths or []

        # Log the search query
        logger.info(
            "Searching for keyword: %s, requesting %d results", keyword, n_results
        )

        # Build the filter conditions
        must_conditions = [
            FieldCondition(
                key=PayloadField.CONTENT.field_name,
                match=MatchText(text=keyword),
            ),
        ]

        # Add file type filter if specified
        if file_type:
            must_conditions.append(
                FieldCondition(
                    key=PayloadField.FILE_TYPE.field_name,
                    match=MatchText(text=file_type),
                )
            )

        file_paths = []
        while len(file_paths) < n_results:
            metadata_filter = Filter(
                must=must_conditions,
                must_not=[
                    FieldCondition(
                        key=PayloadField.FILE_PATH.field_name,
                        match=MatchText(value=file_path),
                    )
                    for file_path in exclude_file_paths
                ],
            )
            if file_path := db_client.search_vectors(metadata_filter=metadata_filter):
                file_paths.append(file_path)
            else:
                break

        # Check if we found any results
        if len(file_paths) == 0:
            logger.info("No files found containing keyword: %s", keyword)
            return f"No files found containing '{keyword}'. Try another keyword."

        response = f"Found {len(file_paths)} files containing '{keyword}':\n"
        # Format the response
        for file_path in file_paths:
            try:
                response += f"\n### File: {file_path}\n"
                path_to_file = os.path.join(GITHUB_DIR, file_path)
                with open(path_to_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Highlight the keyword in the content
                    highlighted_content = content.replace(keyword, f"**{keyword}**")
                    response += highlighted_content
                response += f"\n### End of {file_path}\n"
            except Exception as e:
                logger.error("Error reading file %s: %s", file_path, str(e))
                response += f"\nError reading file {file_path}: {str(e)}\n"

        return response
    except Exception as e:
        logger.error("Error searching for keyword: %s", str(e), exc_info=True)
        return f"Error searching for keyword: {str(e)}"


async def main():
    """
    Main entry point for the MCP server.
    """
    transport = os.getenv("TRANSPORT") or "sse"

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

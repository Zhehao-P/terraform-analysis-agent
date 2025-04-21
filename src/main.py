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
from common.qdrant_client import QdrantDB, PayloadField, FileType
from common.utils import logger, get_embedding_function

load_dotenv()

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
    db_client = QdrantDB(embed_fn=embed_function)

    try:
        yield RepoContext(db_client=db_client)
    finally:
        db_client.client.close()
        # No explicit persistence needed - PersistentClient handles this automatically
        logger.info("Shutting down qdrant client")


# Initialize FastMCP server
mcp = FastMCP(
    "terraform-agent",
    description="MCP server for repository content analysis using RAG",
    lifespan=mcp_lifespan,
    host=os.getenv("HOST") or "0.0.0.0",
    port=int(os.getenv("PORT") or "8000"),
)


@mcp.tool(name="get_src_file_by_name", description="Get all Terraform source files containing the input keywords.")
async def get_src_file_by_name(
    ctx: Context,
    keywords: str,
    n_results: int = 3,
    exclude_file_paths: Optional[list[str]] = None,
) -> str:
    """
    Get all Terraform source files containing the input keywords appear together.
    Try to use the exact resource name to search for Terraform source files.

    Args:
        keywords: keywords to search for in the Terraform source files.
        n_results: Optional maximum number of files to return.
        exclude_file_paths: Optional list of earlier response file paths to exclude

    Returns:
        Pain text with the resource name highlighted in the source file.
    """
    try:
        db_client = ctx.request_context.lifespan_context.db_client
        exclude_file_paths = exclude_file_paths or []

        # Log the search query
        logger.info(
            "Searching for Terraform resource: %s, requesting %d results", keywords, n_results
        )

        # Build the filter conditions for both resource definitions and usage
        must_conditions = [
            FieldCondition(
                key=PayloadField.CONTENT.field_name,
                match=MatchText(text=keywords),
            ),
            FieldCondition(
                key=PayloadField.FILE_TYPE.field_name,
                match=MatchText(text=FileType.CODE.value),
            )
        ]

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
            if file_path := db_client.query_vectors(metadata_filter=metadata_filter):
                file_paths.append(file_path)
            else:
                break

        # Check if we found any results
        if len(file_paths) == 0:
            logger.info("No files found containing resource: %s", keywords)
            return f"No Terraform files found containing '{keywords}'. Try another resource name."

        response = f"Found {len(file_paths)} files containing '{keywords}':\n"
        # Format the response
        for file_path in file_paths:
            try:
                response += f"\n### File: {file_path}\n"
                path_to_file = os.path.join(GITHUB_DIR, file_path)
                with open(path_to_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Highlight the resource name in the content
                    highlighted_content = content.replace(
                        f'resource "{keywords}"',
                        f'resource "**{keywords}**"'
                    ).replace(
                        f'"{keywords}."',
                        f'"**{keywords}**."'
                    ).replace(
                        f'"{keywords}"',
                        f'"**{keywords}**"'
                    )
                    response += highlighted_content
                response += f"\n### End of {file_path}\n"
            except Exception as e:
                logger.error("Error reading file %s: %s", file_path, str(e))
                response += f"\nError reading file {file_path}: {str(e)}\n"

        return response
    except Exception as e:
        logger.error("Error searching for resource: %s", str(e), exc_info=True)
        return f"Error searching for resource: {str(e)}"


async def main():
    """
    Main entry point for the MCP server.
    """
    print("📦 Registered tools:", [t.name for t in mcp._tool_manager.list_tools()])
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

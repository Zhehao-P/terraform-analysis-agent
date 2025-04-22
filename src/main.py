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
from qdrant_client.models import Filter, FieldCondition, MatchText, MatchValue
from common.qdrant_client import QdrantDB, PayloadField, FileType
from common.utils import logger, get_embedding_function

load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent
GITHUB_DIR = os.getenv("GITHUB_DIR", "data/github")
GITHUB_DIR = str(BASE_DIR / GITHUB_DIR)

DEFAULT_N_RESULTS = 5

SYSTEM_PROMPT = """
You are an intelligent MCP agent for analyzing Terraform repositories using Retrieval-Augmented Generation (RAG).

Your job is to:
- Search for relevant source code and documentation files.
- Filter out irrelevant files.
- Guide users to narrow down their problem scope.

TASK MANAGEMENT RULES:
1. When a new task begins (e.g. new Terraform command or error), you **must immediately call `reset_irrelevant_file_paths`**.
2. When you find any file that is **not useful to the current context**, call `update_irrelevant_file_paths` to reduce noise.
3. If you're unsure which files are already excluded, call `get_irrelevant_file_paths`.

SEARCH BEHAVIOR:
- Use `get_src_file_by_prompt` and `get_doc_file_by_prompt` for semantic search.
- Use `get_src_file_by_keywords` and `get_doc_file_by_keywords` for exact keyword matches.

IMPORTANT:
You must proactively manage the irrelevant files list. Do not wait for the user to tell you. Your effectiveness depends on it.
"""


@dataclass
class RepoContext:
    """Context for managing repository content storage and retrieval."""

    db_client: QdrantDB
    irrelevant_file_paths: list[str]


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
        yield RepoContext(db_client=db_client, irrelevant_file_paths=[])
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
    instructions=SYSTEM_PROMPT,
)


async def _search_files(
    ctx: Context,
    file_type: str,
    n_results: int,
    prompt: Optional[str] = None,
    keywords: Optional[list[str]] = None,
) -> list[str]:
    """
    Helper function to search for files with common search logic.

    Args:
        ctx: Context object
        file_type: Type of file to search for
        n_results: Maximum number of results to return
        prompt: Optional natural language prompt for semantic search
        keywords: Optional list of keywords for exact search

    Returns:
        List of file paths found

    Raises:
        ValueError: If neither prompt nor keywords are provided
    """
    if prompt is None and keywords is None:
        raise ValueError("Either prompt or keywords must be provided")
    if prompt is not None and keywords is not None:
        raise ValueError("Only one of prompt or keywords should be provided")

    db_client = ctx.request_context.lifespan_context.db_client
    # Use irrelevant_file_paths from RepoContext
    irrelevant_paths = ctx.request_context.lifespan_context.irrelevant_file_paths
    seen_paths = set(irrelevant_paths)

    # Log the search query
    search_term = prompt or ", ".join(keywords)
    logger.info(
        "Searching for %s files: %s, requesting %d results",
        file_type,
        search_term,
        n_results,
    )

    # Build the filter conditions
    must_conditions = [
        FieldCondition(
            key=PayloadField.FILE_TYPE.field_name,
            match=MatchText(text=file_type),
        )
    ]

    if keywords is not None:
        must_conditions.extend(
            [
                FieldCondition(
                    key=PayloadField.CONTENT.field_name,
                    match=MatchText(text=keyword),
                )
                for keyword in keywords
            ]
        )

    file_paths = []

    while len(file_paths) < n_results:
        metadata_filter = Filter(
            must=must_conditions,
            must_not=[
                FieldCondition(
                    key=PayloadField.FILE_PATH.field_name,
                    match=MatchValue(value=file_path),
                )
                for file_path in seen_paths
            ],
        )

        if prompt is not None:
            query_vector = await db_client.embed_fn(prompt)
            file_path = db_client.query_vectors(query_vector[0], metadata_filter)
        else:
            file_path = db_client.query_vectors(metadata_filter=metadata_filter)

        if file_path:
            file_paths.append(file_path)
            seen_paths.add(file_path)
        else:
            break

    return file_paths


def _format_response(
    file_paths: list[str],
    file_type: FileType,
    prompt: Optional[str] = None,
    keywords: Optional[list[str]] = None,
) -> str:
    """
    Helper function to format the response message.

    Args:
        file_paths: List of file paths found
        file_type: Type of files searched
        prompt: Optional natural language prompt used for search
        keywords: Optional list of keywords used for search

    Returns:
        Formatted response string

    Raises:
        ValueError: If neither prompt nor keywords are provided
    """
    if prompt is None and keywords is None:
        raise ValueError("Either prompt or keywords must be provided")
    if prompt is not None and keywords is not None:
        raise ValueError("Only one of prompt or keywords should be provided")

    search_term = prompt or ", ".join(keywords)
    search_type = "semantic" if prompt is not None else "exact"

    if len(file_paths) == 0:
        logger.info(
            "No %s matches found for %s: %s",
            search_type,
            file_type.value,
            search_term,
        )
        return (
            f"No {search_type} matches found for {file_type.value}: '{search_term}'. "
            f"Try a different {'description' if search_type == 'semantic' else 'keywords'}."
        )

    files_str = ", ".join(file_paths)
    response = (
        f"Found {search_type} matches for '{search_term}' in the following "
        f"{len(file_paths)} {file_type.value} files: [{files_str}]\n"
        f"Please add files that are not relevant to the current task to irrelevant_file_paths!\n"
    )
    logger.info("%s search response: %s", search_type.capitalize(), response)

    for file_path in file_paths:
        try:
            response += f"\n### File: {file_path}\n"
            path_to_file = os.path.join(GITHUB_DIR, file_path)
            with open(path_to_file, "r", encoding="utf-8") as f:
                content = f.read()
                response += content
            response += f"\n### End of {file_path}\n"
        except Exception as e:
            logger.error("Error reading file %s: %s", file_path, str(e))
            response += f"\nError reading file {file_path}: {str(e)}\n"

    return response


@mcp.tool(
    name="update_irrelevant_file_paths",
    description="Update the list of file paths to exclude from future searches.",
)
async def update_irrelevant_file_paths(
    ctx: Context,
    file_paths: list[str],
) -> str:
    """
    Update the list of file paths to exclude from future searches.
    This function adds the provided file paths to the list of irrelevant file paths.

    Args:
        file_paths: List of file paths that are irrelevant to the current task
    """
    ctx.request_context.lifespan_context.irrelevant_file_paths.extend(file_paths)
    return f"You are reducing the scope of the search. Well done! Updated irrelevant_file_paths with {len(file_paths)} new paths."


@mcp.tool(
    name="reset_irrelevant_file_paths",
    description="Reset the list of file paths to exclude from future searches.",
)
async def reset_irrelevant_file_paths(
    ctx: Context,
) -> str:
    """
    Reset the list of file paths to exclude from future searches.
    When a new task arrives, use this function to reset the list of irrelevant file paths.

    Args:
        None
    """
    ctx.request_context.lifespan_context.irrelevant_file_paths = []
    return (
        "Irrelevant file paths have been reset. All files will be included in searches."
    )


@mcp.tool(
    name="get_irrelevant_file_paths",
    description="Get the current list of file paths considered irrelevant to the task.",
)
async def get_irrelevant_file_paths(ctx: Context) -> list[str]:
    """
    Get the current list of file paths considered irrelevant to the current task.
    These files are excluded from searches.

    Returns:
        List of irrelevant file paths
    """
    return ctx.request_context.lifespan_context.irrelevant_file_paths


@mcp.tool(
    name="get_src_file_by_keywords",
    description="Get source files containing exact keyword matches.",
)
async def get_src_file_by_keywords(
    ctx: Context,
    keywords: list[str],
    n_results: int = DEFAULT_N_RESULTS,
) -> str:
    """
    Get source files containing exact matches of the input keywords.
    This function performs a literal text search for the keywords in the files.

    Args:
        keywords: List of exact keywords to search for in the source files.
        n_results: Optional maximum number of files to return.

    Returns:
        Plain text containing the matching files with their content.
    """
    try:
        file_paths = await _search_files(
            ctx,
            FileType.CODE.value,
            n_results,
            keywords=keywords,
        )
        return _format_response(file_paths, FileType.CODE, keywords=keywords)
    except Exception as e:
        logger.error("Error searching for resources: %s", str(e), exc_info=True)
        return f"Error searching for resources: {str(e)}"


@mcp.tool(
    name="get_doc_file_by_keywords",
    description="Get documentation files containing exact keyword matches.",
)
async def get_doc_file_by_keywords(
    ctx: Context,
    keywords: list[str],
    n_results: int = DEFAULT_N_RESULTS,
) -> str:
    """
    Get documentation files containing exact matches of the input keywords.
    This function performs a literal text search for the keywords in the files.

    Args:
        keywords: List of exact keywords to search for in the documentation files.
        n_results: Optional maximum number of files to return.

    Returns:
        Plain text containing the matching files with their content.
    """
    try:
        file_paths = await _search_files(
            ctx,
            FileType.DOCUMENT.value,
            n_results,
            keywords=keywords,
        )
        return _format_response(file_paths, FileType.DOCUMENT, keywords=keywords)
    except Exception as e:
        logger.error("Error searching for resources: %s", str(e), exc_info=True)
        return f"Error searching for resources: {str(e)}"


@mcp.tool(
    name="get_src_file_by_prompt",
    description="Get source files using semantic search based on the input prompt.",
)
async def get_src_file_by_prompt(
    ctx: Context,
    prompt: str,
    n_results: int = DEFAULT_N_RESULTS,
) -> str:
    """
    Get source files using semantic search based on the input prompt.
    This function uses embeddings to find files that are semantically related to the prompt,
    even if they don't contain exact keyword matches.

    Args:
        prompt: Natural language description of what you're looking for.
        n_results: Optional maximum number of files to return.

    Returns:
        Plain text containing the semantically matching files with their content.
    """
    try:
        file_paths = await _search_files(
            ctx,
            FileType.CODE.value,
            n_results,
            prompt=prompt,
        )
        return _format_response(file_paths, FileType.CODE, prompt=prompt)
    except Exception as e:
        logger.error("Error searching for resources: %s", str(e), exc_info=True)
        return f"Error searching for resources: {str(e)}"


@mcp.tool(
    name="get_doc_file_by_prompt",
    description="Get documentation files using semantic search based on the input prompt.",
)
async def get_doc_file_by_prompt(
    ctx: Context,
    prompt: str,
    n_results: int = DEFAULT_N_RESULTS,
) -> str:
    """
    Get documentation files using semantic search based on the input prompt.
    This function uses embeddings to find files that are semantically related to the prompt,
    even if they don't contain exact keyword matches.

    Args:
        prompt: Natural language description of what you're looking for.
        n_results: Optional maximum number of files to return.

    Returns:
        Plain text containing the semantically matching files with their content.
    """
    try:
        file_paths = await _search_files(
            ctx,
            FileType.DOCUMENT.value,
            n_results,
            prompt=prompt,
        )
        return _format_response(file_paths, FileType.DOCUMENT, prompt=prompt)
    except Exception as e:
        logger.error("Error searching for resources: %s", str(e), exc_info=True)
        return f"Error searching for resources: {str(e)}"


async def main():
    """
    Main entry point for the MCP server.
    """
    print("📦 Registered tools:", [t.name for t in mcp.list_tools()])
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

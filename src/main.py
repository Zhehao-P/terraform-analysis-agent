from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
import chromadb
from git import Repo
import os
import asyncio
from pathlib import Path
from config import (
    CHROMA_DB_DIR,
    GITHUB_DIR,
    PROCESS_FILE_EXTENSIONS,
    DEFAULT_HOST,
    DEFAULT_PORT,
    CHROMA_COLLECTION_NAME,
    CHROMA_COLLECTION_METADATA
)
from utils import get_chromadb_client

@dataclass
class RepoContext:
    """Context for managing repository content storage and retrieval."""
    chroma_client: chromadb.Client

@asynccontextmanager
async def mcp_lifespan(server: FastMCP) -> AsyncIterator[RepoContext]:
    """Manages the MCP agent lifecycle including ChromaDB persistence.

    This function:
    1. Uses the configured ChromaDB directory
    2. Initializes ChromaDB with persistent storage
    3. Creates or loads the collection
    4. Handles cleanup on shutdown
    """
    # Create and initialize the ChromaDB client with the helper function
    chroma_client = get_chromadb_client(
        CHROMA_DB_DIR,
        CHROMA_COLLECTION_NAME,
        CHROMA_COLLECTION_METADATA
    )

    try:
        yield RepoContext(chroma_client=chroma_client)
    finally:
        # No explicit persistence needed - PersistentClient handles this automatically
        pass

# Initialize FastMCP server
mcp = FastMCP(
    "repo-analysis-agent",
    description="MCP server for repository content analysis using RAG",
    lifespan=mcp_lifespan,
    host=os.getenv("HOST", DEFAULT_HOST),
    port=os.getenv("PORT", DEFAULT_PORT)
)

@mcp.tool()
async def ingest_github_repo(ctx: Context, repo_url: str) -> str:
    """Ingest a GitHub repository to build a searchable knowledge base.
    
    This tool clones a GitHub repository, processes the code files, and indexes them for semantic search.
    The indexed files can later be used to find relevant code snippets when debugging errors or answering questions.
    
    Args:
        ctx: The MCP server context (automatically provided, no need to specify)
        repo_url: Full URL of the GitHub repository to ingest (e.g., "https://github.com/username/repo")
        
    Returns:
        A success message with the number of files ingested, or an error message if the operation failed.
    """
    try:
        # Get the collection
        collection = ctx.request_context.lifespan_context.chroma_client.get_collection(
            name=CHROMA_COLLECTION_NAME
        )

        # Extract repo name from URL for cache directory
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        repo_cache_dir = GITHUB_DIR / repo_name

        # If repo is already cached, update it; otherwise clone
        if repo_cache_dir.exists():
            repo = Repo(str(repo_cache_dir))
            repo.remotes.origin.pull()  # Update the cached repo
        else:
            # Clone directly into the cache directory
            repo = Repo.clone_from(repo_url, str(repo_cache_dir))

        # Process repository files
        documents = []
        metadatas = []
        ids = []

        for root, _, files in os.walk(str(repo_cache_dir)):
            for file in files:
                if Path(file).suffix in PROCESS_FILE_EXTENSIONS:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        relative_path = file_path[len(str(repo_cache_dir)):]  # Relative path
                        documents.append(content)
                        metadatas.append({'path': relative_path})
                        ids.append(relative_path)

        # Store in ChromaDB in a single batch
        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

        return f"Successfully ingested {len(documents)} files from {repo_url}"
    except Exception as e:
        return f"Error ingesting repository: {str(e)}"

@mcp.tool()
async def analyze_error(ctx: Context, error_message: str) -> str:
    """Analyze an error message and find relevant code snippets from ingested repositories.
    
    This tool performs semantic search against previously ingested code repositories to find
    code examples or documentation related to the error or query you provided.
    
    First ingest one or more repositories using the ingest_github_repo tool, then use this tool
    to analyze specific error messages or queries.
    
    Args:
        ctx: The MCP server context (automatically provided, no need to specify)
        error_message: The error message or query to analyze
        
    Returns:
        Relevant code snippets and their file paths from the ingested repositories.
    """
    try:
        # Get the collection
        collection = ctx.request_context.lifespan_context.chroma_client.get_collection(
            name=CHROMA_COLLECTION_NAME
        )

        # Search for relevant documentation
        results = collection.query(
            query_texts=[error_message],
            n_results=3
        )

        # Format the response
        response = "Found relevant documentation:\n\n"
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
            response += f"{i}. File: {metadata['path']}\n"
            response += f"Content:\n{doc}\n\n"

        return response
    except Exception as e:
        return f"Error analyzing error: {str(e)}"

async def main():
    transport = os.getenv("TRANSPORT")
    if not transport:
        print("❌ Error: TRANSPORT environment variable must be set to either 'sse' or 'stdio'")
        print("   Please set it in your .env file or as an environment variable")
        exit(1)
        
    if transport == 'sse':
        print(f"🚀 MCP server starting at http://{DEFAULT_HOST}:{DEFAULT_PORT} using SSE transport")
        await mcp.run_sse_async()
    elif transport == 'stdio':
        print(f"🚀 MCP server starting using STDIO transport")
        await mcp.run_stdio_async()
    else:
        print(f"❌ Error: TRANSPORT must be either 'sse' or 'stdio', got '{transport}'")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())

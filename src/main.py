from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
import chromadb
from git import Repo
import os
import asyncio
import datetime
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
from utils import get_chromadb_client, setup_logging

# Set up logging
logger = setup_logging()

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
    logger.info("Initializing ChromaDB client")
    chroma_client = get_chromadb_client(
        CHROMA_DB_DIR,
        CHROMA_COLLECTION_NAME,
        CHROMA_COLLECTION_METADATA
    )

    try:
        yield RepoContext(chroma_client=chroma_client)
    finally:
        # No explicit persistence needed - PersistentClient handles this automatically
        logger.info("Shutting down ChromaDB client")
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
async def ingest_github_repo(ctx: Context) -> str:
    """Build a searchable knowledge base from local repositories.
    
    This tool scans all local repositories in the data/github directory and indexes their Terraform and Go files.
    It builds a local vector database that enables semantic search for code snippets when debugging errors.
    The tool automatically handles incremental updates - only new or modified files will be processed.
    
    No parameters needed - all repositories in the data/github directory will be automatically processed.
    
    Args:
        ctx: The MCP server context (automatically provided, no need to specify)
        
    Returns:
        A summary of the ingestion process, including counts of indexed, updated, and skipped files.
    """
    start_time = datetime.datetime.now()
    logger.info(f"Starting knowledge base ingestion at {start_time}")
    
    try:
        # Get the collection
        collection = ctx.request_context.lifespan_context.chroma_client.get_collection(
            name=CHROMA_COLLECTION_NAME
        )
        
        # Process all repositories in the github directory
        total_documents = 0
        total_skipped = 0
        total_updated = 0
        processed_repos = 0
        test_skipped = 0
        
        # Get all directories in the github directory
        repo_dirs = [d for d in GITHUB_DIR.iterdir() if d.is_dir()]
        
        if not repo_dirs:
            logger.warning(f"No local repositories found in {GITHUB_DIR}")
            return f"No local repositories found in {GITHUB_DIR}. Please add repositories to this directory."
        
        logger.info(f"Found {len(repo_dirs)} repositories to process in {GITHUB_DIR}")
        
        # Process each repository
        for repo_dir in repo_dirs:
            repo_name = repo_dir.name
            repo_start_time = datetime.datetime.now()
            logger.info(f"Processing repository: {repo_name}, started at {repo_start_time}")
            
            documents = []
            metadatas = []
            ids = []
            skipped = 0
            updated = 0
            repo_test_skipped = 0

            for root, _, files in os.walk(str(repo_dir)):
                for file in files:
                    if Path(file).suffix in PROCESS_FILE_EXTENSIONS:
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, str(repo_dir))
                        
                        # Skip test files
                        if "test" in relative_path.lower():
                            repo_test_skipped += 1
                            logger.debug(f"Skipping test file: {relative_path}")
                            continue
                        
                        # Create a unique ID that includes the repo name to avoid conflicts
                        file_id = f"{repo_name}/{relative_path}"
                        
                        # Get file last modification time
                        last_modified = str(os.path.getmtime(file_path))
                        
                        # Check if file already exists in collection by querying for this specific ID
                        existing_item = collection.get(ids=[file_id])
                        
                        # If the file exists in the database (ids list not empty)
                        if existing_item['ids']:
                            # Get metadata if available
                            if existing_item['metadatas'] and existing_item['metadatas'][0]:
                                existing_last_modified = existing_item['metadatas'][0].get('last_modified', '')
                                
                                # Compare timestamps
                                if existing_last_modified == last_modified:
                                    # File exists and hasn't changed, skip it
                                    skipped += 1
                                    logger.debug(f"Skipping unchanged file: {file_id}")
                                    continue
                            
                            # File exists but has changed (or no timestamp), update it
                            updated += 1
                            logger.debug(f"Updating modified file: {file_id}")
                            # Delete the old version
                            collection.delete(ids=[file_id])
                        else:
                            logger.debug(f"Processing new file: {file_id}")
                        
                        # Process the file
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            documents.append(content)
                            metadatas.append({
                                'repo': repo_name,
                                'path': relative_path,
                                'full_path': file_id,
                                'last_modified': last_modified
                            })
                            ids.append(file_id)

            # Store in ChromaDB in a single batch
            if documents:
                logger.info(f"Adding {len(documents)} documents to collection from repository {repo_name}")
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                total_documents += len(documents)
                total_skipped += skipped
                total_updated += updated
                test_skipped += repo_test_skipped
                processed_repos += 1
                
                repo_end_time = datetime.datetime.now()
                duration = (repo_end_time - repo_start_time).total_seconds()
                logger.info(f"Repository '{repo_name}' processed in {duration:.2f} seconds: {len(documents)} files added, {updated} updated, {skipped} skipped, {repo_test_skipped} test files skipped")
            else:
                logger.info(f"No documents to add from repository {repo_name} ({repo_test_skipped} test files skipped)")

        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if total_documents > 0:
            result_msg = f"Local knowledge base updated: {processed_repos} repositories processed, {total_documents} total files indexed, {total_updated} updated, {total_skipped} skipped (unchanged), {test_skipped} test files skipped. Completed in {duration:.2f} seconds."
            logger.info(result_msg)
            return result_msg
        else:
            result_msg = f"No new or modified files found in any local repository. Supported extensions: {', '.join(PROCESS_FILE_EXTENSIONS)}. {test_skipped} test files were skipped. Process completed in {duration:.2f} seconds."
            logger.info(result_msg)
            return result_msg
    except Exception as e:
        logger.error(f"Error building knowledge base: {str(e)}", exc_info=True)
        return f"Error building knowledge base: {str(e)}"

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
        print(f"🚀 MCP server starting using SSE transport")
        await mcp.run_sse_async()
    elif transport == 'stdio':
        print(f"🚀 MCP server starting using STDIO transport")
        await mcp.run_stdio_async()
    else:
        print(f"❌ Error: TRANSPORT must be either 'sse' or 'stdio', got '{transport}'")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())

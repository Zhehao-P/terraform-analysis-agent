"""
Process data from GitHub repositories and upload to Qdrant.

This module provides functionality to process files from GitHub repositories,
create embeddings, and store them in a Qdrant vector database. It supports
concurrent processing with configurable limits and comprehensive error handling.
"""

import asyncio
import datetime
from enum import Enum
import os
from pathlib import Path
import uuid
from dotenv import load_dotenv
from tqdm import tqdm
from qdrant_client.models import Filter, FieldCondition, MatchValue
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

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data directories
GITHUB_DIR = os.getenv("GITHUB_DIR", "data/github")

# Processing configuration
MAX_CHARS = 8000
OVERLAP = 1000
CHUNK_BATCH_SIZE = 50
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "10"))


class FileTypeMappedExtension(Enum):
    """
    Enum for file types mapped to their extensions.
    Used to determine the type of file based on its extension.
    """

    CODE = {".tf", ".go"}
    DOCUMENT = {".md"}


def get_file_type(file_relative_path: Path) -> FileType:
    """
    Determine the type of file based on its extension and name.

    Args:
        file_relative_path: Path to the file

    Returns:
        FileType enum value indicating the type of file
    """
    match file_relative_path.suffix:
        case suffix if suffix in FileTypeMappedExtension.CODE.value:
            return FileType.CODE
        case suffix if suffix in FileTypeMappedExtension.DOCUMENT.value:
            return FileType.DOCUMENT
        case _ if "test" in file_relative_path.name.lower():
            return FileType.TEST
        case _:
            return FileType.NOT_SUPPORTED


async def process_file(
    file_path: str, repo_dir: Path, qdrant_db: QdrantDB
) -> list[str]:
    """
    Process a single file and upload to Qdrant asynchronously.
    Processes file in chunks of maximum 50 at a time to manage memory and API limits.

    Args:
        file_path: Full path to the file
        repo_dir: Path to the repository directory
        qdrant_db: Qdrant database instance

    Returns:
        List of IDs for the upsert points

    Raises:
        Exception: If file processing or upload fails
    """
    repo_name = repo_dir.name
    relative_path = os.path.join(repo_name, os.path.relpath(file_path, str(repo_dir)))
    file_type = get_file_type(Path(relative_path))
    last_modified = datetime.datetime.fromtimestamp(
        os.path.getmtime(file_path), tz=datetime.timezone.utc
    ).isoformat()

    metadata_filter = Filter(
        must=[
            FieldCondition(
                key=PayloadField.FILE_PATH.field_name,
                match=MatchValue(value=relative_path),
            ),
            FieldCondition(
                key=PayloadField.LAST_MODIFIED.field_name,
                match=MatchValue(value=last_modified),
            ),
        ]
    )

    if file_type == FileType.NOT_SUPPORTED:
        logger.debug("Skipping unknown file type: %s", relative_path)
        return []

    if qdrant_db.check_metadata_exists(metadata_filter):
        logger.debug("Skipping existing file: %s", relative_path)
        return []

    qdrant_db.delete_vectors_by_filter(metadata_filter)

    logger.info("Processing file: %s", relative_path)

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        start = 0
        chunks = []
        while start < len(text):
            end = start + MAX_CHARS
            chunks.append(text[start:end])
            start += MAX_CHARS - OVERLAP

    all_points = []
    for i in range(0, len(chunks), CHUNK_BATCH_SIZE):
        batch_chunks = chunks[i : i + CHUNK_BATCH_SIZE]
        logger.debug("Processing batch of %d chunks", len(batch_chunks))

        embedded_chunks = await qdrant_db.embed_fn(batch_chunks)
        batch_points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedded_chunk,
                payload={
                    PayloadField.FILE_TYPE.field_name: file_type.value,
                    PayloadField.FILE_PATH.field_name: relative_path,
                    PayloadField.LAST_MODIFIED.field_name: last_modified,
                    PayloadField.REPO.field_name: repo_name,
                    PayloadField.CONTENT.field_name: chunk,
                },
            )
            for chunk, embedded_chunk in zip(batch_chunks, embedded_chunks)
        ]
        all_points.extend(batch_points)

    logger.debug("Upsert points: %s", all_points)
    return qdrant_db.upsert_vectors(points=all_points)


async def process_data(qdrant_db: QdrantDB) -> str:
    """
    Process all files in the configured GitHub directory asynchronously.

    This function processes files concurrently with a configurable limit on
    the number of simultaneous tasks. It handles errors gracefully and provides
    detailed progress information.

    Args:
        qdrant_db: Qdrant database instance

    Returns:
        Status message indicating success or failure, including error count if any

    Configuration:
        MAX_CONCURRENT_TASKS: Environment variable to control concurrent processing
                             (default: 10)
        GITHUB_DIR: Directory containing GitHub repositories to process
    """
    repo_dirs = [d for d in Path(GITHUB_DIR).iterdir() if d.is_dir()]

    if not repo_dirs:
        logger.warning("No local repositories found in %s", GITHUB_DIR)
        return f"No local repositories found in {GITHUB_DIR}."

    logger.info("Found %d repositories to process in %s", len(repo_dirs), GITHUB_DIR)

    # Limit concurrent tasks to prevent resource exhaustion
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    logger.info("Limiting concurrent tasks to %d", MAX_CONCURRENT_TASKS)

    tasks: list[asyncio.Task] = []
    progress_bar = tqdm(total=0, desc="Processing files")
    for repo_dir in repo_dirs:
        logger.info("Processing repository: %s", repo_dir.name)

        for root, _, files in os.walk(str(repo_dir)):
            for file in files:
                file_path = os.path.join(root, file)

                async def wrapped_task(path=file_path, repo=repo_dir):
                    async with semaphore:  # Acquire semaphore before processing
                        try:
                            response = await process_file(path, repo, qdrant_db)
                            logger.debug("Upsert response: %s", response)
                        except Exception as e:
                            logger.error("Error processing file %s: %s", path, str(e))
                            raise
                        finally:
                            progress_bar.update(1)

                tasks.append(asyncio.create_task(wrapped_task()))
    progress_bar.total = len(tasks)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    progress_bar.close()

    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        error_msg = f"Processing completed with {len(errors)} errors"
        logger.error(error_msg)
        return error_msg

    return "Processing completed successfully"


def main():
    """
    Main entry point for the script.
    Sets up the embedding function and Qdrant database, then processes all files.
    """
    embed_function = get_embedding_function()
    qdrant_db = QdrantDB(embed_fn=embed_function)
    asyncio.run(process_data(qdrant_db))
    qdrant_db.build_payload_index()


if __name__ == "__main__":
    main()

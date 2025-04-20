import datetime
from enum import Enum
import os
from pathlib import Path
import uuid
from dotenv import load_dotenv
from common.utils import (
    PayloadField,
    QdrantDB,
    FileType,
    PointStruct,
    setup_logging,
    get_embedding_function,
)
import asyncio
from typing import List

logger = setup_logging(__name__)

load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data directories
GITHUB_DIR = os.getenv("GITHUB_DIR", "data/github")

MAX_CHARS = 8000
OVERLAP = 1000


class FileTypeMappedExtension(Enum):
    CODE = {".tf", ".go"}
    DOCUMENT = {".md"}


def get_file_type(file_relative_path: Path) -> FileType:
    """
    Determine the type of file based on its extension and name.

    Args:
        file_relative_path: Path to the file

    Returns:
        FileType enum value
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


async def process_file(file_path: str, repo_dir: Path, qdrant_db: QdrantDB) -> None:
    """Process a single file and upload to Qdrant."""
    repo_name = repo_dir.name
    relative_path = os.path.relpath(file_path, str(repo_dir))
    relative_path = os.path.join(repo_name, relative_path)
    file_type = get_file_type(Path(relative_path))

    if file_type == FileType.NOT_SUPPORTED:
        logger.debug(f"Skipping unknown file type: {relative_path}")
        return

    last_modified = str(os.path.getmtime(file_path))

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        start = 0
        chunks = []
        while start < len(text):
            end = start + MAX_CHARS
            chunks.append(text[start:end])
            start += MAX_CHARS - OVERLAP

    embedded_chunks = qdrant_db.embed_fn(chunks)
    points = [
        PointStruct(
            id=repo_name + "_" + str(uuid.uuid4()),
            vector=embedded_chunk,
            payload={
                PayloadField.FILE_TYPE.field_name: file_type.value,
                PayloadField.FILE_PATH.field_name: relative_path,
                PayloadField.LAST_MODIFIED.field_name: last_modified,
                PayloadField.REPO.field_name: repo_name,
                PayloadField.CONTENT.field_name: chunk,
            },
        )
        for chunk, embedded_chunk in zip(chunks, embedded_chunks)
    ]
    await qdrant_db.upsert_vectors(points=points)


async def process_data(qdrant_db: QdrantDB):
    repo_dirs = [d for d in Path(GITHUB_DIR).iterdir() if d.is_dir()]

    if not repo_dirs:
        logger.warning(f"No local repositories found in {GITHUB_DIR}")
        return f"No local repositories found in {GITHUB_DIR}. Please add repositories to this directory."

    logger.info(f"Found {len(repo_dirs)} repositories to process in {GITHUB_DIR}")

    tasks: List[asyncio.Task] = []
    for repo_dir in repo_dirs:
        logger.info(f"Processing repository: {repo_dir.name}")

        for root, _, files in os.walk(str(repo_dir)):
            for file in files:
                file_path = os.path.join(root, file)
                task = asyncio.create_task(process_file(file_path, repo_dir, qdrant_db))
                tasks.append(task)

    await asyncio.gather(*tasks, return_exceptions=True)


def main():
    embed_function = get_embedding_function()
    qdrant_db = QdrantDB(embed_fn=embed_function)
    asyncio.run(process_data(qdrant_db))


if __name__ == "__main__":
    main()

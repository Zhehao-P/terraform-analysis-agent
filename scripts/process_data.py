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

logger = setup_logging(__name__)

load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data directories
GITHUB_DIR = os.getenv("GITHUB_DIR", "data/github")

MAX_CHARS = 1000
OVERLAP = 100


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


def process_data(qdrant_db: QdrantDB):
    repo_dirs = [d for d in Path(GITHUB_DIR).iterdir() if d.is_dir()]

    if not repo_dirs:
        logger.warning(f"No local repositories found in {GITHUB_DIR}")
        return f"No local repositories found in {GITHUB_DIR}. Please add repositories to this directory."

    logger.info(f"Found {len(repo_dirs)} repositories to process in {GITHUB_DIR}")

    # Process each repository
    for repo_dir in repo_dirs:
        repo_name = repo_dir.name
        repo_start_time = datetime.datetime.now()
        logger.info(f"Processing repository: {repo_name}, started at {repo_start_time}")

        for root, _, files in os.walk(str(repo_dir)):
            for file in files:
                file_path = os.path.join(root, file)
                # Get the relative path from repo_dir
                relative_path = os.path.relpath(file_path, str(repo_dir))
                # Prepend the repository name to the path
                relative_path = os.path.join(repo_name, relative_path)
                file_type = get_file_type(Path(relative_path))

                if file_type == FileType.NOT_SUPPORTED:
                    logger.debug(f"Skipping unknown file type: {relative_path}")
                    continue

                # Get file last modification time
                last_modified = str(os.path.getmtime(file_path))

                # Process the file
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    start = 0
                    chunks = []
                    while start < len(text):
                        end = start + MAX_CHARS
                        chunks.append(text[start:end])
                        start += MAX_CHARS - OVERLAP

                # Debug log all values
                logger.info("File type: %s", file_type.value)
                logger.info("Relative path: %s", relative_path)
                logger.info("Last modified: %s", last_modified)
                logger.info("Repo name: %s", repo_name)
                logger.info("Content chunk: %s", chunks[0][:50] + "..." if len(chunks[0]) > 50 else chunks[0])

                points = [
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=[0] * 1024,
                        payload={
                            PayloadField.FILE_TYPE.name.lower(): file_type.value,
                            PayloadField.FILE_PATH.name.lower(): relative_path,
                            PayloadField.LAST_MODIFIED.name.lower(): last_modified,
                            PayloadField.REPO.name.lower(): repo_name,
                            PayloadField.CONTENT.name.lower(): chunk,
                        },
                    )
                    for chunk in chunks
                ]

                # Log the first point's payload for verification
                if points:
                    logger.info("Storing point with payload: %s", points[0].payload)

                qdrant_db.upsert_vectors(points=points)

    import pdb

    pdb.set_trace()


def main():
    embed_function = get_embedding_function()
    qdrant_db = QdrantDB(embed_fn=embed_function)
    process_data(qdrant_db)


if __name__ == "__main__":
    main()

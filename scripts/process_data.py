import datetime
from enum import Enum
import os
import tiktoken
from pathlib import Path
from dotenv import load_dotenv
from common.utils import QdrantDB, FileType, setup_logging, get_embedding_function

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


def get_file_type(file_relative_path: Path) -> str:
    match file_relative_path.suffix:
        case _ if file_relative_path.suffix in FileTypeMappedExtension.CODE.value:
            return FileType.CODE
        case _ if file_relative_path.suffix in FileTypeMappedExtension.DOCUMENT.value:
            return FileType.DOCUMENT
        case _ if "test" in file_relative_path.lower():
            return FileType.TEST
        case _:
            return FileType.NOT_SUPPORTED


def process_data(qdrant_db: QdrantDB):
    try:
        # Process all repositories in the github directory
        total_documents = 0
        total_skipped = 0
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
            logger.info(
                f"Processing repository: {repo_name}, started at {repo_start_time}"
            )

            skipped = 0
            updated = 0

            for root, _, files in os.walk(str(repo_dir)):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, str(repo_dir))
                    file_type = get_file_type(relative_path)

                    if file_type == FileType.NOT_SUPPORTED:
                        logger.debug(f"Skipping unknown file type: {relative_path}")
                        continue

                    # Get file last modification time
                    last_modified = str(os.path.getmtime(file_path))

                    # If the file exists in the database (ids list not empty)
                    if existing_item["ids"]:
                        # Get metadata if available
                        if existing_item["metadatas"] and existing_item["metadatas"][0]:
                            existing_last_modified = existing_item["metadatas"][0].get(
                                "last_modified", ""
                            )

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
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        start = 0
                        chunks = []
                        while start < len(text):
                            end = start + MAX_CHARS
                            chunks.append(text[start:end])
                            start += MAX_CHARS - OVERLAP

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


def main():
    embed_function = get_embedding_function()
    qdrant_db = QdrantDB(embed_fn=embed_function)
    process_data(qdrant_db)


if __name__ == "__main__":
    main()

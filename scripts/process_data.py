
import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
from utils import utils

logger = utils.setup_logging(__name__)

load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
GITHUB_DIR = DATA_DIR / os.getenv("GITHUB_DIR_NAME", "github")

PROCESS_FILE_EXTENSIONS = {'.tf', '.go'}

def process_data():
    try:

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

def main():
    qdrant_db = utils.QdrantDB()
    process_data(qdrant_db)

if __name__ == "__main__":
    main()

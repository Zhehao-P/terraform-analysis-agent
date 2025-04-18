from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Data subdirectories
CHROMA_DB_DIR = DATA_DIR / "chromadb"
CACHE_DIR = DATA_DIR / "cache"
GITHUB_DIR = DATA_DIR / "github"

# File processing
PROCESS_FILE_EXTENSIONS = {'.tf', '.go'}

# Server configuration
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = "8050"

# ChromaDB configuration
CHROMA_COLLECTION_NAME = "repo_files"
CHROMA_COLLECTION_METADATA = {"description": "Repository files and documentation"}

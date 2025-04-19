from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data directories
GITHUB_DIR = os.getenv("GITHUB_DIR", "data/github")

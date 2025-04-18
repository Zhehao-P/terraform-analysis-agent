import os
import chromadb
import logging

def setup_logging(module_name="terraform-analysis"):
    """
    Set up basic logging configuration.
    
    Args:
        module_name: Name for the logger
        
    Returns:
        A configured logger instance
    """
    # Check if DEBUG environment variable is set
    debug_mode = os.getenv('DEBUG') is not None
    log_level = logging.DEBUG if debug_mode else logging.INFO
    
    # Configure logging
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Get the logger and ensure it has the right level
    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)
    return logger

# Get logger
logger = setup_logging()

CUSTOM_INSTRUCTIONS = """
Extract the Following Information:

- Key Information: Identify and save the most important details.
- Context: Capture the surrounding context to understand the memory's relevance.
- Connections: Note any relationships to other topics or memories.
- Importance: Highlight why this information might be valuable in the future.
- Source: Record where this information came from when applicable.
"""

def get_chromadb_client(db_path, collection_name, collection_metadata=None):
    """
    Creates and returns a ChromaDB client with the configured embedding function.

    Args:
        db_path: Path to the ChromaDB database
        collection_name: Name of the collection to use
        collection_metadata: Optional metadata for the collection

    Returns:
        chromadb.Client: The configured ChromaDB client

    Raises:
        ValueError: If required API configuration is missing
    """
    logger.info(f"Initializing ChromaDB client with path: {db_path}")
    
    # Get API configuration
    api_key = os.getenv('LLM_API_KEY')
    api_base = os.getenv('LLM_BASE_URL')
    embedding_model = os.getenv('EMBEDDING_MODEL_CHOICE')

    # Validate required configuration
    if not api_key:
        logger.error("LLM_API_KEY environment variable not found")
        raise ValueError("LLM_API_KEY environment variable must be set")

    if not api_base:
        logger.error("LLM_BASE_URL environment variable not found")
        raise ValueError("LLM_BASE_URL environment variable must be set")

    if not embedding_model:
        logger.error("EMBEDDING_MODEL_CHOICE environment variable not found")
        raise ValueError("EMBEDDING_MODEL_CHOICE environment variable must be set")

    logger.info(f"Using embedding model: {embedding_model}")
    
    # Create OpenAI embedding function
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    embedding_func = OpenAIEmbeddingFunction(
        api_key=api_key,
        api_base=api_base,
        model_name=embedding_model
    )
    logger.debug("OpenAI embedding function created")

    # Initialize ChromaDB with persistent storage
    logger.info(f"Creating PersistentClient at: {db_path}")
    chroma_client = chromadb.PersistentClient(path=str(db_path))

    # Validate collection - make sure it can be created or accessed
    metadata = collection_metadata or {}
    try:
        logger.info(f"Getting or creating collection: {collection_name}")
        chroma_client.get_or_create_collection(
            name=collection_name,
            metadata=metadata,
            embedding_function=embedding_func
        )
        logger.info(f"Successfully connected to collection: {collection_name}")
    except Exception as e:
        error_msg = f"Failed to create or access collection '{collection_name}': {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Return only the client
    return chroma_client

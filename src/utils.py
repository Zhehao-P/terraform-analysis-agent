import os
import chromadb

# Custom instructions for memory processing
# These aren't being used right now but Mem0 does support adding custom prompting
# for handling memory retrieval and processing.
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
    # Get API configuration
    api_key = os.getenv('LLM_API_KEY')
    api_base = os.getenv('LLM_BASE_URL')
    embedding_model = os.getenv('EMBEDDING_MODEL_CHOICE')

    # Validate required configuration
    if not api_key:
        raise ValueError("LLM_API_KEY environment variable must be set")

    if not api_base:
        raise ValueError("LLM_BASE_URL environment variable must be set")

    if not embedding_model:
        raise ValueError("EMBEDDING_MODEL_CHOICE environment variable must be set")

    # Create OpenAI embedding function
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    embedding_func = OpenAIEmbeddingFunction(
        api_key=api_key,
        api_base=api_base,
        model_name=embedding_model
    )

    # Initialize ChromaDB with persistent storage
    chroma_client = chromadb.PersistentClient(path=str(db_path))

    # Validate collection - make sure it can be created or accessed
    metadata = collection_metadata or {}
    try:
        chroma_client.get_or_create_collection(
            name=collection_name,
            metadata=metadata,
            embedding_function=embedding_func
        )
    except Exception as e:
        raise ValueError(f"Failed to create or access collection '{collection_name}': {str(e)}")

    # Return only the client
    return chroma_client

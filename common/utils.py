import os
import logging
from enum import Enum
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)


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

# Get logger with default module name
logger = setup_logging()

CUSTOM_INSTRUCTIONS = """
Extract the Following Information:

- Key Information: Identify and save the most important details.
- Context: Capture the surrounding context to understand the memory's relevance.
- Connections: Note any relationships to other topics or memories.
- Importance: Highlight why this information might be valuable in the future.
- Source: Record where this information came from when applicable.
"""

def get_embedding_function():
    """
    Creates and returns an embedding function using OpenAI's API.

    Returns:
        A function that takes text and returns embeddings

    Raises:
        ValueError: If required API configuration is missing
    """
    # Get API configuration
    api_key = os.getenv('LLM_API_KEY')
    api_base = os.getenv('LLM_BASE_URL')
    embedding_model = os.getenv('EMBEDDING_MODEL_CHOICE')
    dimensions = os.getenv('EMBEDDING_DIMENSIONS', 1024)

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

    # Create OpenAI client
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url=api_base
    )

    def embed_batch(texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(
            texts,
            embedding_model,
            dimensions,
            "float"
        )
        return [item.embedding for item in response.data]

    return embed_batch

class FilterType(Enum):
    MUST = "must"
    SHOULD = "should"
    MUST_NOT = "must_not"

class QdrantDB:
    def __init__(
        self,
        host: str = os.getenv("QDRANT_HOST", "localhost"),
        port: int = os.getenv("QDRANT_PORT", 6333),
        collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "knowledge_db"),
        embed_fn: callable = None
    ):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.embed_fn = embed_fn

    def ensure_collection(self, vector_size: int = os.getenv("EMBEDDING_DIMENSIONS", 1024)):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

        for field in ["file_type", "file_role", "file_name", "repo"]:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema="keyword"
            )

    def upsert_vectors(self, points: list[PointStruct]):
        """
        Upsert vectors to the collection.

        Args:
            points: List of PointStruct objects containing vector data

        Returns:
            List of IDs for the upsert points (including auto-generated ones)
        """
        result = self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return result.ids

    def search_vectors(self, query_vector: list[float], limit: int = 10, filters: dict | None = None):
        filter_conditions = None
        if filters:
            filter_conditions = Filter(
                must=[
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filters.items()
                ]
            )

        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_conditions
        )

    def delete_vectors(self, ids: list[str]):
        self.client.delete(collection_name=self.collection_name, points_selector={"points": ids})

    def delete_vectors_by_filter(self, filters: dict[FilterType, dict[str, str | list[str]]]):
        filter_payload = {}

        for filter_type, condition_dict in filters.items():
            field_conditions = []
            for key, value in condition_dict.items():
                if isinstance(value, list):
                    field_conditions.extend([
                        FieldCondition(key=key, match=MatchValue(value=item)) for item in value
                    ])
                else:
                    field_conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            filter_payload[filter_type.value] = field_conditions

        filter_obj = Filter(**filter_payload)

        self.client.delete(
            collection_name=self.collection_name,
            filter=filter_obj
        )

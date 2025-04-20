"""
Utility functions and classes for the Terraform analysis agent.

This module provides common functionality for logging, embeddings, and Qdrant database operations.
"""

import os
import logging
from enum import Enum
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
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
    debug_mode = os.getenv("DEBUG") is not None
    log_level = logging.DEBUG if debug_mode else logging.INFO

    # Configure logging
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    # Get the logger and ensure it has the right level
    logger_obj = logging.getLogger(module_name)
    logger_obj.setLevel(log_level)
    return logger_obj


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
    Create and return an embedding function using OpenAI's API.

    Returns:
        A function that takes text and returns embeddings

    Raises:
        ValueError: If required API configuration is missing
    """
    # Get API configuration
    api_key = os.getenv("LLM_API_KEY")
    api_base = os.getenv("LLM_BASE_URL")
    embedding_model = os.getenv("EMBEDDING_MODEL_CHOICE")
    dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

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

    logger.info("Using embedding model: %s", embedding_model)

    # Create OpenAI client
    client = OpenAI(api_key=api_key, base_url=api_base)

    def embed_batch(texts: list[str]) -> list[list[float]]:
        """
        Create embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        response = client.embeddings.create(
            input=texts,
            model=embedding_model,
            dimensions=dimensions,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]

    return embed_batch


class FilterType(Enum):
    """
    Enum for different types of filter conditions in Qdrant.
    """

    MUST = "must"
    SHOULD = "should"
    MUST_NOT = "must_not"


class FileType(Enum):
    """
    Enum for different types of files in the system.
    """

    CODE = "code"
    DOCUMENT = "document"
    TEST = "test"
    NOT_SUPPORTED = "not_supported"


class PayloadField(Enum):
    """
    Enum for payload fields and their schema types.
    """

    FILE_TYPE = PayloadSchemaType.KEYWORD
    FILE_PATH = PayloadSchemaType.KEYWORD
    REPO = PayloadSchemaType.KEYWORD
    LAST_MODIFIED = PayloadSchemaType.DATETIME


class QdrantDB:
    """
    Client for interacting with Qdrant vector database.

    This class provides methods for managing collections, indexes, and vector operations
    in a Qdrant database. It handles vector storage, search, and filtering operations.
    """
    def __init__(
        self,
        host: str = os.getenv("QDRANT_HOST", "localhost"),
        port: int = int(os.getenv("QDRANT_PORT", "6333")),
        collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "knowledge_db"),
        embed_fn: callable = None,
    ):
        """
        Initialize QdrantDB client with configuration.

        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to use
            embed_fn: Optional embedding function
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.embed_fn = embed_fn
        self.collection = self._ensure_collection()

    def _ensure_collection(
        self, vector_size: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
    ):
        """
        Ensure the collection exists with correct configuration.
        Only creates the collection if it doesn't exist.

        Args:
            vector_size: Size of the vector embeddings
        """
        # Check if collection exists using built-in method
        if not self.client.collection_exists(self.collection_name):
            # Create collection if it doesn't exist
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info("Created collection %s", self.collection_name)
        else:
            logger.info("Collection %s already exists", self.collection_name)

        return self.client.get_collection(self.collection_name)

    def build_payload_index(self):
        """
        Create payload indexes for the collection's fields.
        """
        for field in PayloadField:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field.field_name,
                field_schema=field.schema_type,
            )
            logger.info("Created index for field %s", field.field_name)

    def upsert_vectors(self, points: list[PointStruct]):
        """
        Upsert vectors to the collection.

        Args:
            points: List of PointStruct objects containing vector data

        Returns:
            List of IDs for the upsert points
        """
        result = self.client.upsert(collection_name=self.collection_name, points=points)
        return result.ids

    def search_vectors(
        self, query_vector: list[float], limit: int = 10, filters: dict | None = None
    ):
        """
        Search for similar vectors in the collection.

        Args:
            query_vector: Vector to search for
            limit: Maximum number of results to return
            filters: Optional filter conditions

        Returns:
            Search results matching the query
        """
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
            query_filter=filter_conditions,
        )

    def delete_vectors(self, ids: list[str]):
        """
        Delete vectors by their IDs.

        Args:
            ids: List of vector IDs to delete
        """
        self.client.delete(
            collection_name=self.collection_name, points_selector={"points": ids}
        )

    def delete_vectors_by_filter(
        self, filters: dict[FilterType, dict[str, str | list[str]]]
    ):
        """
        Delete vectors matching the given filters.

        Args:
            filters: Dictionary of filter conditions
        """
        filter_payload = {}

        for filter_type, condition_dict in filters.items():
            field_conditions = []
            for key, value in condition_dict.items():
                if isinstance(value, list):
                    field_conditions.extend(
                        [
                            FieldCondition(key=key, match=MatchValue(value=item))
                            for item in value
                        ]
                    )
                else:
                    field_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            filter_payload[filter_type.value] = field_conditions

        points_selector = Filter(**filter_payload)
        self.client.delete(
            collection_name=self.collection_name, points_selector=points_selector
        )

"""
Utility functions and classes for the Terraform analysis agent.

This module provides common functionality for logging, embeddings, and Qdrant database operations.
"""

import os
import logging
from enum import Enum
from typing import Optional
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    PayloadSchemaType,
)


def setup_logging(module_name="terraform-analysis") -> logging.Logger:
    """
    Set up basic logging configuration.

    Args:
        module_name: Name for the logger

    Returns:
        A configured logger instance
    """
    debug_mode = os.getenv("DEBUG") is not None
    log_level = logging.DEBUG if debug_mode else logging.INFO

    logger_obj = logging.getLogger(module_name)
    logger_obj.setLevel(log_level)

    if not logger_obj.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger_obj.addHandler(handler)

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
    Create and return an async embedding function using OpenAI's API.

    Returns:
        An async function that takes a list of texts and returns their embeddings

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
    client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def embed_batch(texts: list[str]) -> list[list[float]]:
        """
        Create embeddings for a batch of texts asynchronously.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            Exception: If embedding creation fails
        """
        response = await client.embeddings.create(
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
    Each member is a tuple of (field_name, schema_type).
    """

    FILE_TYPE = ("file_type", PayloadSchemaType.KEYWORD)
    FILE_PATH = ("file_path", PayloadSchemaType.KEYWORD)
    REPO = ("repo", PayloadSchemaType.KEYWORD)
    LAST_MODIFIED = ("last_modified", PayloadSchemaType.DATETIME)
    CONTENT = ("content", PayloadSchemaType.TEXT)

    @property
    def field_name(self) -> str:
        """
        Return the field name from the enum value tuple.
        """
        return self.value[0]

    @property
    def schema_type(self) -> PayloadSchemaType:
        """
        Return the schema type from the enum value tuple.
        """
        return self.value[1]


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
            embed_fn: Optional async embedding function
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

        Returns:
            Collection information
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
        Creates indexes for all fields defined in PayloadField enum.
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
        Upsert vectors to the collection synchronously.

        Args:
            points: List of PointStruct objects containing vector data

        Returns:
            UpdateResult containing operation status
        """
        return self.client.upsert(collection_name=self.collection_name, points=points)

    def search_vectors(
        self,
        query_vector: Optional[list[float]] = None,
        metadata_filter: Optional[Filter] = None,
    ) -> Optional[str]:
        """
        Search for content in the database using either vector similarity or metadata filters.

        This method performs a search in the Qdrant database using either a vector similarity
        search or metadata-based filtering. It returns the file path of the first matching
        result, or None if no matches are found.

        Args:
            query_vector: Vector to search for (optional)
            metadata_filter: Filter conditions (optional)

        Returns:
            File path of the first matching result, or None if no matches found

        Raises:
            ValueError: If neither query_vector nor metadata_filter is provided
        """
        if query_vector is None and metadata_filter is None:
            raise ValueError("Either query_vector or metadata_filter must be provided")

        result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=metadata_filter,
        )
        if result is None:
            return None

        result = result.to_dict()
        return result.get("payload", {}).get(PayloadField.FILE_PATH.field_name)

    def check_metadata_exists(self, metadata_filter: Filter) -> bool:
        """
        Check if entries matching the given filter conditions exist in the database.

        Args:
            filter_selector: Filter conditions to check

        Returns:
            True if entries exist, False otherwise
        """
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=metadata_filter,
        )
        return result.count > 0

    def delete_vectors_by_filter(self, metadata_filter: Filter):
        """
        Delete vectors matching the given filter conditions.

        Args:
            metadata_filter: Filter conditions for deletion
        """
        self.client.delete(
            self.collection_name,
            metadata_filter,
        )

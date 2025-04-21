"""
Qdrant vector database client implementation.

This module provides a client class for interacting with Qdrant vector database,
handling operations like collection management, vector storage, and query.
"""

import os
from enum import Enum
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    PayloadSchemaType,
)

from .utils import DIMENSIONS, logger


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
    in a Qdrant database. It handles vector storage, query, and filtering operations.
    Supports both production (server-based) and debug (in-memory) modes.
    """

    def __init__(
        self,
        host: str = os.getenv("QDRANT_HOST") or "localhost",
        port: int = int(os.getenv("QDRANT_PORT") or "6333"),
        collection_name: str = os.getenv("QDRANT_COLLECTION_NAME") or "knowledge_db",
        embed_fn: callable = None,
        debug: bool = False,
    ):
        """
        Initialize QdrantDB client with configuration.

        Args:
            host: Qdrant server host (ignored in debug mode)
            port: Qdrant server port (ignored in debug mode)
            collection_name: Name of the collection to use
            embed_fn: Optional async embedding function
            debug: If True, uses an in-memory client for testing/debugging
        """
        if debug:
            self.client = QdrantClient(":memory:")
            logger.info("Using in-memory Qdrant client for debugging")
        else:
            self.client = QdrantClient(host=host, port=port)
            logger.info(f"Using Qdrant server at {host}:{port}")
        self.collection_name = collection_name
        self.embed_fn = embed_fn
        self.collection = self._ensure_collection()
        logger.info("QdrantDB client initialized successfully")

    def _ensure_collection(
        self, vector_size: int = DIMENSIONS
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

    def query_vectors(
        self,
        query_vector: Optional[list[float]] = None,
        metadata_filter: Optional[Filter] = None,
    ) -> Optional[str]:
        """
        Query for content in the database using either vector similarity or metadata filters.

        This method performs a query in the Qdrant database using either a vector similarity
        query or metadata-based filtering. It returns the file path of the first matching
        result, or None if no matches are found.

        Args:
            query_vector: Vector to query for (optional)
            metadata_filter: Filter conditions (optional)

        Returns:
            File path of the first matching result, or None if no matches found

        Raises:
            ValueError: If neither query_vector nor metadata_filter is provided
        """
        if query_vector is None and metadata_filter is None:
            raise ValueError("Either query_vector or metadata_filter must be provided")

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=metadata_filter,
        ).points
        if len(results) == 0:
            return None

        return results[0].payload.get(PayloadField.FILE_PATH.field_name)

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

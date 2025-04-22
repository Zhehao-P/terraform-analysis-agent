"""
Tests for the QdrantDB class.

This module contains tests for the QdrantDB class using an ephemeral in-memory database.
Tests are run in debug mode to avoid requiring a running Qdrant server.
"""

# pylint: disable=redefined-outer-name

import uuid
import pytest
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue
from common.qdrant_client import QdrantDB, PayloadField, FileType
from common.utils import DIMENSIONS

TEST_COLLECTION_NAME = "test_collection"


@pytest.fixture
def qdrant_db() -> QdrantDB:
    """Create a fresh QdrantDB instance for each test."""
    return QdrantDB(collection_name=TEST_COLLECTION_NAME, debug=True)


@pytest.fixture
def test_points() -> list[PointStruct]:
    """Create test points with proper dimensions."""
    return [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=[0.1] * DIMENSIONS,
            payload={
                PayloadField.FILE_TYPE.field_name: FileType.CODE.value,
                PayloadField.FILE_PATH.field_name: "test1.txt",
                PayloadField.REPO.field_name: "test_repo",
                PayloadField.CONTENT.field_name: "test content 1",
            },
        ),
        PointStruct(
            id=str(uuid.uuid4()),
            vector=[0.2] * DIMENSIONS,
            payload={
                PayloadField.FILE_TYPE.field_name: FileType.DOCUMENT.value,
                PayloadField.FILE_PATH.field_name: "test2.md",
                PayloadField.REPO.field_name: "test_repo",
                PayloadField.CONTENT.field_name: "test content 2",
            },
        ),
    ]


def test_ensure_collection(qdrant_db: QdrantDB):
    """Test that collection is created if it doesn't exist in debug mode."""
    assert qdrant_db.client.collection_exists(TEST_COLLECTION_NAME)


def test_build_payload_index(qdrant_db: QdrantDB):
    """Test that payload indexes are created correctly in debug mode."""
    qdrant_db.build_payload_index()
    collection_info = qdrant_db.client.get_collection(qdrant_db.collection_name)
    assert collection_info.payload_schema is not None


def test_upsert_vectors(qdrant_db: QdrantDB, test_points: list[PointStruct]):
    """Test that vectors can be upserted successfully in debug mode."""
    result = qdrant_db.upsert_vectors(test_points)
    assert result is not None
    count = qdrant_db.client.count(collection_name=qdrant_db.collection_name).count
    assert count == len(test_points)


def test_query_vectors_by_metadata(qdrant_db: QdrantDB, test_points: list[PointStruct]):
    """Test querying vectors using metadata filters in debug mode."""
    qdrant_db.upsert_vectors(test_points)

    # Create a filter for code files
    metadata_filter = Filter(
        must=[
            FieldCondition(
                key=PayloadField.FILE_TYPE.field_name,
                match=MatchValue(value=FileType.CODE.value),
            ),
        ],
    )

    # Query with the filter
    result = qdrant_db.query_vectors(metadata_filter=metadata_filter)
    assert result == "test1.txt"  # Should match the code file


def test_check_metadata_exists(qdrant_db: QdrantDB, test_points: list[PointStruct]):
    """Test checking if metadata exists in debug mode."""
    qdrant_db.upsert_vectors(test_points)

    # Check for existing metadata
    metadata_filter = Filter(
        must=[
            FieldCondition(
                key=PayloadField.FILE_TYPE.field_name,
                match=MatchValue(value=FileType.CODE.value),
            ),
        ],
    )
    assert qdrant_db.check_metadata_exists(metadata_filter)

    # Check for non-existent metadata
    non_existent_filter = Filter(
        must=[
            FieldCondition(
                key=PayloadField.FILE_TYPE.field_name,
                match=MatchValue(value="non_existent_type"),
            ),
        ],
    )
    assert not qdrant_db.check_metadata_exists(non_existent_filter)


def test_delete_vectors_by_filter(qdrant_db: QdrantDB, test_points: list[PointStruct]):
    """Test deleting vectors by filter in debug mode."""
    qdrant_db.upsert_vectors(test_points)

    # Create a filter for code files
    metadata_filter = Filter(
        must=[
            FieldCondition(
                key=PayloadField.FILE_TYPE.field_name,
                match=MatchValue(value=FileType.CODE.value),
            ),
        ],
    )

    # Delete vectors matching the filter
    qdrant_db.delete_vectors_by_filter(metadata_filter)

    # Verify deletion
    assert not qdrant_db.check_metadata_exists(metadata_filter)
    # Verify other points still exist
    other_filter = Filter(
        must=[
            FieldCondition(
                key=PayloadField.FILE_TYPE.field_name,
                match=MatchValue(value=FileType.DOCUMENT.value),
            ),
        ],
    )
    assert qdrant_db.check_metadata_exists(other_filter)

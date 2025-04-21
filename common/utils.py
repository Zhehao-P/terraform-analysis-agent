"""
Utility functions and classes for the Terraform analysis agent.

This module provides common functionality for logging, embeddings, and other utilities.
"""

import os
import logging
from openai import AsyncOpenAI

DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS") or "1024")

def setup_logging(module_name="terraform-analysis") -> logging.Logger:
    """
    Set up basic logging configuration.

    Args:
        module_name: Name for the logger

    Returns:
        A configured logger instance
    """
    debug_mode = os.getenv("DEBUG")
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
            dimensions=DIMENSIONS,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]

    logger.info("Using api endpoint: %s", api_base)
    logger.info("Using embedding model: %s", embedding_model)
    return embed_batch

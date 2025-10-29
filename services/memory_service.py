"""Memory service for SoyeBot - handles memory storage, retrieval, and AI-based extraction."""

import logging
import json
import io
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from functools import lru_cache
from contextlib import contextmanager

from services.database_service import DatabaseService

logger = logging.getLogger(__name__)

# Optional imports for semantic search
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not available, semantic search disabled")


class MemoryService:
    """Service for managing user memories with AI-based extraction and retrieval."""

    def __init__(
        self,
        db_service: DatabaseService,
        retrieval_mode: str = 'inject_all',
        embedding_model_name: str = 'all-MiniLM-L6-v2',
        cache_size: int = 50,
    ):
        """Initialize memory service.

        Args:
            db_service: DatabaseService instance
            retrieval_mode: 'inject_all' or 'semantic_search'
            embedding_model_name: Name of sentence transformer model
            cache_size: LRU cache size for memory retrieval
        """
        self.db_service = db_service
        self.retrieval_mode = retrieval_mode
        self.embedding_model_name = embedding_model_name
        self.cache_size = cache_size
        self.embedding_model = None
        self._load_embedding_model()

    def _load_embedding_model(self) -> None:
        """Load embedding model if semantic search is enabled."""
        if self.retrieval_mode == 'semantic_search' and EMBEDDINGS_AVAILABLE:
            try:
                logger.info(f"Loading embedding model: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}. Falling back to inject_all mode.")
                self.retrieval_mode = 'inject_all'
                self.embedding_model = None

    def unload_embedding_model(self) -> None:
        """Unload embedding model to free memory."""
        if self.embedding_model is not None:
            try:
                del self.embedding_model
                self.embedding_model = None
                logger.info("Embedding model unloaded")
            except Exception as e:
                logger.warning(f"Failed to unload embedding model: {e}")

    @staticmethod
    def get_gemini_function_calling_tools() -> List[Dict[str, Any]]:
        """Get function calling tools definition for Gemini API.

        Returns:
            List of tool definitions for function calling
        """
        # Import here to avoid circular dependencies
        import google.generativeai as genai

        return [
            genai.protos.Tool(
                function_declarations=[
                    genai.protos.FunctionDeclaration(
                        name="save_user_fact",
                        description="Save a fact learned about the user. Use this to remember important information about the user for future conversations.",
                        parameters=genai.protos.Schema(
                            type_=genai.protos.Type.OBJECT,
                            properties={
                                "fact": genai.protos.Schema(
                                    type_=genai.protos.Type.STRING,
                                    description="The fact about the user (e.g., 'User likes programming', 'User is from Korea')",
                                ),
                                "category": genai.protos.Schema(
                                    type_=genai.protos.Type.STRING,
                                    description="Category of the fact (e.g., 'hobby', 'location', 'profession')",
                                    enum=["hobby", "location", "profession", "education", "interest", "other"],
                                ),
                            },
                            required=["fact", "category"],
                        ),
                    ),
                    genai.protos.FunctionDeclaration(
                        name="save_preference",
                        description="Save a user preference or setting. Use this when the user expresses preferences about how you should interact with them.",
                        parameters=genai.protos.Schema(
                            type_=genai.protos.Type.OBJECT,
                            properties={
                                "preference": genai.protos.Schema(
                                    type_=genai.protos.Type.STRING,
                                    description="The preference (e.g., 'Prefers concise responses', 'Uses formal language')",
                                ),
                            },
                            required=["preference"],
                        ),
                    ),
                    genai.protos.FunctionDeclaration(
                        name="save_key_memory",
                        description="Save an important moment or memory that should be remembered long-term.",
                        parameters=genai.protos.Schema(
                            type_=genai.protos.Type.OBJECT,
                            properties={
                                "memory": genai.protos.Schema(
                                    type_=genai.protos.Type.STRING,
                                    description="The important memory or moment",
                                ),
                                "importance": genai.protos.Schema(
                                    type_=genai.protos.Type.INTEGER,
                                    description="Importance level from 1-10",
                                ),
                            },
                            required=["memory", "importance"],
                        ),
                    ),
                    genai.protos.FunctionDeclaration(
                        name="update_interaction_pattern",
                        description="Update user interaction patterns like favorite topics or sentiment.",
                        parameters=genai.protos.Schema(
                            type_=genai.protos.Type.OBJECT,
                            properties={
                                "topic": genai.protos.Schema(
                                    type_=genai.protos.Type.STRING,
                                    description="Topic discussed in this interaction",
                                ),
                                "sentiment": genai.protos.Schema(
                                    type_=genai.protos.Type.STRING,
                                    description="Overall sentiment of the interaction",
                                    enum=["positive", "neutral", "negative"],
                                ),
                            },
                            required=["topic"],
                        ),
                    ),
                ]
            ),
        ]

    def save_memory(
        self,
        memory_type: str,
        content: str,
        importance_score: float = 0.5,
    ) -> Optional[Any]:
        """Save a unified memory (shared across all users).

        Args:
            memory_type: Type of memory ('fact', 'preference', 'key_memory')
            content: Memory content
            importance_score: Importance score (0.0-1.0)

        Returns:
            Memory object or None
        """
        embedding = None
        try:
            # Generate embedding if semantic search is enabled
            if self.retrieval_mode == 'semantic_search' and self.embedding_model:
                embedding = self._encode_text(content)
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")

        return self.db_service.save_memory(
            memory_type=memory_type,
            content=content,
            importance_score=importance_score,
            embedding=embedding,
        )

    def _encode_text(self, text: str) -> Optional[bytes]:
        """Encode text to embedding vector.

        Args:
            text: Text to encode

        Returns:
            Binary encoded embedding or None
        """
        if not self.embedding_model:
            return None

        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            # Convert numpy array to bytes
            buffer = io.BytesIO()
            np.save(buffer, embedding)
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            return None

    def _decode_embedding(self, embedding_bytes: bytes) -> Optional[np.ndarray]:
        """Decode embedding from bytes.

        Args:
            embedding_bytes: Binary encoded embedding

        Returns:
            Numpy array or None
        """
        try:
            buffer = io.BytesIO(embedding_bytes)
            return np.load(buffer)
        except Exception as e:
            logger.error(f"Failed to decode embedding: {e}")
            return None

    @contextmanager
    def _with_embedding_model(self):
        """Context manager for safely loading embedding model."""
        model_was_loaded = self.embedding_model is not None
        try:
            if not model_was_loaded and self.retrieval_mode == 'semantic_search' and EMBEDDINGS_AVAILABLE:
                self._load_embedding_model()
            yield
        finally:
            if not model_was_loaded and self.embedding_model is not None:
                self.unload_embedding_model()

    def retrieve_memories(
        self,
        query: Optional[str] = None,
        max_memories: int = 20,
    ) -> List[Dict[str, Any]]:
        """Retrieve unified memories (shared across all users).

        Args:
            query: Search query (for semantic search)
            max_memories: Maximum number of memories to return

        Returns:
            List of memory dictionaries
        """
        if self.retrieval_mode == 'inject_all':
            return self._retrieve_inject_all(max_memories)
        elif self.retrieval_mode == 'semantic_search':
            return self._retrieve_semantic_search(query, max_memories)
        else:
            logger.warning(f"Unknown retrieval mode: {self.retrieval_mode}")
            return self._retrieve_inject_all(max_memories)

    def _retrieve_inject_all(self, max_memories: int) -> List[Dict[str, Any]]:
        """Retrieve all recent unified memories (simple mode).

        Args:
            max_memories: Maximum number of memories

        Returns:
            List of memory dictionaries
        """
        memories = self.db_service.get_memories(limit=max_memories)
        return self._format_memories(memories)

    def _retrieve_semantic_search(
        self,
        query: Optional[str],
        max_memories: int,
    ) -> List[Dict[str, Any]]:
        """Retrieve unified memories using semantic search.

        Args:
            query: Search query
            max_memories: Maximum number of memories

        Returns:
            List of memory dictionaries
        """
        if not query:
            return self._retrieve_inject_all(max_memories)

        with self._with_embedding_model():
            try:
                # Get all unified memories
                all_memories = self.db_service.get_memories(limit=1000)
                if not all_memories:
                    return []

                # Encode query
                query_embedding = self._encode_text(query)
                if not query_embedding:
                    return self._retrieve_inject_all(max_memories)

                query_vec = self._decode_embedding(query_embedding)
                if query_vec is None:
                    return self._retrieve_inject_all(max_memories)

                # Calculate similarities
                similarities = []
                for memory in all_memories:
                    if not memory.embedding:
                        similarities.append((memory, 0.0))
                        continue

                    memory_vec = self._decode_embedding(memory.embedding)
                    if memory_vec is None:
                        similarities.append((memory, 0.0))
                        continue

                    # Cosine similarity
                    similarity = np.dot(query_vec, memory_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(memory_vec) + 1e-10
                    )
                    similarities.append((memory, similarity))

                # Sort by similarity and return top K
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_memories = [m for m, _ in similarities[:max_memories]]
                return self._format_memories(top_memories)

            except Exception as e:
                logger.error(f"Semantic search failed: {e}")
                return self._retrieve_inject_all(max_memories)

    @staticmethod
    def _format_memories(memories: List[Any]) -> List[Dict[str, Any]]:
        """Format memories for injection into prompts.

        Args:
            memories: List of Memory objects

        Returns:
            List of formatted memory dictionaries
        """
        formatted = []
        for memory in memories:
            formatted.append({
                'id': memory.id,
                'type': memory.memory_type,
                'content': memory.content,
                'importance': memory.importance_score,
                'timestamp': memory.timestamp.isoformat() if memory.timestamp else None,
            })
        return formatted

    def get_memories_context(self, query: Optional[str] = None) -> str:
        """Get unified memories formatted as context string for prompts.

        Args:
            query: Search query (optional)

        Returns:
            Formatted memory context string
        """
        memories = self.retrieve_memories(query, max_memories=20)
        if not memories:
            return ""

        context = "## Shared Memories\n\n"
        by_type = {}
        for memory in memories:
            mem_type = memory['type']
            if mem_type not in by_type:
                by_type[mem_type] = []
            by_type[mem_type].append(memory)

        for mem_type, type_memories in by_type.items():
            context += f"### {mem_type.title()}s\n"
            for memory in type_memories:
                context += f"- {memory['content']}\n"
            context += "\n"

        return context.strip()

    def set_retrieval_mode(self, mode: str) -> bool:
        """Set retrieval mode dynamically.

        Args:
            mode: 'inject_all' or 'semantic_search'

        Returns:
            True if mode was set successfully
        """
        if mode not in ('inject_all', 'semantic_search'):
            logger.warning(f"Invalid retrieval mode: {mode}")
            return False

        old_mode = self.retrieval_mode
        self.retrieval_mode = mode

        if mode == 'semantic_search' and not self.embedding_model and EMBEDDINGS_AVAILABLE:
            self._load_embedding_model()
        elif mode == 'inject_all' and self.embedding_model:
            self.unload_embedding_model()

        logger.info(f"Retrieval mode changed: {old_mode} -> {mode}")
        return True

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.unload_embedding_model()

    # --- Handler functions for Gemini function calls ---

    def handle_save_user_fact(self, user_id: str, fact: str, category: str) -> str:
        """Handler for save_user_fact function call (saves to unified memory).

        Args:
            user_id: Discord user ID (not used, for compatibility)
            fact: The fact about the user
            category: Category of the fact

        Returns:
            Confirmation message
        """
        try:
            self.save_memory(
                memory_type='fact',
                content=f"[{category.title()}] {fact}",
                importance_score=0.7,
            )
            return f"✓ Saved fact: {fact}"
        except Exception as e:
            logger.error(f"Failed to save fact: {e}")
            return f"✗ Failed to save fact: {e}"

    def handle_save_preference(self, user_id: str, preference: str) -> str:
        """Handler for save_preference function call (saves to unified memory).

        Args:
            user_id: Discord user ID (not used, for compatibility)
            preference: The user preference

        Returns:
            Confirmation message
        """
        try:
            self.save_memory(
                memory_type='preference',
                content=preference,
                importance_score=0.8,
            )
            return f"✓ Noted preference: {preference}"
        except Exception as e:
            logger.error(f"Failed to save preference: {e}")
            return f"✗ Failed to save preference: {e}"

    def handle_save_key_memory(self, user_id: str, memory: str, importance: int) -> str:
        """Handler for save_key_memory function call (saves to unified memory).

        Args:
            user_id: Discord user ID (not used, for compatibility)
            memory: The important memory
            importance: Importance level (1-10)

        Returns:
            Confirmation message
        """
        try:
            importance_score = min(1.0, max(0.0, importance / 10.0))
            self.save_memory(
                memory_type='key_memory',
                content=memory,
                importance_score=importance_score,
            )
            return f"✓ Saved important memory: {memory}"
        except Exception as e:
            logger.error(f"Failed to save key memory: {e}")
            return f"✗ Failed to save key memory: {e}"

    def handle_update_interaction_pattern(
        self,
        user_id: str,
        topic: str,
        sentiment: Optional[str] = None,
    ) -> str:
        """Handler for update_interaction_pattern function call.

        Args:
            user_id: Discord user ID
            topic: Topic discussed
            sentiment: Sentiment ('positive', 'neutral', 'negative')

        Returns:
            Confirmation message
        """
        try:
            sentiment_value = None
            if sentiment == 'positive':
                sentiment_value = 0.8
            elif sentiment == 'neutral':
                sentiment_value = 0.5
            elif sentiment == 'negative':
                sentiment_value = 0.2

            self.db_service.update_interaction_pattern(
                user_id=user_id,
                topic=topic,
                sentiment=sentiment_value,
            )
            return f"✓ Updated interaction pattern - Topic: {topic}"
        except Exception as e:
            logger.error(f"Failed to update interaction pattern: {e}")
            return f"✗ Failed to update pattern: {e}"

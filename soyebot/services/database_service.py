"""Database service for SoyeBot."""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from database.models import Base, User, Memory, ConversationHistory, InteractionPattern

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for managing database operations."""

    def __init__(self, database_path: str = 'soyebot.db'):
        """Initialize database service.

        Args:
            database_path: Path to SQLite database file
        """
        self.database_url = f'sqlite:///{database_path}'
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=2,
            max_overflow=0,
            connect_args={'check_same_thread': False},
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def close(self) -> None:
        """Close database connection."""
        self.engine.dispose()

    # --- User Operations ---

    def get_or_create_user(self, user_id: str, username: str) -> User:
        """Get or create a user.

        Args:
            user_id: Discord user ID
            username: Discord username

        Returns:
            User object
        """
        session = self.get_session()
        try:
            user = session.query(User).filter_by(user_id=str(user_id)).first()
            if not user:
                user = User(user_id=str(user_id), username=username)
                session.add(user)
                session.commit()
                logger.info(f"Created new user: {user_id}")
            else:
                user.last_seen = datetime.utcnow()
                session.commit()
            return user
        finally:
            session.close()

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID.

        Args:
            user_id: Discord user ID

        Returns:
            User object or None
        """
        session = self.get_session()
        try:
            return session.query(User).filter_by(user_id=str(user_id)).first()
        finally:
            session.close()

    def update_user_last_seen(self, user_id: str) -> None:
        """Update user's last seen timestamp.

        Args:
            user_id: Discord user ID
        """
        session = self.get_session()
        try:
            user = session.query(User).filter_by(user_id=str(user_id)).first()
            if user:
                user.last_seen = datetime.utcnow()
                session.commit()
        finally:
            session.close()

    # --- Memory Operations ---

    def save_memory(
        self,
        memory_type: str,
        content: str,
        importance_score: float = 0.5,
        embedding: Optional[bytes] = None,
    ) -> Memory:
        """Save a unified memory (shared across all users).

        Args:
            memory_type: Type of memory ('fact', 'preference', 'key_memory')
            content: Memory content
            importance_score: Importance score (0.0-1.0)
            embedding: Binary encoded embedding vector

        Returns:
            Memory object
        """
        session = self.get_session()
        try:
            memory = Memory(
                memory_type=memory_type,
                content=content,
                importance_score=importance_score,
                embedding=embedding,
            )
            session.add(memory)
            session.commit()
            logger.info(f"Saved unified memory: {memory_type}")
            return memory
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    def get_memories(
        self,
        memory_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Memory]:
        """Get unified memories (shared across all users).

        Args:
            memory_type: Filter by memory type (optional)
            limit: Maximum number of memories to return

        Returns:
            List of Memory objects
        """
        session = self.get_session()
        try:
            query = session.query(Memory)
            if memory_type:
                query = query.filter_by(memory_type=memory_type)
            memories = query.order_by(Memory.importance_score.desc(), Memory.timestamp.desc()).limit(limit).all()
            return memories
        finally:
            session.close()

    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            True if deleted, False otherwise
        """
        session = self.get_session()
        try:
            memory = session.query(Memory).filter_by(id=memory_id).first()
            if memory:
                session.delete(memory)
                session.commit()
                logger.info(f"Deleted memory {memory_id}")
                return True
            return False
        finally:
            session.close()

    def delete_all_memories(self, user_id: str = None) -> int:
        """Delete all unified memories (shared across all users).

        Args:
            user_id: Discord user ID (ignored for unified memory, kept for compatibility)

        Returns:
            Number of deleted memories
        """
        session = self.get_session()
        try:
            count = session.query(Memory).delete()
            session.commit()
            logger.info(f"Deleted {count} unified memories")
            return count
        finally:
            session.close()

    # --- Conversation History Operations ---

    def save_conversation(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
    ) -> ConversationHistory:
        """Save conversation message.

        Args:
            user_id: Discord user ID
            session_id: Session/thread ID (message ID)
            role: 'user' or 'assistant'
            content: Message content

        Returns:
            ConversationHistory object
        """
        session = self.get_session()
        try:
            conversation = ConversationHistory(
                user_id=str(user_id),
                session_id=str(session_id),
                role=role,
                content=content,
            )
            session.add(conversation)
            session.commit()
            return conversation
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    def get_conversation_history(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[ConversationHistory]:
        """Get conversation history.

        Args:
            user_id: Discord user ID
            session_id: Filter by session ID (optional)
            limit: Maximum number of messages to return

        Returns:
            List of ConversationHistory objects
        """
        session_db = self.get_session()
        try:
            query = session_db.query(ConversationHistory).filter_by(user_id=str(user_id))
            if session_id:
                query = query.filter_by(session_id=str(session_id))
            history = query.order_by(ConversationHistory.timestamp.desc()).limit(limit).all()
            return list(reversed(history))  # Return in chronological order
        finally:
            session_db.close()

    def delete_old_conversations(self, days: int = 7) -> int:
        """Delete conversation history older than specified days.

        Args:
            days: Days to keep (default: 7)

        Returns:
            Number of deleted records
        """
        session = self.get_session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            count = session.query(ConversationHistory).filter(
                ConversationHistory.timestamp < cutoff_date
            ).delete()
            session.commit()
            logger.info(f"Deleted {count} old conversation records")
            return count
        finally:
            session.close()

    # --- Interaction Pattern Operations ---

    def get_or_create_interaction_pattern(self, user_id: str) -> InteractionPattern:
        """Get or create interaction pattern for user.

        Args:
            user_id: Discord user ID

        Returns:
            InteractionPattern object
        """
        session = self.get_session()
        try:
            pattern = session.query(InteractionPattern).filter_by(user_id=str(user_id)).first()
            if not pattern:
                pattern = InteractionPattern(user_id=str(user_id))
                session.add(pattern)
                session.commit()
            return pattern
        finally:
            session.close()

    def update_interaction_pattern(
        self,
        user_id: str,
        topic: Optional[str] = None,
        sentiment: Optional[float] = None,
    ) -> None:
        """Update interaction pattern.

        Args:
            user_id: Discord user ID
            topic: Topic to add to favorites (optional)
            sentiment: Sentiment value to include in average (optional)
        """
        session = self.get_session()
        try:
            pattern = session.query(InteractionPattern).filter_by(user_id=str(user_id)).first()
            if not pattern:
                pattern = InteractionPattern(user_id=str(user_id))
                session.add(pattern)

            pattern.total_messages += 1
            pattern.last_interaction = datetime.utcnow()

            if topic:
                topics = pattern.get_favorite_topics()
                topics[topic] = topics.get(topic, 0) + 1
                pattern.set_favorite_topics(topics)

            if sentiment is not None:
                # Update rolling average
                old_avg = pattern.sentiment_avg
                new_avg = (old_avg * (pattern.total_messages - 1) + sentiment) / pattern.total_messages
                pattern.sentiment_avg = new_avg

            session.commit()
        finally:
            session.close()

    def get_interaction_pattern(self, user_id: str) -> Optional[InteractionPattern]:
        """Get interaction pattern for user.

        Args:
            user_id: Discord user ID

        Returns:
            InteractionPattern object or None
        """
        session = self.get_session()
        try:
            return session.query(InteractionPattern).filter_by(user_id=str(user_id)).first()
        finally:
            session.close()

    # --- Database Maintenance ---

    def cleanup_expired_data(self, conversation_retention_days: int = 7) -> Dict[str, int]:
        """Clean up old data.

        Args:
            conversation_retention_days: Days to keep conversation history

        Returns:
            Dictionary with cleanup statistics
        """
        stats = {
            'deleted_conversations': self.delete_old_conversations(conversation_retention_days),
        }
        logger.info(f"Database cleanup completed: {stats}")
        return stats

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with database stats
        """
        session = self.get_session()
        try:
            return {
                'total_users': session.query(func.count(User.id)).scalar(),
                'total_memories': session.query(func.count(Memory.id)).scalar(),
                'total_conversations': session.query(func.count(ConversationHistory.id)).scalar(),
                'users_with_patterns': session.query(func.count(InteractionPattern.id)).scalar(),
            }
        finally:
            session.close()

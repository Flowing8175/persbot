"""Database service for SoyeBot."""

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, Optional

from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from database.models import Base, User, InteractionPattern

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
            pool_size=5,  # Increased from 2 for better concurrency
            max_overflow=10,  # Allow overflow with limit for spike handling
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,  # Recycle connections every hour
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

    @contextmanager
    def _session_scope(self) -> Iterator[Session]:
        session = self.get_session()
        try:
            yield session
        finally:
            session.close()

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
        with self._session_scope() as session:
            user = session.query(User).filter_by(user_id=str(user_id)).first()
            if not user:
                user = User(user_id=str(user_id), username=username)
                session.add(user)
                session.commit()
                logger.info("Created new user: %s", user_id)
            else:
                user.last_seen = datetime.utcnow()
                session.commit()
            return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID.

        Args:
            user_id: Discord user ID

        Returns:
            User object or None
        """
        with self._session_scope() as session:
            return session.query(User).filter_by(user_id=str(user_id)).first()

    def update_user_last_seen(self, user_id: str) -> None:
        """Update user's last seen timestamp.

        Args:
            user_id: Discord user ID
        """
        with self._session_scope() as session:
            user = session.query(User).filter_by(user_id=str(user_id)).first()
            if user:
                user.last_seen = datetime.utcnow()
                session.commit()

    # --- Interaction Pattern Operations ---

    def get_or_create_interaction_pattern(self, user_id: str) -> InteractionPattern:
        """Get or create interaction pattern for user.

        Args:
            user_id: Discord user ID

        Returns:
            InteractionPattern object
        """
        with self._session_scope() as session:
            pattern = session.query(InteractionPattern).filter_by(user_id=str(user_id)).first()
            if not pattern:
                pattern = InteractionPattern(user_id=str(user_id))
                session.add(pattern)
                session.commit()
            return pattern

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
        with self._session_scope() as session:
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
                old_avg = pattern.sentiment_avg
                new_avg = (old_avg * (pattern.total_messages - 1) + sentiment) / pattern.total_messages
                pattern.sentiment_avg = new_avg

            session.commit()

    def get_interaction_pattern(self, user_id: str) -> Optional[InteractionPattern]:
        """Get interaction pattern for user.

        Args:
            user_id: Discord user ID

        Returns:
            InteractionPattern object or None
        """
        with self._session_scope() as session:
            return session.query(InteractionPattern).filter_by(user_id=str(user_id)).first()

    # --- Database Maintenance ---

    def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up old data.

        Returns:
            Dictionary with cleanup statistics
        """
        stats = {}
        logger.info(f"Database cleanup completed: {stats}")
        return stats

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with database stats
        """
        with self._session_scope() as session:
            return {
                'total_users': session.query(func.count(User.id)).scalar(),
                'users_with_patterns': session.query(func.count(InteractionPattern.id)).scalar(),
            }

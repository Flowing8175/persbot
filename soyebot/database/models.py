"""SQLAlchemy models for SoyeBot database."""

import json
from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    LargeBinary,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class User(Base):
    """User model for storing Discord user information."""
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(20), unique=True, nullable=False, index=True)
    username = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    config_json = Column(Text, default='{}', nullable=False)  # User preferences

    # Relationships
    interaction_pattern = relationship('InteractionPattern', back_populates='user', uselist=False, cascade='all, delete-orphan')

    __table_args__ = (
        Index('idx_user_id', 'user_id'),
    )


class Memory(Base):
    """Memory model for storing unified (shared) memories."""
    __tablename__ = 'memories'

    id = Column(Integer, primary_key=True, autoincrement=True)
    memory_type = Column(String(50), nullable=False)  # 'fact', 'preference', 'key_memory'
    content = Column(Text, nullable=False)
    embedding = Column(LargeBinary, nullable=True)  # Stored numpy array for semantic search
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    importance_score = Column(Float, default=0.5, nullable=False)  # 0.0 to 1.0

    __table_args__ = (
        Index('idx_memory_type', 'memory_type'),
        Index('idx_memory_timestamp', 'timestamp'),
        Index('idx_memory_importance', 'importance_score'),
    )


class InteractionPattern(Base):
    """Interaction pattern model for storing user interaction statistics."""
    __tablename__ = 'interaction_patterns'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(20), ForeignKey('users.user_id'), unique=True, nullable=False, index=True)
    total_messages = Column(Integer, default=0, nullable=False)
    last_interaction = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    favorite_topics = Column(Text, default='{}', nullable=False)  # JSON encoded dict
    sentiment_avg = Column(Float, default=0.0, nullable=False)  # Average sentiment
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    user = relationship('User', back_populates='interaction_pattern')

    __table_args__ = (
        Index('idx_pattern_user', 'user_id'),
    )

    def get_favorite_topics(self) -> dict:
        """Get favorite topics as dict."""
        try:
            return json.loads(self.favorite_topics)
        except (json.JSONDecodeError, TypeError):
            return {}

    def set_favorite_topics(self, topics: dict) -> None:
        """Set favorite topics from dict."""
        self.favorite_topics = json.dumps(topics)

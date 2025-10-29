"""SQLAlchemy models for SoyeBot database."""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base

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


    __table_args__ = (
        Index('idx_user_id', 'user_id'),
    )


class Memory(Base):
    """Unified memory model shared across all users."""
    __tablename__ = 'memories'

    id = Column(Integer, primary_key=True, autoincrement=True)
    memory_type = Column(String(50), nullable=False, index=True)  # 'fact', 'preference', 'key_memory'
    content = Column(Text, nullable=False)
    embedding = Column(Text, nullable=True)  # Stored as JSON or binary
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    importance_score = Column(Integer, default=0, nullable=False)

    __table_args__ = (
        Index('idx_memory_type', 'memory_type'),
        Index('idx_timestamp', 'timestamp'),
    )


class InteractionPattern(Base):
    """Interaction pattern model for tracking user interaction trends."""
    __tablename__ = 'interaction_patterns'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(20), unique=True, nullable=False, index=True)
    total_messages = Column(Integer, default=0, nullable=False)
    sentiment_avg = Column(Integer, default=0, nullable=False)
    favorite_topics = Column(Text, default='{}', nullable=False)  # JSON string
    last_interaction = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index('idx_user_id_pattern', 'user_id'),
    )



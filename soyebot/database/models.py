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



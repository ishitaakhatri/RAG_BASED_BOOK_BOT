from sqlalchemy import Column, String, Integer, Text, DateTime, JSON, Index
from sqlalchemy.sql import func
from .db import Base

class ConversationTurn(Base):
    __tablename__ = "conversation_turns"

    # ID format: "sess_{uuid}_turn_{int}"
    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False)
    turn_number = Column(Integer, nullable=False)
    
    # Unlimited text storage (Postgres TEXT)
    user_query = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    
    # The concise summary used for vector search embedding
    search_payload = Column(Text, nullable=True)
    
    # JSON for extra metadata (sources, referenced_turn, etc.)
    raw_metadata = Column(JSON, default={})
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Indices for fast lookups
    __table_args__ = (
        Index('idx_session_turn', 'session_id', 'turn_number'),
        Index('idx_session_created', 'session_id', 'created_at'),
    )
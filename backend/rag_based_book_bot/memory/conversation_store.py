"""
Async Conversation Store
Implementation: "Piggyback" Strategy (Postgres + Pinecone)
Uses modern Pinecone v5 Client and Async SQLAlchemy
"""

import time
import logging
from typing import List, Dict, Optional
from pinecone import Pinecone
from sqlalchemy import select, delete, func, desc
from sqlalchemy.exc import SQLAlchemyError

from app_config import get_config
from ..db import AsyncSessionLocal
from ..models import ConversationTurn
from .embedding_utils import embed_query_for_search

logger = logging.getLogger(__name__)
settings = get_config()

# --- Global Client Instances ---
_pc = None
_index = None

def get_pinecone_index():
    """Singleton accessor for Pinecone Index"""
    global _pc, _index
    if _index is None:
        # Modern Pinecone Client Usage (v5.0+)
        _pc = Pinecone(api_key=settings.vector_db.api_key)
        _index = _pc.Index(settings.vector_db.index_name)
    return _index

# --- ASYNC STORE IMPLEMENTATION ---

async def save_conversation_turn(
    session_id: str,
    turn_number: int,
    user_query: str,
    assistant_response: str,
    search_summary: Optional[str] = None,
    sources_used: Optional[List[str]] = None,
    **kwargs
) -> Dict:
    """
    Saves a turn to Postgres (Content) and Pinecone (Index).
    """
    vector_id = f"{session_id}_turn{turn_number}"
    
    # Fallback if no summary provided
    payload_to_embed = search_summary if search_summary else f"Q: {user_query}\nA: {assistant_response[:200]}..."

    async with AsyncSessionLocal() as session:
        try:
            # 1. Deduplication check
            existing = await session.get(ConversationTurn, vector_id)
            if existing:
                return {"success": True, "skipped": True, "turn_id": vector_id}

            # 2. Write to Postgres (Source of Truth)
            new_turn = ConversationTurn(
                id=vector_id,
                session_id=session_id,
                turn_number=turn_number,
                user_query=user_query,
                assistant_response=assistant_response,
                search_payload=payload_to_embed,
                raw_metadata={
                    "sources_used": sources_used or [],
                    **kwargs
                }
            )
            session.add(new_turn)
            await session.commit()
            logger.info(f"✅ [DB] Saved turn {vector_id} to Postgres")

            # 3. Write to Pinecone (Index)
            try:
                embedding = embed_query_for_search(payload_to_embed)
                index = get_pinecone_index()
                
                index.upsert(
                    vectors=[{
                        "id": vector_id,
                        "values": embedding,
                        "metadata": {
                            "session_id": session_id,
                            "turn_number": int(turn_number),
                            "timestamp": float(time.time())
                        }
                    }],
                    namespace="conversations"
                )
                logger.info(f"✅ [Pinecone] Indexed {vector_id}")
                
            except Exception as e:
                # Non-blocking failure for vector index
                logger.error(f"⚠️ [Pinecone] Indexing failed for {vector_id}: {e}")

            return {"success": True, "turn_id": vector_id}

        except SQLAlchemyError as e:
            await session.rollback()
            logger.error(f"❌ [DB] Failed to save turn: {e}")
            return {"success": False, "error": str(e)}

async def load_conversation(session_id: str, max_turns: int = 10) -> List[Dict]:
    """Load history directly from Postgres (Fast & Cheap)"""
    async with AsyncSessionLocal() as session:
        try:
            stmt = (
                select(ConversationTurn)
                .where(ConversationTurn.session_id == session_id)
                .order_by(ConversationTurn.turn_number.asc())
            )
            result = await session.execute(stmt)
            turns = result.scalars().all()
            
            history = []
            for t in turns[-max_turns:]:
                # Safe metadata extraction
                meta = t.raw_metadata or {}
                history.append({
                    "turn_number": t.turn_number,
                    "user_query": t.user_query,
                    "assistant_response": t.assistant_response,
                    "timestamp": t.created_at.timestamp() if t.created_at else 0,
                    "sources_used": meta.get("sources_used", []),
                    "needs_retrieval": meta.get("needs_retrieval", True),
                    "referenced_turn": meta.get("referenced_turn")
                })
            
            return history
        except Exception as e:
            logger.error(f"❌ Failed to load history: {e}")
            return []

async def search_conversation_context(
    session_id: str,
    query: str,
    top_k: int = 5
) -> List[Dict]:
    """
    Hybrid Search:
    1. Vector Search in Pinecone -> Get IDs
    2. Fetch Content from Postgres -> Get Text
    """
    try:
        # Step 1: Pinecone Search
        query_embedding = embed_query_for_search(query)
        index = get_pinecone_index()
        
        results = index.query(
            vector=query_embedding,
            filter={"session_id": session_id},
            top_k=top_k,
            namespace="conversations"
        )
        
        if not results.matches:
            return []
            
        id_score_map = {m.id: m.score for m in results.matches}
        target_ids = list(id_score_map.keys())
        
        if not target_ids:
            return []

        # Step 2: Postgres Fetch
        async with AsyncSessionLocal() as session:
            stmt = select(ConversationTurn).where(ConversationTurn.id.in_(target_ids))
            db_results = await session.execute(stmt)
            turns = db_results.scalars().all()
            
        # Step 3: Merge
        relevant_turns = []
        for t in turns:
            relevant_turns.append({
                "turn_number": t.turn_number,
                "user_query": t.user_query,
                "assistant_response": t.assistant_response,
                "relevance_score": id_score_map.get(t.id, 0),
                "timestamp": t.created_at.timestamp() if t.created_at else 0
            })
            
        relevant_turns.sort(key=lambda x: x["relevance_score"], reverse=True)
        return relevant_turns

    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        return []

async def delete_session(session_id: str):
    """Clean up both DB and Vector Store"""
    async with AsyncSessionLocal() as session:
        try:
            # 1. DB Delete
            await session.execute(
                delete(ConversationTurn).where(ConversationTurn.session_id == session_id)
            )
            await session.commit()
            
            # 2. Pinecone Delete (Best Effort)
            try:
                index = get_pinecone_index()
                index.delete(filter={"session_id": session_id}, namespace="conversations")
            except Exception:
                pass
                
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

async def list_all_sessions(user_id: Optional[str] = None, limit: int = 50) -> List[Dict]:
    """List sessions by aggregating turns in Postgres"""
    async with AsyncSessionLocal() as session:
        try:
            # 🔥 UPDATED: Include min(created_at) to satisfy Pydantic models
            stmt = (
                select(
                    ConversationTurn.session_id,
                    func.min(ConversationTurn.created_at).label("created_at"),
                    func.max(ConversationTurn.created_at).label("last_update"),
                    func.count(ConversationTurn.id).label("message_count")
                )
                .group_by(ConversationTurn.session_id)
                .order_by(desc("last_update"))
                .limit(limit)
            )
            
            result = await session.execute(stmt)
            rows = result.all()
            
            sessions = []
            for row in rows:
                sessions.append({
                    "session_id": row.session_id,
                    "updated_at": row.last_update.timestamp() if row.last_update else 0,
                    "created_at": row.created_at.timestamp() if row.created_at else 0,
                    "message_count": row.message_count,
                    "title": f"Session {row.session_id[:8]}...", # Simplified title
                    "last_message": "View conversation details" # Placeholder for missing field
                })
            return sessions
        except Exception as e:
            logger.error(f"❌ Failed to list sessions: {e}")
            return []

async def search_across_sessions(query: str, user_id: Optional[str] = None, top_k: int = 10) -> List[Dict]:
    """Search Pinecone across all sessions, then hydrate from DB"""
    try:
        query_embedding = embed_query_for_search(query)
        index = get_pinecone_index()
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace="conversations"
        )
        
        if not results.matches:
            return []

        id_score_map = {m.id: m.score for m in results.matches}
        target_ids = list(id_score_map.keys())

        if not target_ids:
            return []

        async with AsyncSessionLocal() as session:
            stmt = select(ConversationTurn).where(ConversationTurn.id.in_(target_ids))
            db_results = await session.execute(stmt)
            turns = db_results.scalars().all()
            
        matches = []
        for t in turns:
            matches.append({
                "session_id": t.session_id,
                "turn_number": t.turn_number,
                "user_query": t.user_query,
                "assistant_response": t.assistant_response,
                "relevance_score": id_score_map.get(t.id, 0),
                "timestamp": t.created_at.timestamp() if t.created_at else 0
            })
            
        matches.sort(key=lambda x: x["relevance_score"], reverse=True)
        return matches

    except Exception as e:
        logger.error(f"❌ Cross-session search failed: {e}")
        return []

def get_session_metadata(session_id: str) -> Optional[Dict]:
    """Metadata stub"""
    return {"session_id": session_id}
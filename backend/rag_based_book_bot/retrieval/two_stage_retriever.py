"""
Two-Stage Retrieval System
Stage 1: Search summaries for high-level matches
Stage 2: Fetch full chunks for top summary matches
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

try:
    from pinecone import Pinecone
    _HAS_PINECONE = True
except Exception:
    _HAS_PINECONE = False

logger = logging.getLogger("two_stage_retriever")

# Config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "coding-books")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "books_rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@dataclass
class TwoStageConfig:
    """Configuration for two-stage retrieval"""
    # Stage 1: Summary retrieval
    summary_top_k: int = 5  # How many summaries to retrieve
    summary_score_threshold: float = 0.7  # Minimum similarity for summaries
    
    # Stage 2: Full chunk retrieval
    fetch_full_for_top_n: int = 2  # Fetch full chunks for top N summaries
    chunks_per_summary: int = 3  # Max chunks to fetch per summary
    
    # Fallback to direct chunk retrieval
    enable_fallback: bool = True  # If no good summaries, search chunks directly
    fallback_top_k: int = 5  # Chunks to retrieve in fallback mode
    
    # Query routing
    use_query_routing: bool = True  # Smart routing based on query type


class QueryRouter:
    """
    Decides whether to use summary-first or direct chunk retrieval
    based on query characteristics
    """
    
    # Keywords indicating high-level/conceptual queries (use summaries)
    CONCEPTUAL_KEYWORDS = [
        'what is', 'explain', 'overview', 'introduction', 'concept',
        'difference between', 'compare', 'types of', 'categories',
        'how does', 'why', 'main idea', 'summarize', 'describe',
        'approaches', 'methods', 'techniques', 'strategies'
    ]
    
    # Keywords indicating specific/detailed queries (skip summaries)
    SPECIFIC_KEYWORDS = [
        'code', 'example', 'implementation', 'syntax', 'function',
        'parameter', 'error', 'bug', 'specific', 'exact', 'formula',
        'equation', 'algorithm steps', 'line by line', 'debug'
    ]
    
    @staticmethod
    def should_use_summaries(query: str) -> bool:
        """
        Determine if query should use summary-first retrieval
        
        Args:
            query: User query string
            
        Returns:
            True if summary-first is recommended, False otherwise
        """
        query_lower = query.lower()
        
        # Check for specific keywords (skip summaries)
        for keyword in QueryRouter.SPECIFIC_KEYWORDS:
            if keyword in query_lower:
                logger.info(f"🎯 Query contains '{keyword}' - using direct chunk retrieval")
                return False
        
        # Check for conceptual keywords (use summaries)
        for keyword in QueryRouter.CONCEPTUAL_KEYWORDS:
            if keyword in query_lower:
                logger.info(f"🎯 Query contains '{keyword}' - using summary-first retrieval")
                return True
        
        # Default: use summaries for longer queries (more likely conceptual)
        word_count = len(query.split())
        if word_count > 8:
            logger.info(f"🎯 Long query ({word_count} words) - using summary-first retrieval")
            return True
        
        # Short, ambiguous queries - direct chunk retrieval
        logger.info(f"🎯 Short query ({word_count} words) - using direct chunk retrieval")
        return False


class TwoStageRetriever:
    """
    Two-stage retrieval system:
    1. Search summaries for relevant sections
    2. Fetch full chunks from top matching summaries
    """
    
    def __init__(self, config: Optional[TwoStageConfig] = None):
        self.config = config or TwoStageConfig()
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"✅ Loaded embedding model: {EMBEDDING_MODEL}")
        
        # Initialize query router
        self.router = QueryRouter()
        
        # Initialize Pinecone
        self.pinecone_index = None
        if not _HAS_PINECONE:
            logger.error("Pinecone library not installed")
            raise ImportError("Pinecone required for retrieval")
        
        if not PINECONE_API_KEY:
            logger.error("PINECONE_API_KEY not set")
            raise ValueError("PINECONE_API_KEY required")
        
        try:
            pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
            
            # Get index with host
            PINECONE_HOST = os.getenv("PINECONE_INDEX_HOST")
            
            if PINECONE_HOST:
                self.pinecone_index = pinecone_client.Index(PINECONE_INDEX, host=PINECONE_HOST)
            else:
                # Auto-detect host
                indexes = pinecone_client.list_indexes()
                target_idx = None
                for idx in indexes:
                    if idx.name == PINECONE_INDEX:
                        target_idx = idx
                        break
                
                if target_idx:
                    self.pinecone_index = pinecone_client.Index(PINECONE_INDEX, host=target_idx.host)
                else:
                    raise ValueError(f"Index '{PINECONE_INDEX}' not found")
            
            logger.info(f"✅ Connected to Pinecone index: {PINECONE_INDEX}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for query"""
        try:
            embedding = self.embedding_model.encode(query)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise
    
    def _search_summaries(
        self,
        query_embedding: List[float],
        top_k: int,
        book_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for relevant summaries (Stage 1)
        
        Args:
            query_embedding: Query vector
            top_k: Number of summaries to retrieve
            book_id: Optional book filter
            
        Returns:
            List of summary results with metadata
        """
        try:
            # Build filter for summaries
            filter_dict = {"type": "summary"}
            if book_id:
                filter_dict["book_id"] = book_id
            
            logger.info(f"🔍 Stage 1: Searching summaries (top_k={top_k})...")
            
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=PINECONE_NAMESPACE,
                filter=filter_dict,
                include_metadata=True
            )
            
            matches = results.get('matches', [])
            logger.info(f"   Found {len(matches)} summary matches")
            
            # Filter by score threshold
            filtered_matches = [
                m for m in matches 
                if m.get('score', 0) >= self.config.summary_score_threshold
            ]
            
            logger.info(f"   {len(filtered_matches)} summaries above threshold ({self.config.summary_score_threshold})")
            
            return filtered_matches
            
        except Exception as e:
            logger.error(f"Summary search failed: {e}")
            return []
    
    def _search_chunks(
        self,
        query_embedding: List[float],
        top_k: int,
        book_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for full chunks directly (fallback or direct mode)
        
        Args:
            query_embedding: Query vector
            top_k: Number of chunks to retrieve
            book_id: Optional book filter
            
        Returns:
            List of chunk results with metadata
        """
        try:
            # Build filter for full chunks
            filter_dict = {"type": "full"}
            if book_id:
                filter_dict["book_id"] = book_id
            
            logger.info(f"🔍 Searching full chunks (top_k={top_k})...")
            
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=PINECONE_NAMESPACE,
                filter=filter_dict,
                include_metadata=True
            )
            
            matches = results.get('matches', [])
            logger.info(f"   Found {len(matches)} chunk matches")
            
            return matches
            
        except Exception as e:
            logger.error(f"Chunk search failed: {e}")
            return []
    
    def _fetch_chunks_for_summary(
        self,
        summary_metadata: Dict,
        query_embedding: List[float],
        max_chunks: int
    ) -> List[Dict]:
        """
        Fetch full chunks linked to a summary (Stage 2)
        
        Args:
            summary_metadata: Metadata from summary match
            query_embedding: Query vector for re-ranking
            max_chunks: Maximum chunks to return
            
        Returns:
            List of chunk results
        """
        try:
            # Get linked chunk IDs from summary metadata
            linked_ids_str = summary_metadata.get('linked_chunk_ids', '[]')
            
            # Parse the linked IDs (stored as string representation of list)
            import ast
            try:
                linked_chunk_indices = ast.literal_eval(linked_ids_str)
            except:
                logger.warning(f"Could not parse linked_chunk_ids: {linked_ids_str}")
                return []
            
            if not linked_chunk_indices:
                logger.debug("No linked chunks found in summary metadata")
                return []
            
            # Search for chunks with similar metadata (same section)
            section_title = summary_metadata.get('section_title', '')
            book_id = summary_metadata.get('book_id')
            
            # Query chunks from the same section
            filter_dict = {
                "type": "full",
                "book_id": book_id,
                "section_title": section_title
            }
            
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=max_chunks,
                namespace=PINECONE_NAMESPACE,
                filter=filter_dict,
                include_metadata=True
            )
            
            return results.get('matches', [])
            
        except Exception as e:
            logger.error(f"Failed to fetch chunks for summary: {e}")
            return []
    
    def retrieve(
        self,
        query: str,
        book_id: Optional[str] = None,
        use_summaries: Optional[bool] = None
    ) -> Dict:
        """
        Main retrieval method with two-stage logic
        
        Args:
            query: User query
            book_id: Optional book filter
            use_summaries: Override automatic routing (None = auto-detect)
            
        Returns:
            Dict with retrieved results and metadata
        """
        logger.info(f"🔍 Retrieving for query: '{query[:100]}...'")
        
        # Generate query embedding
        query_embedding = self._embed_query(query)
        
        # Determine retrieval strategy
        if use_summaries is None and self.config.use_query_routing:
            use_summaries = self.router.should_use_summaries(query)
        elif use_summaries is None:
            use_summaries = True  # Default to summary-first
        
        results = {
            "query": query,
            "strategy": "summary_first" if use_summaries else "direct_chunks",
            "summaries": [],
            "chunks": [],
            "total_chunks": 0
        }
        
        if use_summaries:
            # TWO-STAGE RETRIEVAL
            logger.info("📊 Using two-stage retrieval (summaries → chunks)")
            
            # Stage 1: Search summaries
            summary_matches = self._search_summaries(
                query_embedding,
                top_k=self.config.summary_top_k,
                book_id=book_id
            )
            
            results["summaries"] = summary_matches
            
            if summary_matches:
                # Stage 2: Fetch full chunks for top summaries
                all_chunks = []
                
                for i, summary_match in enumerate(summary_matches[:self.config.fetch_full_for_top_n]):
                    logger.info(f"📄 Stage 2: Fetching chunks for summary {i+1}/{self.config.fetch_full_for_top_n}")
                    
                    chunks = self._fetch_chunks_for_summary(
                        summary_match.get('metadata', {}),
                        query_embedding,
                        max_chunks=self.config.chunks_per_summary
                    )
                    
                    # Add source summary info to chunks
                    for chunk in chunks:
                        chunk['source_summary'] = {
                            'section_title': summary_match['metadata'].get('section_title'),
                            'score': summary_match.get('score')
                        }
                    
                    all_chunks.extend(chunks)
                    logger.info(f"   Retrieved {len(chunks)} chunks")
                
                results["chunks"] = all_chunks
                results["total_chunks"] = len(all_chunks)
                
                logger.info(f"✅ Two-stage retrieval complete: {len(summary_matches)} summaries → {len(all_chunks)} chunks")
            
            else:
                # Fallback: No good summaries found
                if self.config.enable_fallback:
                    logger.info("⚠️ No summaries found, falling back to direct chunk search")
                    chunk_matches = self._search_chunks(
                        query_embedding,
                        top_k=self.config.fallback_top_k,
                        book_id=book_id
                    )
                    results["chunks"] = chunk_matches
                    results["total_chunks"] = len(chunk_matches)
                    results["strategy"] = "fallback_chunks"
        
        else:
            # DIRECT CHUNK RETRIEVAL
            logger.info("📊 Using direct chunk retrieval")
            
            chunk_matches = self._search_chunks(
                query_embedding,
                top_k=self.config.fallback_top_k,
                book_id=book_id
            )
            
            results["chunks"] = chunk_matches
            results["total_chunks"] = len(chunk_matches)
        
        return results
    
    def format_context(self, retrieval_results: Dict, include_summaries: bool = True) -> str:
        """
        Format retrieval results into context string for LLM
        
        Args:
            retrieval_results: Results from retrieve()
            include_summaries: Whether to include summary text in context
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add summaries if requested
        if include_summaries and retrieval_results.get("summaries"):
            context_parts.append("## Section Overviews\n")
            for i, summary in enumerate(retrieval_results["summaries"][:3], 1):
                metadata = summary.get('metadata', {})
                section_title = metadata.get('section_title', 'Section')
                summary_text = metadata.get('summary_text', '')
                score = summary.get('score', 0)
                
                context_parts.append(f"### {i}. {section_title} (relevance: {score:.2f})")
                context_parts.append(f"{summary_text}\n")
        
        # Add full chunks
        if retrieval_results.get("chunks"):
            context_parts.append("\n## Detailed Content\n")
            for i, chunk in enumerate(retrieval_results["chunks"], 1):
                metadata = chunk.get('metadata', {})
                text = metadata.get('text', '')
                section_title = metadata.get('section_title', 'N/A')
                page = metadata.get('page_start', 'N/A')
                score = chunk.get('score', 0)
                
                context_parts.append(f"### Chunk {i} (Page {page}, Section: {section_title}, Score: {score:.2f})")
                context_parts.append(f"{text}\n")
        
        return "\n".join(context_parts)


# Convenience function
def create_two_stage_retriever(config: Optional[TwoStageConfig] = None) -> TwoStageRetriever:
    """Factory function to create retriever"""
    return TwoStageRetriever(config)

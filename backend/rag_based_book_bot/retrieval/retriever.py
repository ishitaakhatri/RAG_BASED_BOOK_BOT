"""
Unified 4-Pass Hierarchical Retriever
Integrates: Dense -> Reranking -> Multi-Hop -> Compression
"""
import os
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

try:
    from pinecone import Pinecone
    _HAS_PINECONE = True
except Exception:
    _HAS_PINECONE = False

# Import your custom modules
from rag_based_book_bot.retrieval.cross_encoder_reranker import CrossEncoderReranker
from rag_based_book_bot.retrieval.multi_hop_expander import MultiHopExpander
from rag_based_book_bot.retrieval.context_compressor import EnhancedContextCompressor

logger = logging.getLogger("hierarchical_retriever")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "coding-books")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "books_rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

@dataclass
class RetrievalConfig:
    initial_top_k: int = 50
    rerank_top_k: int = 10
    enable_multi_hop: bool = True
    multi_hop_threshold: float = 0.6
    max_context_tokens: int = 4000

class HierarchicalRetriever:
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        if not _HAS_PINECONE or not PINECONE_API_KEY:
            raise ImportError("Pinecone credentials missing")
        self.pinecone_index = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX)
        
        # Load Components
        try:
            self.reranker = CrossEncoderReranker()
            self.has_reranker = True
        except: self.has_reranker = False

        try:
            self.expander = MultiHopExpander()
            self.has_expander = True
        except: self.has_expander = False

        self.compressor = EnhancedContextCompressor(max_tokens=self.config.max_context_tokens)

    def retrieve(self, query: str, book_filter: Optional[str] = None) -> Dict:
        logger.info(f"🚀 Starting Retrieval for: '{query}'")
        
        # Helper wrapper for the Expander to call back into
        def raw_retrieve_wrapper(q_text, top_k):
            return self._dense_search_and_group(q_text, top_k, book_filter)

        # ---------------------------------------------------------
        # PASS 1: Dense Search (Vector DB)
        # ---------------------------------------------------------
        initial_candidates = raw_retrieve_wrapper(query, self.config.initial_top_k)
        
        # ---------------------------------------------------------
        # PASS 2: Reranking (Cross-Encoder)
        # ---------------------------------------------------------
        reranked_candidates = initial_candidates
        if self.has_reranker and initial_candidates:
            # Reranker expects [(text, metadata), ...]
            rerank_input = [(c['combined_text'], c) for c in initial_candidates]
            
            # Get top K
            reranked_results = self.reranker.rerank(query, rerank_input, self.config.rerank_top_k)
            
            # Unpack back to list of dicts
            reranked_candidates = []
            for _, meta, score in reranked_results:
                meta['score'] = score
                reranked_candidates.append(meta)

        # ---------------------------------------------------------
        # PASS 3: Multi-Hop Expansion (If confidence is low)
        # ---------------------------------------------------------
        final_candidates = reranked_candidates
        best_score = final_candidates[0]['score'] if final_candidates else 0
        
        if self.has_expander and self.config.enable_multi_hop and (best_score < self.config.multi_hop_threshold):
            logger.info(f"Triggering Multi-Hop (Score: {best_score:.2f})")
            
            # Use the ROBUST logic from your MultiHopExpander class
            # We pass the reranked candidates as the starting point
            final_candidates = self.expander.multi_hop_retrieve(
                query=query,
                initial_results=reranked_candidates,
                retrieval_fn=raw_retrieve_wrapper, # Callback
                max_hops=2,
                top_k_per_hop=3
            )
            
            # Re-sort after hopping
            final_candidates.sort(key=lambda x: x.get('score', 0), reverse=True)

        # ---------------------------------------------------------
        # PASS 4: Context Compression
        # ---------------------------------------------------------
        compressor_inputs = []
        for cand in final_candidates:
            # Ensure required fields exist for the compressor
            compressor_inputs.append({
                "text": cand['combined_text'],
                "metadata": {
                    "book_title": cand.get('book_title', 'Unknown'),
                    "section_title": cand.get('section_title', 'Unknown'),
                    "hierarchy_path": cand.get('hierarchy_path', ''),
                    "author": cand.get('author', ''),
                    "page_start": cand.get('chunk_index', 0)
                },
                "score": cand.get('score', 0)
            })
            
        final_context = self.compressor.compress_context(
            compressor_inputs, 
            query, 
            preserve_code=True,
            use_semantic_dedup=True
        )
        
        return {
            "context": final_context,
            "raw_results": final_candidates,
            "stats": {
                "initial_count": len(initial_candidates),
                "reranked_count": len(reranked_candidates),
                "final_count": len(final_candidates)
            }
        }

    def _dense_search_and_group(self, query: str, top_k: int, book_filter: str) -> List[Dict]:
        """
        Performs dense search and RECONSTRUCTS the section hierarchy.
        This is critical for the 'Book -> Chapter -> Section' view.
        """
        query_vec = self.embedding_model.encode(query).tolist()
        filter_dict = {"book_title": book_filter} if book_filter else {}
        
        results = self.pinecone_index.query(
            vector=query_vec, top_k=top_k, namespace=PINECONE_NAMESPACE,
            filter=filter_dict, include_metadata=True
        )
        
        # Group chunks by their Unique Section ID (Book + Hierarchy Path)
        groups = {}
        for match in results.get('matches', []):
            meta = match['metadata']
            
            # Create a unique key for the section
            h_path = meta.get('hierarchy_path', 'unknown')
            book_title = meta.get('book_title', 'unknown')
            group_key = f"{book_title}::{h_path}"
            
            if group_key not in groups:
                groups[group_key] = {
                    "group_id": group_key,
                    "book_title": book_title,
                    "section_title": meta.get('section_title'),
                    "hierarchy_path": h_path,
                    "author": meta.get('author', ''),
                    "chunks": [],
                    "max_score": match['score']
                }
            
            # Add chunk text and index to the group
            groups[group_key]['chunks'].append({
                "text": meta.get('text', ''),
                "index": int(meta.get('chunk_index', 0))
            })
            # Keep the highest score seen for this group
            groups[group_key]['max_score'] = max(groups[group_key]['max_score'], match['score'])

        # Flatten groups back into single "Section" objects
        flattened = []
        for g in groups.values():
            # Sort chunks by index to reconstruct the original text order
            g['chunks'].sort(key=lambda x: x['index'])
            combined = "\n".join([c['text'] for c in g['chunks']])
            
            flattened.append({
                "group_id": g['group_id'],
                "book_title": g['book_title'],
                "section_title": g['section_title'],
                "hierarchy_path": g['hierarchy_path'],
                "combined_text": combined,
                "score": g['max_score'],
                "author": g.get('author')
            })
            
        # Initial sort by dense score
        flattened.sort(key=lambda x: x['score'], reverse=True)
        return flattened

def create_retriever():
    return HierarchicalRetriever()
"""
Unified 4-Pass Hierarchical Retriever
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

from rag_based_book_bot.retrieval.cross_encoder_reranker import CrossEncoderReranker
from rag_based_book_bot.retrieval.multi_hop_expander import MultiHopExpander
from rag_based_book_bot.retrieval.context_compressor import EnhancedContextCompressor

logger = logging.getLogger("hierarchical_retriever")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "coding-books")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "books_rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

@dataclass
class RetrievalConfig:
    initial_top_k: int = 50
    rerank_top_k: int = 10
    enable_multi_hop: bool = True
    multi_hop_threshold: float = 0.5
    max_context_tokens: int = 4000

class HierarchicalRetriever:
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        if not _HAS_PINECONE or not PINECONE_API_KEY:
            raise ImportError("Pinecone credentials missing")
        self.pinecone_index = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX)
        
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
        
        def raw_retrieve_fn(q_text, k):
            return self._dense_search_and_group(q_text, k, book_filter)

        # Pass 1: Dense Search
        grouped = raw_retrieve_fn(query, self.config.initial_top_k)
        
        # Pass 2: Reranking
        final_candidates = grouped
        if self.has_reranker and grouped:
            rerank_input = [(g['combined_text'], g) for g in grouped]
            reranked = self.reranker.rerank(query, rerank_input, self.config.rerank_top_k)
            final_candidates = []
            for _, meta, score in reranked:
                meta['score'] = score
                final_candidates.append(meta)

        # Pass 3: Multi-Hop
        best_score = final_candidates[0]['score'] if final_candidates else 0
        if self.has_expander and self.config.enable_multi_hop and (best_score < self.config.multi_hop_threshold):
            logger.info(f"Triggering Multi-Hop (Score: {best_score:.2f})")
            context_strs = [r['combined_text'] for r in final_candidates[:3]]
            concepts = self.expander.extract_concepts(query, context_strs)
            
            existing_ids = {c['group_id'] for c in final_candidates}
            for concept in concepts:
                results = raw_retrieve_fn(concept, k=3)
                for res in results:
                    if res['group_id'] not in existing_ids:
                        final_candidates.append(res)
                        existing_ids.add(res['group_id'])
            
            final_candidates.sort(key=lambda x: x['score'], reverse=True)

        # Pass 4: Compression
        compressor_inputs = []
        for cand in final_candidates:
            compressor_inputs.append({
                "text": cand['combined_text'],
                "metadata": {
                    "book_title": cand['book_title'],
                    "section_title": cand['section_title'],
                    "hierarchy_path": cand['hierarchy_path'],
                    "score": cand['score']
                }
            })
            
        final_context = self.compressor.compress_context(compressor_inputs, query, preserve_code=True)
        
        return {
            "context": final_context,
            "raw_results": final_candidates
        }

    def _dense_search_and_group(self, query: str, top_k: int, book_filter: str) -> List[Dict]:
        query_vec = self.embedding_model.encode(query).tolist()
        filter_dict = {"book_title": book_filter} if book_filter else {}
        
        results = self.pinecone_index.query(
            vector=query_vec, top_k=top_k, namespace=PINECONE_NAMESPACE,
            filter=filter_dict, include_metadata=True
        )
        
        groups = {}
        for match in results.get('matches', []):
            meta = match['metadata']
            h_path = meta.get('hierarchy_path', 'unknown')
            group_key = f"{meta.get('book_title')}::{h_path}"
            
            if group_key not in groups:
                groups[group_key] = {
                    "group_id": group_key,
                    "book_title": meta.get('book_title'),
                    "section_title": meta.get('section_title'),
                    "hierarchy_path": h_path,
                    "chunks": [],
                    "max_score": match['score']
                }
            groups[group_key]['chunks'].append({
                "text": meta.get('text', ''),
                "index": int(meta.get('chunk_index', 0))
            })
            groups[group_key]['max_score'] = max(groups[group_key]['max_score'], match['score'])

        flattened = []
        for g in groups.values():
            g['chunks'].sort(key=lambda x: x['index'])
            combined = "\n".join([c['text'] for c in g['chunks']])
            flattened.append({
                "group_id": g['group_id'],
                "book_title": g['book_title'],
                "section_title": g['section_title'],
                "hierarchy_path": g['hierarchy_path'],
                "combined_text": combined,
                "score": g['max_score']
            })
            
        flattened.sort(key=lambda x: x['score'], reverse=True)
        return flattened

def create_retriever():
    return HierarchicalRetriever()
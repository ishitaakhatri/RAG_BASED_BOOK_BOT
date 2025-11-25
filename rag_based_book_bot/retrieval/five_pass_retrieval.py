"""
5-Pass Retrieval Pipeline for RAG Book Bot

PASS 1: Coarse Semantic Search (broad recall, top 50-100)
PASS 2: Cross-Encoder Reranking (precision boost, top 15-20)
PASS 3: Query Expansion / Multi-Hop (find cross-chapter connections)
PASS 4: Graph/Cluster Expansion (avoid chapter-name bias)
PASS 5: Compression + Deduplication (final context assembly)
"""

import os
import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from openai import OpenAI

from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RetrievalResult:
    """Single retrieval result with metadata"""
    chunk_id: str
    text: str
    score: float
    metadata: Dict
    pass_name: str = ""
    cluster_id: Optional[int] = None
    concept_tags: List[str] = field(default_factory=list)


@dataclass
class RetrievalConfig:
    """Configuration for 5-pass retrieval"""
    # Pass 1: Coarse search
    coarse_top_k: int = 80
    
    # Pass 2: Reranking
    rerank_top_k: int = 15
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Pass 3: Multi-hop
    enable_multihop: bool = True
    multihop_iterations: int = 2
    concepts_per_iteration: int = 3
    
    # Pass 4: Graph expansion
    enable_graph_expansion: bool = True
    cluster_epsilon: float = 0.3
    expand_per_cluster: int = 2
    
    # Pass 5: Compression
    final_token_budget: int = 2000
    similarity_dedup_threshold: float = 0.85
    enable_summarization: bool = True


class FivePassRetriever:
    """
    Industry-standard 5-pass retrieval pipeline
    """
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        
        # Initialize models
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.cross_encoder = CrossEncoder(self.config.rerank_model)
        
        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(os.getenv("PINECONE_INDEX_NAME", "coding-books"))
        self.namespace = os.getenv("PINECONE_NAMESPACE", "books_rag")
        
        # Initialize OpenAI for query expansion and summarization
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Tokenizer for compression
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Cache for concept graph (in production, use Redis or similar)
        self.concept_graph = defaultdict(set)
        
    def retrieve(
        self,
        query: str,
        book_filter: Optional[str] = None,
        chapter_filter: Optional[str] = None,
        user_context: Optional[str] = None
    ) -> Tuple[str, List[RetrievalResult], Dict]:
        """
        Main retrieval pipeline - runs all 5 passes
        
        Returns:
            - final_context: Compressed context string for LLM
            - all_results: All retrieved results with metadata
            - stats: Statistics about the retrieval process
        """
        stats = {
            "pass1_count": 0,
            "pass2_count": 0,
            "pass3_count": 0,
            "pass4_count": 0,
            "pass5_count": 0,
            "final_tokens": 0,
            "duplicate_removed": 0
        }
        
        print(f"\n{'='*60}")
        print(f"5-PASS RETRIEVAL PIPELINE")
        print(f"Query: {query[:80]}...")
        print(f"{'='*60}\n")
        
        # =====================================================================
        # PASS 1: COARSE SEMANTIC SEARCH (Broad Recall)
        # =====================================================================
        print(f"[PASS 1] Coarse Semantic Search (top {self.config.coarse_top_k})...")
        
        pass1_results = self._pass1_coarse_search(
            query, 
            book_filter, 
            chapter_filter
        )
        stats["pass1_count"] = len(pass1_results)
        print(f"  → Retrieved {len(pass1_results)} candidates")
        
        if not pass1_results:
            return "", [], stats
        
        # =====================================================================
        # PASS 2: CROSS-ENCODER RERANKING (Precision Boost)
        # =====================================================================
        print(f"\n[PASS 2] Cross-Encoder Reranking (top {self.config.rerank_top_k})...")
        
        pass2_results = self._pass2_rerank(query, pass1_results)
        stats["pass2_count"] = len(pass2_results)
        print(f"  → Reranked to {len(pass2_results)} high-precision results")
        
        # =====================================================================
        # PASS 3: QUERY EXPANSION / MULTI-HOP (Cross-Chapter Discovery)
        # =====================================================================
        pass3_results = pass2_results.copy()
        
        if self.config.enable_multihop:
            print(f"\n[PASS 3] Multi-Hop Retrieval ({self.config.multihop_iterations} hops)...")
            
            additional_results = self._pass3_multihop(
                query,
                pass2_results,
                book_filter,
                chapter_filter
            )
            
            # Merge with pass2 results
            existing_ids = {r.chunk_id for r in pass3_results}
            for result in additional_results:
                if result.chunk_id not in existing_ids:
                    pass3_results.append(result)
                    existing_ids.add(result.chunk_id)
            
            stats["pass3_count"] = len(pass3_results) - stats["pass2_count"]
            print(f"  → Added {stats['pass3_count']} cross-chapter results")
        
        # =====================================================================
        # PASS 4: GRAPH/CLUSTER EXPANSION (Semantic Neighborhoods)
        # =====================================================================
        pass4_results = pass3_results.copy()
        
        if self.config.enable_graph_expansion:
            print(f"\n[PASS 4] Graph/Cluster Expansion...")
            
            additional_results = self._pass4_graph_expansion(
                pass3_results,
                book_filter
            )
            
            # Merge
            existing_ids = {r.chunk_id for r in pass4_results}
            for result in additional_results:
                if result.chunk_id not in existing_ids:
                    pass4_results.append(result)
                    existing_ids.add(result.chunk_id)
            
            stats["pass4_count"] = len(pass4_results) - len(pass3_results)
            print(f"  → Added {stats['pass4_count']} cluster-expanded results")
        
        # =====================================================================
        # PASS 5: COMPRESSION + DEDUPLICATION (Final Context Assembly)
        # =====================================================================
        print(f"\n[PASS 5] Compression & Deduplication (budget: {self.config.final_token_budget} tokens)...")
        
        final_context, pass5_results, dedup_count = self._pass5_compress(
            query,
            pass4_results,
            user_context
        )
        
        stats["pass5_count"] = len(pass5_results)
        stats["duplicate_removed"] = dedup_count
        stats["final_tokens"] = len(self.tokenizer.encode(final_context))
        
        print(f"  → Final context: {stats['final_tokens']} tokens, {len(pass5_results)} chunks")
        print(f"  → Removed {dedup_count} duplicates\n")
        
        print(f"{'='*60}")
        print(f"RETRIEVAL COMPLETE")
        print(f"  Total candidates: {stats['pass1_count']}")
        print(f"  After reranking: {stats['pass2_count']}")
        print(f"  After multi-hop: {stats['pass2_count'] + stats['pass3_count']}")
        print(f"  After expansion: {len(pass4_results)}")
        print(f"  Final chunks: {stats['pass5_count']}")
        print(f"  Final tokens: {stats['final_tokens']}")
        print(f"{'='*60}\n")
        
        return final_context, pass5_results, stats
    
    # =========================================================================
    # PASS 1: COARSE SEMANTIC SEARCH
    # =========================================================================
    
    def _pass1_coarse_search(
        self,
        query: str,
        book_filter: Optional[str],
        chapter_filter: Optional[str]
    ) -> List[RetrievalResult]:
        """
        Broad semantic search for high recall
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Build filter
        filter_dict = {}
        if book_filter:
            filter_dict["book_title"] = book_filter
        if chapter_filter:
            filter_dict["chapter_numbers"] = {"$in": [chapter_filter]}
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=self.config.coarse_top_k,
            namespace=self.namespace,
            filter=filter_dict if filter_dict else None,
            include_metadata=True
        )
        
        # Convert to RetrievalResult
        retrieval_results = []
        for match in results.get("matches", []):
            retrieval_results.append(RetrievalResult(
                chunk_id=match["id"],
                text=match.get("metadata", {}).get("text", ""),
                score=match.get("score", 0.0),
                metadata=match.get("metadata", {}),
                pass_name="pass1_coarse"
            ))
        
        return retrieval_results
    
    # =========================================================================
    # PASS 2: CROSS-ENCODER RERANKING
    # =========================================================================
    
    def _pass2_rerank(
        self,
        query: str,
        candidates: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Rerank using cross-encoder for true relevance
        """
        if not candidates:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [(query, result.text) for result in candidates]
        
        # Get cross-encoder scores
        ce_scores = self.cross_encoder.predict(pairs)
        
        # Update scores
        for result, ce_score in zip(candidates, ce_scores):
            result.score = float(ce_score)
            result.pass_name = "pass2_rerank"
        
        # Sort by new scores and take top-k
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        return candidates[:self.config.rerank_top_k]
    
    # =========================================================================
    # PASS 3: MULTI-HOP RETRIEVAL
    # =========================================================================
    
    def _pass3_multihop(
        self,
        original_query: str,
        initial_results: List[RetrievalResult],
        book_filter: Optional[str],
        chapter_filter: Optional[str]
    ) -> List[RetrievalResult]:
        """
        Multi-hop retrieval to find cross-chapter connections
        """
        all_multihop_results = []
        
        for hop in range(self.config.multihop_iterations):
            print(f"    Hop {hop + 1}/{self.config.multihop_iterations}...")
            
            # Extract key concepts from top results
            concepts = self._extract_concepts(
                original_query,
                initial_results[:5]  # Use top 5 from previous hop
            )
            
            if not concepts:
                break
            
            print(f"      Extracted concepts: {', '.join(concepts[:5])}")
            
            # Search with each concept
            for concept in concepts[:self.config.concepts_per_iteration]:
                # Generate expanded query
                expanded_query = f"{original_query} {concept}"
                
                # Retrieve with expanded query
                hop_results = self._pass1_coarse_search(
                    expanded_query,
                    book_filter,
                    None  # Don't filter chapter for multi-hop
                )
                
                # Add to results
                for result in hop_results[:3]:  # Top 3 per concept
                    result.pass_name = f"pass3_multihop_hop{hop+1}"
                    result.concept_tags.append(concept)
                    all_multihop_results.append(result)
            
            # Update initial_results for next hop
            initial_results = all_multihop_results[-10:] if all_multihop_results else initial_results
        
        return all_multihop_results
    
    def _extract_concepts(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[str]:
        """
        Extract key concepts using LLM
        """
        # Combine top results
        context = "\n\n".join([r.text[:300] for r in results[:3]])
        
        prompt = f"""Given this query and context, extract 5 key technical concepts, terms, or topics that could help find related information in other chapters.

Query: {query}

Context:
{context}

Return only a comma-separated list of concepts (e.g., "neural networks, backpropagation, gradient descent, loss function, optimization")."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            
            concepts_text = response.choices[0].message.content.strip()
            concepts = [c.strip() for c in concepts_text.split(",")]
            
            return concepts[:5]
        
        except Exception as e:
            print(f"      Warning: Concept extraction failed ({e})")
            return []
    
    # =========================================================================
    # PASS 4: GRAPH/CLUSTER EXPANSION
    # =========================================================================
    
    def _pass4_graph_expansion(
        self,
        results: List[RetrievalResult],
        book_filter: Optional[str]
    ) -> List[RetrievalResult]:
        """
        Expand using semantic clustering
        """
        if len(results) < 3:
            return []
        
        # Get embeddings for all results
        embeddings = self.embedding_model.encode([r.text for r in results])
        
        # Cluster using DBSCAN
        clustering = DBSCAN(
            eps=self.config.cluster_epsilon,
            min_samples=2,
            metric="cosine"
        )
        
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Find cluster centroids
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            if label != -1:  # Ignore noise
                clusters[label].append(idx)
                results[idx].cluster_id = int(label)
        
        print(f"    Found {len(clusters)} semantic clusters")
        
        # Expand each cluster
        expansion_results = []
        
        for cluster_id, member_indices in clusters.items():
            if len(member_indices) < 2:
                continue
            
            # Get cluster members
            cluster_members = [results[i] for i in member_indices]
            
            # Compute centroid
            cluster_embeddings = embeddings[member_indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Search near centroid
            similar_results = self.index.query(
                vector=centroid.tolist(),
                top_k=self.config.expand_per_cluster + len(member_indices),
                namespace=self.namespace,
                filter={"book_title": book_filter} if book_filter else None,
                include_metadata=True
            )
            
            # Add new results not already in cluster
            existing_ids = {results[i].chunk_id for i in member_indices}
            
            for match in similar_results.get("matches", []):
                chunk_id = match["id"]
                if chunk_id not in existing_ids:
                    expansion_results.append(RetrievalResult(
                        chunk_id=chunk_id,
                        text=match.get("metadata", {}).get("text", ""),
                        score=match.get("score", 0.0),
                        metadata=match.get("metadata", {}),
                        pass_name="pass4_cluster_expansion",
                        cluster_id=cluster_id
                    ))
                    existing_ids.add(chunk_id)
                    
                    if len(expansion_results) >= self.config.expand_per_cluster * len(clusters):
                        break
        
        return expansion_results
    
    # =========================================================================
    # PASS 5: COMPRESSION & DEDUPLICATION
    # =========================================================================
    
    def _pass5_compress(
        self,
        query: str,
        results: List[RetrievalResult],
        user_context: Optional[str]
    ) -> Tuple[str, List[RetrievalResult], int]:
        """
        Final compression, deduplication, and context assembly
        """
        # Step 1: Remove duplicates by similarity
        unique_results, dedup_count = self._deduplicate_by_similarity(results)
        
        # Step 2: Sort by score
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        # Step 3: Select within token budget
        final_results = []
        total_tokens = 0
        
        for result in unique_results:
            chunk_tokens = len(self.tokenizer.encode(result.text))
            
            if total_tokens + chunk_tokens <= self.config.final_token_budget:
                final_results.append(result)
                total_tokens += chunk_tokens
            else:
                break
        
        # Step 4: Assemble final context
        final_context = self._assemble_context(query, final_results, user_context)
        
        return final_context, final_results, dedup_count
    
    def _deduplicate_by_similarity(
        self,
        results: List[RetrievalResult]
    ) -> Tuple[List[RetrievalResult], int]:
        """
        Remove near-duplicate chunks by text similarity
        """
        if not results:
            return [], 0
        
        # Compute embeddings
        texts = [r.text for r in results]
        embeddings = self.embedding_model.encode(texts)
        
        # Compute pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Greedy deduplication
        keep_mask = [True] * len(results)
        removed = 0
        
        for i in range(len(results)):
            if not keep_mask[i]:
                continue
            
            for j in range(i + 1, len(results)):
                if not keep_mask[j]:
                    continue
                
                if similarities[i, j] >= self.config.similarity_dedup_threshold:
                    # Keep the higher-scored one
                    if results[i].score >= results[j].score:
                        keep_mask[j] = False
                        removed += 1
                    else:
                        keep_mask[i] = False
                        removed += 1
                        break
        
        unique_results = [r for i, r in enumerate(results) if keep_mask[i]]
        
        return unique_results, removed
    
    def _assemble_context(
        self,
        query: str,
        results: List[RetrievalResult],
        user_context: Optional[str]
    ) -> str:
        """
        Assemble final context string
        """
        context_parts = []
        
        # Add user context if provided
        if user_context:
            context_parts.append(f"USER CONTEXT:\n{user_context}\n")
        
        # Add query
        context_parts.append(f"QUERY: {query}\n")
        
        # Group by source
        by_source = defaultdict(list)
        for result in results:
            book = result.metadata.get("book_title", "Unknown")
            chapter = result.metadata.get("chapter_numbers", [""])[0] or "N/A"
            key = f"{book} - Ch.{chapter}"
            by_source[key].append(result)
        
        # Assemble by source
        context_parts.append("\nRETRIEVED INFORMATION:\n")
        
        for source_key, source_results in by_source.items():
            context_parts.append(f"\n[SOURCE: {source_key}]")
            
            for i, result in enumerate(source_results, 1):
                page = result.metadata.get("page_start", "?")
                context_parts.append(
                    f"\n[{i}] (Page {page}, Score: {result.score:.2f})\n"
                    f"{result.text}\n"
                )
        
        return "\n".join(context_parts)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Initialize retriever
    config = RetrievalConfig(
        coarse_top_k=80,
        rerank_top_k=15,
        enable_multihop=True,
        multihop_iterations=2,
        enable_graph_expansion=True,
        final_token_budget=2000
    )
    
    retriever = FivePassRetriever(config)
    
    # Example query
    query = "How do I implement gradient descent with momentum in Python?"
    
    # Retrieve
    final_context, results, stats = retriever.retrieve(
        query=query,
        book_filter=None,  # Search all books
        chapter_filter=None
    )
    
    print("\n" + "="*60)
    print("FINAL CONTEXT (for LLM):")
    print("="*60)
    print(final_context[:1000] + "...")
    print("\n" + "="*60)
    print(f"Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("="*60)
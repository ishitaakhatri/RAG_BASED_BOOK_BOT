"""
Hierarchical Retrieval Logic
Located at: backend/rag_based_book_bot/retrieval/hierarchical_retrieval.py

Implements smart retrieval based on hierarchy:
- If asking about section → retrieve all subsections
- If asking about subsection → retrieve only that subsection
"""
from typing import List, Dict, Set
from pinecone import Index


class HierarchicalRetriever:
    """
    Handles hierarchical retrieval logic
    
    Given initial chunks, expands based on hierarchy relationships
    """
    
    def __init__(self, index: Index, namespace: str = "books_rag"):
        self.index = index
        self.namespace = namespace
    
    def expand_with_hierarchy(
        self,
        initial_chunks: List[Dict],
        expand_children: bool = True,
        expand_siblings: bool = False
    ) -> List[Dict]:
        """
        Expand initial chunks with hierarchical relationships
        
        Args:
            initial_chunks: Initial retrieved chunks with metadata
            expand_children: If True, include all children (subsections)
            expand_siblings: If True, include sibling sections
        
        Returns:
            Expanded list of chunks
        """
        if not initial_chunks:
            return []
        
        print(f"\n[Hierarchical Expansion]")
        print(f"  Input: {len(initial_chunks)} chunks")
        
        expanded_chunks = {}  # Use dict to avoid duplicates
        
        # Add initial chunks
        for chunk in initial_chunks:
            chunk_id = chunk.get('id') or chunk.get('metadata', {}).get('chunk_id')
            if chunk_id:
                expanded_chunks[chunk_id] = chunk
        
        # Expand children
        if expand_children:
            children_added = self._expand_children(initial_chunks, expanded_chunks)
            print(f"  Added {children_added} children chunks")
        
        # Expand siblings
        if expand_siblings:
            siblings_added = self._expand_siblings(initial_chunks, expanded_chunks)
            print(f"  Added {siblings_added} sibling chunks")
        
        result = list(expanded_chunks.values())
        print(f"  Output: {len(result)} total chunks")
        
        return result
    
    def _expand_children(self, initial_chunks: List[Dict], expanded_chunks: Dict) -> int:
        """
        For each chunk, retrieve all children (subsections under a section)
        """
        children_ids = set()
        
        # Collect all children IDs
        for chunk in initial_chunks:
            metadata = chunk.get('metadata', {})
            children = metadata.get('children_ids', [])
            
            if children:
                children_ids.update(children)
        
        if not children_ids:
            return 0
        
        # Retrieve children chunks
        added = 0
        for child_id in children_ids:
            if child_id in expanded_chunks:
                continue  # Already have it
            
            # Query Pinecone by chunk_id
            try:
                results = self.index.query(
                    vector=[0.0] * 384,  # Dummy vector
                    top_k=10,
                    namespace=self.namespace,
                    filter={"chunk_id": child_id},
                    include_metadata=True
                )
                
                for match in results.get('matches', []):
                    chunk_id = match.get('metadata', {}).get('chunk_id')
                    if chunk_id and chunk_id not in expanded_chunks:
                        expanded_chunks[chunk_id] = match
                        added += 1
            except Exception as e:
                print(f"    Warning: Failed to retrieve child {child_id}: {e}")
        
        return added
    
    def _expand_siblings(self, initial_chunks: List[Dict], expanded_chunks: Dict) -> int:
        """
        For each chunk, retrieve sibling chunks (same parent)
        """
        parent_ids = set()
        
        # Collect parent IDs
        for chunk in initial_chunks:
            metadata = chunk.get('metadata', {})
            parent_id = metadata.get('parent_id')
            
            if parent_id:
                parent_ids.add(parent_id)
        
        if not parent_ids:
            return 0
        
        # For each parent, get all children
        added = 0
        for parent_id in parent_ids:
            try:
                results = self.index.query(
                    vector=[0.0] * 384,
                    top_k=50,
                    namespace=self.namespace,
                    filter={"parent_id": parent_id},
                    include_metadata=True
                )
                
                for match in results.get('matches', []):
                    chunk_id = match.get('metadata', {}).get('chunk_id')
                    if chunk_id and chunk_id not in expanded_chunks:
                        expanded_chunks[chunk_id] = match
                        added += 1
            except Exception as e:
                print(f"    Warning: Failed to retrieve siblings for parent {parent_id}: {e}")
        
        return added
    
    def smart_expansion(self, initial_chunks: List[Dict], query: str) -> List[Dict]:
        """
        Smart expansion based on query analysis
        
        Rules:
        - If query is about a high-level topic → expand children
        - If query is specific → don't expand
        """
        # Simple heuristic: check query for hierarchy keywords
        query_lower = query.lower()
        
        high_level_keywords = [
            'chapter', 'overview', 'explain', 'what is', 'introduction',
            'all about', 'everything', 'complete', 'entire', 'whole'
        ]
        
        specific_keywords = [
            'example', 'code', 'implement', 'specific', 'detail',
            'exactly', 'particular', 'precise'
        ]
        
        is_high_level = any(kw in query_lower for kw in high_level_keywords)
        is_specific = any(kw in query_lower for kw in specific_keywords)
        
        # Decision
        if is_high_level and not is_specific:
            print("  → High-level query detected: expanding children")
            return self.expand_with_hierarchy(
                initial_chunks,
                expand_children=True,
                expand_siblings=False
            )
        else:
            print("  → Specific query detected: no expansion")
            return initial_chunks
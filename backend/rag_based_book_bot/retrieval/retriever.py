"""
Hierarchical Search Engine
Specialized for retrieving and reconstructing section-level context from vector fragments.
Supports multi-namespace search (Books + Research Papers).
"""
import logging
from typing import List, Dict, Optional
from collections import defaultdict

logger = logging.getLogger("hierarchical_retriever")

class HierarchicalSearchEngine:
    def __init__(self, pinecone_index, embedding_model):
        self.index = pinecone_index
        self.embedding_model = embedding_model

    def search_and_group(
        self, 
        query: str, 
        top_k: int, 
        namespaces: List[str], 
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        1. Encodes query.
        2. Searches multiple namespaces (e.g., 'books_rag', 'papers_rag').
        3. Groups fragmented chunks into complete Sections/Chapters.
        4. Returns sorted 'Super Chunks'.
        """
        try:
            query_vec = self.embedding_model.encode(query).tolist()
            
            # 1. Search all requested namespaces
            raw_matches = []
            for ns in namespaces:
                try:
                    results = self.index.query(
                        vector=query_vec,
                        top_k=top_k,
                        namespace=ns,
                        filter=metadata_filter,
                        include_metadata=True
                    )
                    # Tag matches with their source namespace
                    for m in results.get('matches', []):
                        if 'metadata' not in m:
                            m['metadata'] = {}
                        m['metadata']['_source_namespace'] = ns
                        raw_matches.append(m)
                except Exception as e:
                    logger.warning(f"Search failed for namespace '{ns}': {e}")
            
            if not raw_matches:
                return []

            # 2. Group by Section ID (Book + Hierarchy Path)
            # This stitches "Part 1", "Part 2" of "Introduction" back together.
            grouped = defaultdict(lambda: {
                "chunks": [],
                "max_score": 0.0,
                "metadata": None,
                "namespace": ""
            })

            for match in raw_matches:
                meta = match.get('metadata', {})
                
                # Robust Group Key Generation
                book_title = meta.get('book_title', 'unknown_book')
                chapter = meta.get('chapter_title', 'general')
                section = meta.get('section_title', '')
                
                # If it's a paper, use title as book_title if needed
                if not book_title and meta.get('_source_namespace') == 'papers_rag':
                    book_title = meta.get('source', 'unknown_paper')

                # Create unique ID for this logical section
                # Structure: "Book Title :: Chapter X :: Section Y"
                group_key = f"{book_title}::{chapter}::{section}"
                
                group = grouped[group_key]
                
                # Add chunk data
                group["chunks"].append({
                    "text": meta.get('text', ''),
                    "index": int(meta.get('chunk_index', 0)),
                    "id": match['id']
                })
                
                # Track metadata (from the highest scoring chunk or first found)
                if group["max_score"] < match['score']:
                    group["max_score"] = match['score']
                    group["metadata"] = meta
                    group["namespace"] = meta.get('_source_namespace')
                
                # Ensure metadata exists if it was the first chunk
                if group["metadata"] is None:
                    group["metadata"] = meta
                    group["namespace"] = meta.get('_source_namespace')

            # 3. Flatten Groups into "Section Chunks"
            structured_results = []
            
            for key, data in grouped.items():
                # Sort by original text order (chunk_index)
                data["chunks"].sort(key=lambda x: x["index"])
                
                # Merge text into one coherent block
                combined_text = "\n".join([c["text"] for c in data["chunks"]])
                meta = data["metadata"]
                
                # Construct display title
                chapter_title = meta.get('chapter_title', 'Unknown Chapter')
                section_title = meta.get('section_title', '')
                
                if section_title and section_title != chapter_title:
                    display_title = f"{chapter_title}: {section_title}"
                else:
                    display_title = chapter_title

                # Use the ID of the first chunk as the representative ID
                representative_id = data["chunks"][0]["id"]

                structured_results.append({
                    "id": representative_id,
                    "text": combined_text,
                    "score": data["max_score"],
                    "metadata": {
                        "book_title": meta.get('book_title', ''),
                        "author": meta.get('author', ''),
                        "chapter_title": chapter_title,
                        "section_title": section_title,
                        "page_number": int(meta.get('page_number', 0) or meta.get('page_start', 0)),
                        "chunk_type": meta.get('chunk_type', 'text'),
                        "namespace": data["namespace"],
                        "display_title": display_title
                    }
                })

            # 4. Final Sort by Score
            structured_results.sort(key=lambda x: x['score'], reverse=True)
            
            return structured_results[:top_k]

        except Exception as e:
            logger.error(f"Hierarchical search error: {e}")
            raise e

def create_retriever(index, model):
    return HierarchicalSearchEngine(index, model)
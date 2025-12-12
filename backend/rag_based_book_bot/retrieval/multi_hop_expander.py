"""
Multi-Hop Query Expander (PASS 3) - Updated to use LangChain

Extracts key concepts from initial retrieval results,
generates secondary queries, and retrieves additional context.

This solves the "cross-chapter information" problem.
"""

import re
from typing import List, Set, Callable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os


class MultiHopExpander:
    """
    Performs multi-hop retrieval to find related information
    across different chapters and sections.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize with Gemini LLM via LangChain for concept extraction.
        
        Args:
            api_key: Google API key (defaults to env var)
        """
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
        )
    
    def extract_concepts(
        self,
        query: str,
        initial_chunks: List[str],
        max_concepts: int = 5
    ) -> List[str]:
        """
        Extract key concepts from query and initial results.
        
        Args:
            query: Original user query
            initial_chunks: Text from top retrieved chunks
            max_concepts: Maximum concepts to extract
        
        Returns:
            List of concept strings for secondary queries
        """
        # Combine initial context
        context = "\n\n".join(initial_chunks[:3])  # Top 3 chunks only
        
        # Prompt for concept extraction
        prompt = f"""Given this query and retrieved context, extract {max_concepts} key concepts or terms that would be useful for finding related information.

Query: {query}

Context:
{context[:1500]}

Extract specific technical terms, algorithm names, related concepts, or prerequisite topics.
Return ONLY the concepts, one per line, no explanations."""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            concepts_text = response.content.strip()
            concepts = [c.strip() for c in concepts_text.split('\n') if c.strip()]
            
            # Clean and deduplicate
            concepts = self._clean_concepts(concepts, query)
            
            return concepts[:max_concepts]
            
        except Exception as e:
            print(f"Concept extraction failed: {e}")
            # Fallback: simple keyword extraction
            return self._fallback_concept_extraction(query, initial_chunks[:2])
    
    def _clean_concepts(self, concepts: List[str], original_query: str) -> List[str]:
        """Remove duplicates and concepts already in query"""
        cleaned = []
        seen = set()
        query_lower = original_query.lower()
        
        for concept in concepts:
            # Remove numbering, bullets
            concept = re.sub(r'^\d+[\.\)]\s*', '', concept)
            concept = re.sub(r'^[-*]\s*', '', concept)
            concept = concept.strip().strip('"\'')
            
            if not concept or len(concept) < 3:
                continue
            
            # Skip if already in query
            if concept.lower() in query_lower:
                continue
            
            # Deduplicate
            if concept.lower() not in seen:
                seen.add(concept.lower())
                cleaned.append(concept)
        
        return cleaned
    
    def _fallback_concept_extraction(
        self,
        query: str,
        chunks: List[str]
    ) -> List[str]:
        """Simple regex-based concept extraction as fallback"""
        # Combine text
        text = query + " " + " ".join(chunks)
        
        # Extract capitalized technical terms
        concepts = set()
        
        # Pattern 1: CamelCase or acronyms
        concepts.update(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text))
        concepts.update(re.findall(r'\b[A-Z]{2,}\b', text))
        
        # Pattern 2: Common ML terms
        ml_terms = [
            'neural network', 'gradient descent', 'backpropagation',
            'loss function', 'optimizer', 'regularization',
            'overfitting', 'underfitting', 'cross-validation',
            'feature engineering', 'hyperparameter', 'activation function'
        ]
        
        text_lower = text.lower()
        for term in ml_terms:
            if term in text_lower:
                concepts.add(term.title())
        
        return list(concepts)[:5]
    
    def multi_hop_retrieve(
        self,
        query: str,
        initial_results: List[dict],
        retrieval_fn: Callable,
        max_hops: int = 2,
        top_k_per_hop: int = 3
    ) -> List[dict]:
        """
        Perform multi-hop retrieval.
        
        Args:
            query: Original query
            initial_results: Results from first retrieval (Pass 1)
            retrieval_fn: Function to call for retrieval: fn(query_text, top_k) -> List[dict]
            max_hops: Maximum number of hops (default: 2)
            top_k_per_hop: Results per hop
        
        Returns:
            Combined list of unique chunks from all hops
        """
        all_results = list(initial_results)  # Start with initial
        seen_ids = {r.get('id', str(i)) for i, r in enumerate(initial_results)}
        
        current_chunks = [r.get('text', '') for r in initial_results[:5]]
        
        for hop in range(max_hops):
            # Extract concepts for this hop
            concepts = self.extract_concepts(query, current_chunks, max_concepts=3)
            
            if not concepts:
                break
            
            print(f"  Hop {hop + 1}: Expanding with concepts: {concepts}")
            
            # Retrieve for each concept
            hop_results = []
            for concept in concepts:
                try:
                    results = retrieval_fn(concept, top_k=top_k_per_hop)
                    hop_results.extend(results)
                except Exception as e:
                    print(f"    Retrieval failed for '{concept}': {e}")
                    continue
            
            # Deduplicate
            new_results = []
            for r in hop_results:
                # FIXED: More robust ID handling
                if 'id' in r and r['id']:
                    r_id = r['id']
                else:
                    # Generate stable ID from content hash
                    import hashlib
                    text_hash = hashlib.md5(r.get('text', '')[:500].encode()).hexdigest()[:8]
                    r_id = f"hop_{hop}_{text_hash}"
                    r['id'] = r_id  # Add ID to result
                
                if r_id not in seen_ids:
                    seen_ids.add(r_id)
                    new_results.append(r)
                    all_results.append(r)
            
            if not new_results:
                print(f"  → No new unique chunks found in hop {hop + 1}, stopping")
                break

            print(f"  → Added {len(new_results)} new unique chunks")
            
            # Prepare for next hop
            current_chunks = [r.get('text', '') for r in new_results[:3]]
        
        return all_results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    expander = MultiHopExpander()
    
    # Simulate initial results
    query = "How to implement a CNN in Keras?"
    initial_results = [
        {"id": "1", "text": "Convolutional Neural Networks (CNNs) use convolution layers...", "score": 0.9},
        {"id": "2", "text": "Keras provides Conv2D layer for building CNNs...", "score": 0.85},
    ]
    
    # Extract concepts
    concepts = expander.extract_concepts(
        query,
        [r['text'] for r in initial_results]
    )
    
    print("Extracted concepts for multi-hop:")
    for c in concepts:
        print(f"  - {c}")
"""
Lightweight Concept Graph Builder

Builds a knowledge graph during ingestion to enhance Pass 4 retrieval.
Extracts concepts, entities, and relationships from chunks.

This solves the "different chapter name" problem by linking:
- Technical terms
- Functions/classes
- Concepts
- Entities
Across chapters regardless of chapter titles.
"""

import re
import json
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, field

import spacy
from sentence_transformers import SentenceTransformer


@dataclass
class ConceptNode:
    """A concept in the knowledge graph"""
    name: str
    type: str  # "term", "function", "class", "concept", "entity"
    frequency: int = 0
    chunk_ids: Set[str] = field(default_factory=set)
    related_concepts: Set[str] = field(default_factory=set)
    importance_score: float = 0.0


@dataclass
class ConceptEdge:
    """Relationship between concepts"""
    source: str
    target: str
    relation_type: str  # "co-occurs", "uses", "inherits", "explains"
    weight: float = 1.0


class ConceptGraphBuilder:
    """
    Builds a concept knowledge graph during book ingestion.
    
    Usage:
        builder = ConceptGraphBuilder()
        
        # During ingestion, for each chunk:
        concepts = builder.extract_concepts(chunk_text, chunk_id)
        
        # After ingestion:
        graph = builder.build_graph()
        builder.save_graph(graph, "book_graph.json")
        
        # During retrieval:
        related = builder.find_related_chunks(query_concepts, graph)
    """
    
    def __init__(self):
        # Load spaCy for NER and dependency parsing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Embedding model for semantic similarity
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Accumulators
        self.concepts: Dict[str, ConceptNode] = {}
        self.co_occurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Domain-specific patterns for coding books
        self.code_patterns = {
            "function": re.compile(r'\b([a-z_][a-z0-9_]*)\s*\('),
            "class": re.compile(r'\bclass\s+([A-Z][A-Za-z0-9_]*)'),
            "import": re.compile(r'\bimport\s+([a-z_][a-z0-9_.]*)'),
            "variable": re.compile(r'\b([a-z_][a-z0-9_]*)\s*='),
        }
        
        # Technical term dictionary (expandable)
        self.technical_terms = self._load_technical_terms()
    
    def _load_technical_terms(self) -> Set[str]:
        """Load common technical terms for ML/AI/programming"""
        return {
            # ML/AI terms
            "neural network", "deep learning", "machine learning", "gradient descent",
            "backpropagation", "loss function", "optimizer", "regularization",
            "overfitting", "underfitting", "bias", "variance", "cross-validation",
            "precision", "recall", "accuracy", "f1 score", "confusion matrix",
            "feature engineering", "dimensionality reduction", "pca", "t-sne",
            "clustering", "classification", "regression", "supervised learning",
            "unsupervised learning", "reinforcement learning", "transfer learning",
            "convolution", "pooling", "activation function", "dropout", "batch normalization",
            "lstm", "rnn", "cnn", "transformer", "attention", "encoder", "decoder",
            "gan", "autoencoder", "variational", "generative", "discriminative",
            
            # Programming terms
            "function", "class", "method", "variable", "parameter", "argument",
            "loop", "iteration", "recursion", "algorithm", "data structure",
            "array", "list", "dictionary", "tuple", "set", "string",
            "integer", "float", "boolean", "object", "instance", "inheritance",
            "polymorphism", "encapsulation", "abstraction", "interface",
            "exception", "error", "debugging", "testing", "optimization",
            
            # Python specific
            "numpy", "pandas", "matplotlib", "scikit-learn", "tensorflow", "pytorch",
            "keras", "scipy", "seaborn", "jupyter", "pip", "conda",
        }
    
    def extract_concepts(
        self,
        text: str,
        chunk_id: str,
        metadata: Optional[Dict] = None
    ) -> List[ConceptNode]:
        """
        Extract concepts from a chunk of text.
        
        Returns:
            List of extracted ConceptNode objects
        """
        extracted = []
        text_lower = text.lower()
        
        # 1. Extract technical terms
        for term in self.technical_terms:
            if term in text_lower:
                self._add_or_update_concept(
                    name=term,
                    type="term",
                    chunk_id=chunk_id
                )
                extracted.append(self.concepts[term])
        
        # 2. Extract code elements
        if metadata and metadata.get("contains_code", False):
            # Functions
            for match in self.code_patterns["function"].finditer(text):
                func_name = match.group(1)
                if len(func_name) > 2:  # Skip very short names
                    self._add_or_update_concept(
                        name=func_name,
                        type="function",
                        chunk_id=chunk_id
                    )
                    extracted.append(self.concepts[func_name])
            
            # Classes
            for match in self.code_patterns["class"].finditer(text):
                class_name = match.group(1)
                self._add_or_update_concept(
                    name=class_name,
                    type="class",
                    chunk_id=chunk_id
                )
                extracted.append(self.concepts[class_name])
            
            # Imports
            for match in self.code_patterns["import"].finditer(text):
                import_name = match.group(1)
                self._add_or_update_concept(
                    name=import_name,
                    type="library",
                    chunk_id=chunk_id
                )
                extracted.append(self.concepts[import_name])
        
        # 3. Extract named entities using spaCy
        if self.nlp:
            doc = self.nlp(text[:1000])  # Limit for performance
            
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT", "TECH", "ALGORITHM"]:
                    self._add_or_update_concept(
                        name=ent.text.lower(),
                        type="entity",
                        chunk_id=chunk_id
                    )
                    extracted.append(self.concepts[ent.text.lower()])
        
        # 4. Extract key phrases (noun phrases)
        if self.nlp:
            doc = self.nlp(text[:1000])
            
            for chunk in doc.noun_chunks:
                phrase = chunk.text.lower().strip()
                
                # Only keep technical-sounding phrases
                if (len(phrase.split()) <= 3 and 
                    any(term in phrase for term in self.technical_terms)):
                    
                    self._add_or_update_concept(
                        name=phrase,
                        type="concept",
                        chunk_id=chunk_id
                    )
                    extracted.append(self.concepts[phrase])
        
        # 5. Track co-occurrences
        concept_names = [c.name for c in extracted]
        for i, c1 in enumerate(concept_names):
            for c2 in concept_names[i+1:]:
                key = tuple(sorted([c1, c2]))
                self.co_occurrence[key] += 1
        
        return extracted
    
    def _add_or_update_concept(
        self,
        name: str,
        type: str,
        chunk_id: str
    ):
        """Add a new concept or update existing"""
        if name in self.concepts:
            self.concepts[name].frequency += 1
            self.concepts[name].chunk_ids.add(chunk_id)
        else:
            self.concepts[name] = ConceptNode(
                name=name,
                type=type,
                frequency=1,
                chunk_ids={chunk_id}
            )
    
    def build_graph(self) -> Dict:
        """
        Build the final concept graph with edges.
        
        Returns:
            Graph dictionary with nodes and edges
        """
        print("Building concept graph...")
        
        # Calculate importance scores
        self._calculate_importance_scores()
        
        # Build edges from co-occurrences
        edges = []
        
        for (c1, c2), count in self.co_occurrence.items():
            if count >= 2:  # Only keep frequently co-occurring
                # Normalize weight
                f1 = self.concepts[c1].frequency
                f2 = self.concepts[c2].frequency
                weight = count / min(f1, f2)
                
                edges.append(ConceptEdge(
                    source=c1,
                    target=c2,
                    relation_type="co-occurs",
                    weight=weight
                ))
                
                # Add to related concepts
                self.concepts[c1].related_concepts.add(c2)
                self.concepts[c2].related_concepts.add(c1)
        
        # Filter nodes: keep only important ones
        important_concepts = {
            name: concept for name, concept in self.concepts.items()
            if concept.importance_score > 0.1 or concept.frequency >= 2
        }
        
        print(f"  Extracted {len(important_concepts)} concepts")
        print(f"  Built {len(edges)} edges")
        
        # Build graph dict
        graph = {
            "nodes": {
                name: {
                    "name": c.name,
                    "type": c.type,
                    "frequency": c.frequency,
                    "chunk_ids": list(c.chunk_ids),
                    "related_concepts": list(c.related_concepts),
                    "importance_score": c.importance_score
                }
                for name, c in important_concepts.items()
            },
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "type": e.relation_type,
                    "weight": e.weight
                }
                for e in edges
            ]
        }
        
        return graph
    
    def _calculate_importance_scores(self):
        """Calculate importance score for each concept (like PageRank)"""
        # Simple importance: frequency + number of connections
        max_freq = max(c.frequency for c in self.concepts.values()) if self.concepts else 1
        
        for concept in self.concepts.values():
            # Normalized frequency
            freq_score = concept.frequency / max_freq
            
            # Connection score (how many other concepts it's related to)
            connection_score = len(concept.related_concepts) / len(self.concepts) if self.concepts else 0
            
            # Combined score
            concept.importance_score = 0.6 * freq_score + 0.4 * connection_score
    
    def find_related_chunks(
        self,
        query_concepts: List[str],
        graph: Dict,
        max_hops: int = 2
    ) -> Set[str]:
        """
        Find chunks related to query concepts via graph traversal.
        
        This is used in Pass 4 retrieval.
        
        Args:
            query_concepts: List of concept names from query
            graph: Concept graph
            max_hops: Maximum hops in graph traversal
        
        Returns:
            Set of chunk IDs
        """
        related_chunks = set()
        visited_concepts = set()
        
        # BFS from query concepts
        queue = [(c, 0) for c in query_concepts if c in graph["nodes"]]
        
        while queue:
            concept_name, hop = queue.pop(0)
            
            if concept_name in visited_concepts or hop > max_hops:
                continue
            
            visited_concepts.add(concept_name)
            
            # Add chunks containing this concept
            concept_node = graph["nodes"].get(concept_name)
            if concept_node:
                related_chunks.update(concept_node["chunk_ids"])
                
                # Explore related concepts
                if hop < max_hops:
                    for related in concept_node["related_concepts"]:
                        if related not in visited_concepts:
                            queue.append((related, hop + 1))
        
        return related_chunks
    
    def save_graph(self, graph: Dict, filepath: str):
        """Save graph to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(graph, f, indent=2)
        
        print(f"Saved concept graph to {filepath}")
    
    def load_graph(self, filepath: str) -> Dict:
        """Load graph from JSON file"""
        with open(filepath, 'r') as f:
            graph = json.load(f)
        
        print(f"Loaded concept graph from {filepath}")
        print(f"  Nodes: {len(graph['nodes'])}")
        print(f"  Edges: {len(graph['edges'])}")
        
        return graph
    
    def visualize_concept_network(
        self,
        graph: Dict,
        central_concept: str,
        max_depth: int = 2
    ):
        """
        Generate a simple text visualization of concept connections.
        
        For production, use networkx + matplotlib for visual graphs.
        """
        if central_concept not in graph["nodes"]:
            print(f"Concept '{central_concept}' not found in graph")
            return
        
        print(f"\n{'='*60}")
        print(f"CONCEPT NETWORK: {central_concept}")
        print(f"{'='*60}\n")
        
        # BFS to find related concepts
        visited = set()
        queue = [(central_concept, 0)]
        
        while queue:
            concept, depth = queue.pop(0)
            
            if concept in visited or depth > max_depth:
                continue
            
            visited.add(concept)
            
            # Print concept
            indent = "  " * depth
            node = graph["nodes"][concept]
            
            print(f"{indent}{'└─' if depth > 0 else ''} {concept}")
            print(f"{indent}   Type: {node['type']}, Freq: {node['frequency']}, "
                  f"Importance: {node['importance_score']:.2f}")
            print(f"{indent}   Chunks: {len(node['chunk_ids'])}")
            
            # Add related concepts to queue
            if depth < max_depth:
                for related in node["related_concepts"][:5]:  # Limit to top 5
                    if related not in visited:
                        queue.append((related, depth + 1))


# =============================================================================
# INTEGRATION WITH INGESTION PIPELINE
# =============================================================================

def integrate_with_book_ingestion():
    """
    Shows how to integrate with book_ingestion.py
    """
    
    integration_code = '''
# Add to book_ingestion.py

from concept_graph import ConceptGraphBuilder

class BookIngestionService:
    def __init__(self):
        # ... existing init code ...
        
        # Add concept graph builder
        self.concept_builder = ConceptGraphBuilder()
    
    def embed_and_upsert(self, chunks: List[Chunk]):
        """Modified to extract concepts"""
        if not chunks:
            return
        
        # Extract embeddings
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        batch = []
        for c, emb in zip(chunks, embeddings):
            # EXTRACT CONCEPTS HERE
            concepts = self.concept_builder.extract_concepts(
                text=c.text,
                chunk_id=str(uuid.uuid4()),
                metadata={"contains_code": c.contains_code}
            )
            
            # Store concept names in metadata
            concept_names = [concept.name for concept in concepts]
            
            md = {
                # ... existing metadata ...
                "contains_code": bool(c.contains_code),
                "concepts": concept_names[:10]  # Store top 10 concepts
            }
            
            batch.append({
                "id": str(uuid.uuid4()),
                "values": emb.tolist(),
                "metadata": md
            })
            
            if len(batch) >= BATCH_SIZE:
                self.index.upsert(vectors=batch, namespace=NAMESPACE)
                batch = []
        
        if batch:
            self.index.upsert(vectors=batch, namespace=NAMESPACE)
        
        # BUILD AND SAVE GRAPH
        graph = self.concept_builder.build_graph()
        graph_path = "concept_graphs/" + self.book_id + "_graph.json"
        self.concept_builder.save_graph(graph, graph_path)
    '''
    
    return integration_code


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Build concept graph from sample text
    builder = ConceptGraphBuilder()
    
    sample_chunks = [
        {
            "id": "chunk1",
            "text": """
            Neural networks are the foundation of deep learning. They consist of layers
            of interconnected nodes that process information. The gradient descent algorithm
            is used to train these networks by minimizing the loss function.
            """,
            "metadata": {"contains_code": False}
        },
        {
            "id": "chunk2",
            "text": """
            import numpy as np
            
            def gradient_descent(X, y, learning_rate=0.01):
                weights = np.random.randn(X.shape[1])
                for i in range(100):
                    predictions = X.dot(weights)
                    loss = np.mean((predictions - y) ** 2)
                    gradients = 2 * X.T.dot(predictions - y) / len(y)
                    weights -= learning_rate * gradients
                return weights
            """,
            "metadata": {"contains_code": True}
        },
        {
            "id": "chunk3",
            "text": """
            The loss function measures how well the neural network is performing.
            Common loss functions include mean squared error for regression and
            cross-entropy for classification tasks. Backpropagation uses the chain
            rule to compute gradients of the loss with respect to the weights.
            """,
            "metadata": {"contains_code": False}
        }
    ]
    
    # Extract concepts from each chunk
    for chunk in sample_chunks:
        concepts = builder.extract_concepts(
            chunk["text"],
            chunk["id"],
            chunk["metadata"]
        )
        print(f"\nChunk {chunk['id']}: {len(concepts)} concepts")
        for concept in concepts[:5]:
            print(f"  - {concept.name} ({concept.type})")
    
    # Build graph
    graph = builder.build_graph()
    
    # Visualize
    builder.visualize_concept_network(graph, "neural network", max_depth=2)
    
    # Find related chunks
    query_concepts = ["gradient descent", "loss function"]
    related_chunks = builder.find_related_chunks(query_concepts, graph)
    print(f"\n\nRelated chunks for {query_concepts}: {related_chunks}")
"""
Semantic Cluster Manager (PASS 4)

Builds and maintains semantic clusters of chunks during ingestion.
Enables cluster-based expansion at query time.

This solves the "chapter-name bias" problem.
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import List, Dict, Optional, Set
import pickle
import os


class ClusterManager:
    """
    Manages semantic clustering of document chunks.
    
    During ingestion: Build clusters of similar chunks
    During retrieval: Expand results by retrieving cluster neighbors
    """
    
    def __init__(
        self,
        n_clusters: int = 100,
        cache_dir: str = "./cluster_cache"
    ):
        """
        Initialize cluster manager.
        
        Args:
            n_clusters: Number of semantic clusters to create
            cache_dir: Directory to cache cluster models
        """
        self.n_clusters = n_clusters
        self.cache_dir = cache_dir
        self.kmeans = None
        self.chunk_to_cluster: Dict[str, int] = {}
        self.cluster_to_chunks: Dict[int, Set[str]] = {}
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def build_clusters(
        self,
        embeddings: np.ndarray,
        chunk_ids: List[str]
    ):
        """
        Build clusters from chunk embeddings.
        
        Args:
            embeddings: Array of shape (n_chunks, embedding_dim)
            chunk_ids: List of chunk IDs corresponding to embeddings
        """
        print(f"Building {self.n_clusters} semantic clusters from {len(embeddings)} chunks...")
        
        # Use MiniBatchKMeans for efficiency with large datasets
        self.kmeans = MiniBatchKMeans(
            n_clusters=min(self.n_clusters, len(embeddings)),
            batch_size=1000,
            random_state=42
        )
        
        # Fit
        cluster_labels = self.kmeans.fit_predict(embeddings)
        
        # Build mappings
        self.chunk_to_cluster = {}
        self.cluster_to_chunks = {i: set() for i in range(self.n_clusters)}
        
        for chunk_id, cluster_id in zip(chunk_ids, cluster_labels):
            self.chunk_to_cluster[chunk_id] = int(cluster_id)
            self.cluster_to_chunks[int(cluster_id)].add(chunk_id)
        
        print(f"✅ Clustering complete. Average cluster size: {len(chunk_ids) / self.n_clusters:.1f}")
    
    def get_cluster_neighbors(
        self,
        chunk_ids: List[str],
        max_neighbors: int = 10
    ) -> List[str]:
        """
        Get neighboring chunks from same clusters.
        
        Args:
            chunk_ids: List of chunk IDs from initial retrieval
            max_neighbors: Maximum neighbors to return
        
        Returns:
            List of neighbor chunk IDs
        """
        if not self.chunk_to_cluster:
            return []
        
        # Find clusters of input chunks
        cluster_ids = set()
        for chunk_id in chunk_ids:
            if chunk_id in self.chunk_to_cluster:
                cluster_ids.add(self.chunk_to_cluster[chunk_id])
        
        # Collect neighbors from these clusters
        neighbors = set()
        for cluster_id in cluster_ids:
            neighbors.update(self.cluster_to_chunks.get(cluster_id, set()))
        
        # Remove the original chunks
        neighbors -= set(chunk_ids)
        
        # Return up to max_neighbors
        return list(neighbors)[:max_neighbors]
    
    def expand_with_clusters(
        self,
        initial_chunks: List[Dict],
        all_chunks_lookup: Dict[str, Dict],
        max_expand: int = 5
    ) -> List[Dict]:
        """
        Expand initial results with cluster neighbors.
        
        Args:
            initial_chunks: Initial retrieved chunks (with 'id' field)
            all_chunks_lookup: Lookup dict {chunk_id: full_chunk_data}
            max_expand: Maximum chunks to add
        
        Returns:
            Expanded list of chunks
        """
        # Get IDs of initial chunks
        chunk_ids = [c.get('id') for c in initial_chunks if c.get('id')]
        
        # Find neighbors
        neighbor_ids = self.get_cluster_neighbors(chunk_ids, max_neighbors=max_expand)
        
        # Lookup full data
        expanded = list(initial_chunks)  # Start with originals
        
        for neighbor_id in neighbor_ids:
            if neighbor_id in all_chunks_lookup:
                neighbor_chunk = all_chunks_lookup[neighbor_id]
                neighbor_chunk['source'] = 'cluster_expansion'
                expanded.append(neighbor_chunk)
        
        return expanded
    
    def save(self, book_id: str):
        """Save cluster model to disk"""
        if self.kmeans is None:
            return
        
        save_path = os.path.join(self.cache_dir, f"clusters_{book_id}.pkl")
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'chunk_to_cluster': self.chunk_to_cluster,
                'cluster_to_chunks': self.cluster_to_chunks,
                'n_clusters': self.n_clusters
            }, f)
        
        print(f"✅ Clusters saved to {save_path}")
    
    def load(self, book_id: str) -> bool:
        """Load cluster model from disk"""
        load_path = os.path.join(self.cache_dir, f"clusters_{book_id}.pkl")
        
        if not os.path.exists(load_path):
            return False
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.kmeans = data['kmeans']
        self.chunk_to_cluster = data['chunk_to_cluster']
        self.cluster_to_chunks = data['cluster_to_chunks']
        self.n_clusters = data['n_clusters']
        
        print(f"✅ Clusters loaded from {load_path}")
        return True
    
    def get_cluster_statistics(self) -> Dict:
        """Get statistics about clusters"""
        if not self.cluster_to_chunks:
            return {}
        
        sizes = [len(chunks) for chunks in self.cluster_to_chunks.values()]
        
        return {
            'n_clusters': self.n_clusters,
            'total_chunks': sum(sizes),
            'avg_cluster_size': np.mean(sizes),
            'min_cluster_size': min(sizes),
            'max_cluster_size': max(sizes)
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Simulate embeddings
    n_chunks = 1000
    embedding_dim = 384
    
    embeddings = np.random.rand(n_chunks, embedding_dim).astype(np.float32)
    chunk_ids = [f"chunk_{i}" for i in range(n_chunks)]
    
    # Build clusters
    manager = ClusterManager(n_clusters=50)
    manager.build_clusters(embeddings, chunk_ids)
    
    # Get stats
    stats = manager.get_cluster_statistics()
    print("\nCluster Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Find neighbors
    test_chunk_ids = ["chunk_0", "chunk_1", "chunk_2"]
    neighbors = manager.get_cluster_neighbors(test_chunk_ids, max_neighbors=10)
    print(f"\nNeighbors of {test_chunk_ids[:2]}:")
    print(f"  Found {len(neighbors)} neighbors")
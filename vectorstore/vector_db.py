"""
Vector database module for the RAG chatbot.
Manages storage and retrieval of document embeddings using FAISS.
"""

import os
import numpy as np
import faiss
import pickle
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json


class VectorDatabase:
    """
    A class to manage vector storage and retrieval using FAISS.
    Provides fast similarity search for large collections of embeddings.
    """

    def __init__(self, embedding_dim: int, index_type: str = "flat"):
        """
        Initialize the vector database.

        Args:
            embedding_dim: Dimension of the embedding vectors
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.chunks_metadata = []  # Store chunk data alongside vectors
        self.is_trained = False

        self._create_index()

    def _create_index(self):
        """Create a FAISS index based on the specified type."""
        if self.index_type == "flat":
            # Exact search, good for small to medium datasets
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product (cosine similarity)
            self.is_trained = True
        elif self.index_type == "ivf":
            # Approximate search, good for large datasets
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            self.is_trained = False  # Needs training
        elif self.index_type == "hnsw":
            # Hierarchical Navigable Small World, very fast approximate search
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 16
            self.is_trained = True
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        print(f"✓ Created {self.index_type} index with dimension {self.embedding_dim}")

    def add_embeddings(self, embeddings: np.ndarray, chunks_data: List[Dict]):
        """
        Add embeddings and their metadata to the database.

        Args:
            embeddings: Array of embedding vectors (n_vectors, embedding_dim)
            chunks_data: List of chunk metadata dictionaries
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}")

        if len(embeddings) != len(chunks_data):
            raise ValueError("Number of embeddings must match number of chunk metadata entries")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Train index if necessary
        if not self.is_trained:
            print("Training index...")
            self.index.train(embeddings)
            self.is_trained = True
            print("✓ Index trained")

        # Add embeddings to index
        start_id = len(self.chunks_metadata)
        self.index.add(embeddings)

        # Store metadata with IDs
        for i, chunk_data in enumerate(chunks_data):
            chunk_with_id = chunk_data.copy()
            chunk_with_id['vector_id'] = start_id + i
            self.chunks_metadata.append(chunk_with_id)

        print(f"✓ Added {len(embeddings)} embeddings to database")
        print(f"  Total vectors in database: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for similar embeddings.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return

        Returns:
            List of chunks with similarity scores
        """
        if self.index.ntotal == 0:
            return []

        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            chunk_data = self.chunks_metadata[idx].copy()
            chunk_data['similarity_score'] = float(score)
            results.append(chunk_data)

        return results

    def save(self, directory_path: str, name: str = "vector_db"):
        """
        Save the vector database to disk.

        Args:
            directory_path: Directory to save the database
            name: Name prefix for the saved files
        """
        directory_path = Path(directory_path)
        directory_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = directory_path / f"{name}.faiss"
        faiss.write_index(self.index, str(index_path))

        # Save metadata
        metadata_path = directory_path / f"{name}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'chunks_metadata': self.chunks_metadata,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'is_trained': self.is_trained
            }, f)

        # Save config for easy loading
        config_path = directory_path / f"{name}_config.json"
        config = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'total_vectors': self.index.ntotal,
            'is_trained': self.is_trained
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✓ Vector database saved to {directory_path}")
        print(f"  Index file: {index_path}")
        print(f"  Metadata file: {metadata_path}")
        print(f"  Config file: {config_path}")

    @classmethod
    def load(cls, directory_path: str, name: str = "vector_db") -> 'VectorDatabase':
        """
        Load a vector database from disk.

        Args:
            directory_path: Directory containing the database files
            name: Name prefix of the saved files

        Returns:
            Loaded VectorDatabase instance
        """
        directory_path = Path(directory_path)

        # Load config
        config_path = directory_path / f"{name}_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Create instance
        db = cls(config['embedding_dim'], config['index_type'])

        # Load FAISS index
        index_path = directory_path / f"{name}.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        db.index = faiss.read_index(str(index_path))
        db.is_trained = config['is_trained']

        # Load metadata
        metadata_path = directory_path / f"{name}_metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            db.chunks_metadata = metadata['chunks_metadata']

        print(f"✓ Vector database loaded from {directory_path}")
        print(f"  Total vectors: {db.index.ntotal}")
        print(f"  Index type: {db.index_type}")
        print(f"  Embedding dimension: {db.embedding_dim}")

        return db

    def get_stats(self) -> Dict:
        """Get database statistics."""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.embedding_dim,
            'index_type': self.index_type,
            'is_trained': self.is_trained,
            'total_chunks': len(self.chunks_metadata)
        }

    def clear(self):
        """Clear all data from the database."""
        self._create_index()
        self.chunks_metadata = []
        print("✓ Database cleared")


def build_vector_database_from_embeddings(embedded_chunks: List[Dict],
                                          embedding_dim: int,
                                          index_type: str = "flat") -> VectorDatabase:
    """
    Convenience function to build a vector database from embedded chunks.

    Args:
        embedded_chunks: List of chunks with embeddings
        embedding_dim: Dimension of embeddings
        index_type: Type of FAISS index to use

    Returns:
        Populated VectorDatabase instance
    """
    # Create database
    db = VectorDatabase(embedding_dim, index_type)

    # Extract embeddings and metadata
    embeddings = np.array([chunk['embedding'] for chunk in embedded_chunks]).astype(np.float32)

    # Remove embeddings from metadata to avoid duplication
    chunks_metadata = []
    for chunk in embedded_chunks:
        metadata = chunk.copy()
        if 'embedding' in metadata:
            del metadata['embedding']  # Don't store embeddings twice
        chunks_metadata.append(metadata)

    # Add to database
    db.add_embeddings(embeddings, chunks_metadata)

    return db


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Vector Database...")

    # Create some test embeddings
    embedding_dim = 384
    n_vectors = 100

    # Generate random embeddings for testing
    np.random.seed(42)
    test_embeddings = np.random.random((n_vectors, embedding_dim)).astype(np.float32)

    # Create test metadata
    test_metadata = []
    for i in range(n_vectors):
        test_metadata.append({
            'id': f'test_chunk_{i}',
            'text': f'This is test chunk number {i} with some sample content.',
            'document_id': f'doc_{i // 10}',
            'chunk_index': i % 10
        })

    # Test different index types
    for index_type in ["flat", "hnsw"]:
        print(f"\n🔍 Testing {index_type} index...")

        # Create database
        db = VectorDatabase(embedding_dim, index_type)

        # Add embeddings
        db.add_embeddings(test_embeddings, test_metadata)

        # Test search
        query_embedding = np.random.random(embedding_dim).astype(np.float32)
        results = db.search(query_embedding, top_k=5)

        print(f"✓ Search returned {len(results)} results")
        for i, result in enumerate(results):
            print(f"  {i + 1}. Score: {result['similarity_score']:.3f}, ID: {result['id']}")

        # Test save/load
        save_dir = f"test_vector_db_{index_type}"
        db.save(save_dir)

        # Load and test
        loaded_db = VectorDatabase.load(save_dir)
        loaded_results = loaded_db.search(query_embedding, top_k=3)

        print(f"✓ Loaded database returned {len(loaded_results)} results")

        # Print stats
        stats = loaded_db.get_stats()
        print(f"✓ Database stats: {stats}")

    print("\n✅ Vector database test completed!")
"""
Test script to upgrade from simple similarity search to FAISS vector database.
This will use your existing embeddings and show the performance improvement.
"""

import time
import numpy as np
from pathlib import Path

from embeddings.embedding_generator import EmbeddingGenerator
from vectorstore.vector_db import VectorDatabase, build_vector_database_from_embeddings

def compare_search_methods():
    """Compare simple similarity search vs FAISS vector database."""
    print("🔥 Comparing Search Methods: Simple vs FAISS...")

    # Load existing embeddings
    try:
        embedder = EmbeddingGenerator(model_type="sentence-bert")
        embedded_chunks = embedder.load_embeddings("embeddings/document_embeddings.pkl")
        print(f"✓ Loaded {len(embedded_chunks)} embedded chunks")
    except FileNotFoundError:
        print("❌ No embeddings found. Run test_embeddings.py first!")
        return

    # Test queries
    test_queries = [
        "What is RAG and how does it work?",
        "How to add new documents to the system?",
        "What file formats are supported?",
        "How accurate is the RAG system?",
        "Benefits of using vector databases"
    ]

    print(f"\nTesting with {len(test_queries)} queries on {len(embedded_chunks)} chunks...")

    # Method 1: Simple similarity search (your current method)
    print("\n🐌 Method 1: Simple Similarity Search")
    start_time = time.time()

    simple_results = {}
    for query in test_queries:
        results = embedder.similarity_search(query, embedded_chunks, top_k=3)
        simple_results[query] = results

    simple_time = time.time() - start_time
    print(f"   Time taken: {simple_time:.4f} seconds")

    # Method 2: FAISS Vector Database
    print("\n🚀 Method 2: FAISS Vector Database")
    start_time = time.time()

    # Build vector database
    embedding_dim = embedded_chunks[0]['embedding'].shape[0]
    vector_db = build_vector_database_from_embeddings(embedded_chunks, embedding_dim, "flat")

    faiss_results = {}
    for query in test_queries:
        query_embedding = embedder.generate_embedding(query)
        results = vector_db.search(query_embedding, top_k=3)
        faiss_results[query] = results

    faiss_time = time.time() - start_time
    print(f"   Time taken: {faiss_time:.4f} seconds")

    # Compare results
    print(f"\n📊 Performance Comparison:")
    print(f"   Simple search: {simple_time:.4f}s")
    print(f"   FAISS search:  {faiss_time:.4f}s")
    print(f"   Speedup: {simple_time/faiss_time:.2f}x faster")

    # Show that results are similar
    print(f"\n🔍 Results Comparison (first query):")
    query = test_queries[0]
    print(f"   Query: '{query}'")

    print(f"\n   Simple Search Results:")
    for i, result in enumerate(simple_results[query]):
        print(f"   {i+1}. Score: {result['similarity_score']:.3f}")
        print(f"      Text: {result['text'][:80]}...")

    print(f"\n   FAISS Search Results:")
    for i, result in enumerate(faiss_results[query]):
        print(f"   {i+1}. Score: {result['similarity_score']:.3f}")
        print(f"      Text: {result['text'][:80]}...")

    # Save the vector database for future use
    print(f"\n💾 Saving FAISS Vector Database...")
    vectorstore_dir = Path("vectorstore")
    vectorstore_dir.mkdir(exist_ok=True)

    vector_db.save("vectorstore", "document_vectors")

    return vector_db

def test_vector_database_features():
    """Test advanced features of the vector database."""
    print("\n🧪 Testing Vector Database Features...")

    # Load the saved vector database
    try:
        vector_db = VectorDatabase.load("vectorstore", "document_vectors")
        print("✓ Loaded vector database from disk")
    except FileNotFoundError:
        print("❌ Vector database not found. Run compare_search_methods() first!")
        return

    # Test database stats
    stats = vector_db.get_stats()
    print(f"\n📊 Database Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Test different search sizes
    embedder = EmbeddingGenerator(model_type="sentence-bert")
    query = "What is artificial intelligence?"
    query_embedding = embedder.generate_embedding(query)

    print(f"\n🔍 Testing Different Search Sizes:")
    print(f"   Query: '{query}'")

    for k in [1, 3, 5]:
        results = vector_db.search(query_embedding, top_k=k)
        print(f"\n   Top {k} results:")
        for i, result in enumerate(results):
            print(f"   {i+1}. Score: {result['similarity_score']:.3f}")
            print(f"      From: {result.get('document_id', 'unknown')}")
            print(f"      Text: {result['text'][:60]}...")

def demonstrate_scalability():
    """Show how FAISS scales with larger datasets."""
    print("\n⚡ Demonstrating Scalability...")

    # Create progressively larger datasets
    embedding_dim = 384
    dataset_sizes = [10, 100, 1000]

    for size in dataset_sizes:
        print(f"\n📈 Testing with {size} vectors...")

        # Generate random embeddings
        np.random.seed(42)
        embeddings = np.random.random((size, embedding_dim)).astype(np.float32)

        # Create metadata
        metadata = []
        for i in range(size):
            metadata.append({
                'id': f'test_{i}',
                'text': f'Test document {i} with sample content.',
                'document_id': f'doc_{i//10}'
            })

        # Test different index types (skip HNSW on Mac due to compatibility issues)
        for index_type in ["flat"]:
            db = VectorDatabase(embedding_dim, index_type)

            # Time the addition
            start_time = time.time()
            db.add_embeddings(embeddings, metadata)
            add_time = time.time() - start_time

            # Time the search
            query_embedding = np.random.random(embedding_dim).astype(np.float32)
            start_time = time.time()
            results = db.search(query_embedding, top_k=5)
            search_time = time.time() - start_time

            print(f"   {index_type:>4} index: Add {add_time:.4f}s, Search {search_time:.6f}s")

if __name__ == "__main__":
    try:
        # Run the comparison
        vector_db = compare_search_methods()

        if vector_db:
            # Test additional features
            test_vector_database_features()

            # Show scalability
            demonstrate_scalability()

            print("\n🎉 Vector database upgrade completed!")
            print("\n💡 Key Benefits:")
            print("   ✓ Faster search performance")
            print("   ✓ Better memory efficiency")
            print("   ✓ Scalable to millions of vectors")
            print("   ✓ Persistent storage")
            print("   ✓ Multiple index types for different use cases")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
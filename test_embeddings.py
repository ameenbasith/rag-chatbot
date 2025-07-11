"""
Test script for the complete document processing + embedding pipeline.
This combines document chunking with embedding generation.
"""

import os
from pathlib import Path
from ingestion.document_processor import DocumentProcessor
from embeddings.embedding_generator import EmbeddingGenerator


def test_complete_pipeline():
    """Test the complete pipeline from documents to embeddings."""
    print("🚀 Testing Complete RAG Pipeline...")

    # Step 1: Process documents (we already have sample docs from previous test)
    print("\n📄 Step 1: Processing Documents...")
    processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)

    data_dir = Path("data")
    if not data_dir.exists() or not list(data_dir.glob("*.txt")):
        print("❌ No sample documents found. Run test_processor.py first!")
        return

    # Process all documents
    all_chunks = []
    for file_path in data_dir.glob("*.txt"):
        print(f"   Processing {file_path.name}...")
        doc_data = processor.load_document(file_path)
        chunks = processor.chunk_text(doc_data['content'], doc_data['filename'])
        all_chunks.extend(chunks)

    print(f"   ✓ Total chunks created: {len(all_chunks)}")

    # Step 2: Generate embeddings
    print("\n🧠 Step 2: Generating Embeddings...")
    embedder = EmbeddingGenerator(model_type="sentence-bert")

    # This might take a moment as it downloads the model on first run
    embedded_chunks = embedder.embed_document_chunks(all_chunks)

    # Step 3: Save embeddings
    print("\n💾 Step 3: Saving Embeddings...")
    embeddings_dir = Path("embeddings")
    embeddings_dir.mkdir(exist_ok=True)

    embedder.save_embeddings(embedded_chunks, "embeddings/document_embeddings.pkl")

    # Step 4: Test semantic search
    print("\n🔍 Step 4: Testing Semantic Search...")

    test_queries = [
        "What is RAG?",
        "How do I add new documents?",
        "What file formats are supported?",
        "How accurate is the system?"
    ]

    for query in test_queries:
        print(f"\n   Query: '{query}'")
        results = embedder.similarity_search(query, embedded_chunks, top_k=3)

        for i, result in enumerate(results):
            print(f"   {i + 1}. Score: {result['similarity_score']:.3f}")
            print(f"      From: {result['document_id']}")
            print(f"      Text: {result['text'][:100]}...")

    # Step 5: Test loading embeddings
    print("\n📂 Step 5: Testing Loading Embeddings...")
    loaded_chunks = embedder.load_embeddings("embeddings/document_embeddings.pkl")
    print(f"   ✓ Successfully loaded {len(loaded_chunks)} embedded chunks")

    print("\n🎉 Complete pipeline test successful!")
    print(f"   📊 Summary:")
    print(f"   - Documents processed: {len(set(chunk['document_id'] for chunk in all_chunks))}")
    print(f"   - Total chunks: {len(all_chunks)}")
    print(f"   - Embedding dimension: {embedder.embedding_dim}")
    print(f"   - Model used: {embedder.model_type}:{embedder.model_name}")


def demonstrate_semantic_power():
    """Show how semantic search is different from keyword search."""
    print("\n🪄 Demonstrating Semantic Search Power...")

    # Load existing embeddings
    try:
        embedder = EmbeddingGenerator(model_type="sentence-bert")
        embedded_chunks = embedder.load_embeddings("embeddings/document_embeddings.pkl")
    except FileNotFoundError:
        print("   ❌ No embeddings found. Run the complete pipeline test first!")
        return

    # These queries test semantic understanding vs keyword matching
    semantic_tests = [
        {
            'query': 'machine intelligence',  # No exact match, but should find AI content
            'description': 'Query uses "machine intelligence" - should find AI/ML content'
        },
        {
            'query': 'adding documents',  # Should find content about "add new documents"
            'description': 'Query uses "adding" - should find "add new documents" content'
        },
        {
            'query': 'precision of answers',  # Should find accuracy content
            'description': 'Query asks about "precision" - should find "accuracy" content'
        }
    ]

    for test in semantic_tests:
        print(f"\n   🔍 {test['description']}")
        print(f"       Query: '{test['query']}'")

        results = embedder.similarity_search(test['query'], embedded_chunks, top_k=2)

        for i, result in enumerate(results):
            print(f"       {i + 1}. Score: {result['similarity_score']:.3f}")
            print(f"          Text: {result['text'][:120]}...")


if __name__ == "__main__":
    try:
        test_complete_pipeline()
        demonstrate_semantic_power()
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("   - Make sure you ran test_processor.py first")
        print("   - Check if you have enough memory (close other apps)")
        print("   - The model download might take a few minutes on first run")
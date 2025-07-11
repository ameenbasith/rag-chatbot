"""
Test script for the document processor.
Run this to make sure your document processing is working correctly.
"""

import os
from pathlib import Path
from ingestion.document_processor import DocumentProcessor


def create_sample_documents():
    """Create some sample documents for testing."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Create a sample text file
    sample_txt = """
    Welcome to the RAG Chatbot Documentation

    This is a comprehensive guide to understanding and using our Retrieval-Augmented Generation system.

    What is RAG?
    RAG stands for Retrieval-Augmented Generation. It's a powerful AI technique that combines:
    - Document retrieval systems
    - Large language models
    - Vector databases for semantic search

    How does it work?
    1. Documents are processed and split into chunks
    2. Each chunk is converted to a vector representation (embedding)
    3. When a user asks a question, we find the most relevant chunks
    4. These chunks are used as context for the AI to generate an answer

    Benefits of RAG:
    - Reduces hallucination in AI responses
    - Allows AI to work with specific, up-to-date information
    - Provides transparency by showing source material
    - Can be updated with new information without retraining

    This system is particularly useful for:
    - Customer support with product documentation
    - Internal knowledge management
    - Research assistance
    - Educational applications
    """

    with open(data_dir / "sample_guide.txt", "w") as f:
        f.write(sample_txt)

    # Create another sample file
    sample_faq = """
    Frequently Asked Questions

    Q: How accurate is the RAG system?
    A: The accuracy depends on the quality of your source documents. Since RAG grounds its responses in real content, it's typically much more accurate than standalone language models.

    Q: What file formats are supported?
    A: Currently, we support PDF, DOCX, TXT, and Markdown files. More formats can be added as needed.

    Q: How do I add new documents?
    A: Simply place your documents in the data folder and restart the system. The ingestion process will automatically detect and process new files.

    Q: Can I use this system offline?
    A: The document processing and retrieval can work offline, but you'll need an internet connection for the language model API calls unless you use a local model.
    """

    with open(data_dir / "faq.txt", "w") as f:
        f.write(sample_faq)

    print("✓ Sample documents created in data/ folder")


def test_document_processor():
    """Test the document processor with sample files."""
    print("🧪 Testing Document Processor...")

    # Create sample documents
    create_sample_documents()

    # Initialize processor
    processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)

    # Test processing individual files
    data_dir = Path("data")
    for file_path in data_dir.glob("*.txt"):
        print(f"\n📄 Processing {file_path.name}...")

        try:
            # Load document
            doc_data = processor.load_document(file_path)
            print(f"   Document size: {doc_data['size']} characters")

            # Create chunks
            chunks = processor.chunk_text(doc_data['content'], doc_data['filename'])
            print(f"   Created {len(chunks)} chunks")

            # Show first chunk as example
            if chunks:
                print(f"   First chunk preview: {chunks[0]['text'][:100]}...")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

    # Test processing entire directory
    print(f"\n📁 Processing entire directory...")
    try:
        all_docs = processor.process_directory("data")
        total_chunks = sum(len(doc['chunks']) for doc in all_docs)
        print(f"   Processed {len(all_docs)} documents with {total_chunks} total chunks")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")

    print("\n✅ Document processor test completed!")


if __name__ == "__main__":
    test_document_processor()
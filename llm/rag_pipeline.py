"""
Complete RAG pipeline that combines document retrieval with LLM generation.
This is the core orchestrator that brings everything together.
"""

import os
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import time

# Import our modules
from embeddings.embedding_generator import EmbeddingGenerator
from vectorstore.vector_db import VectorDatabase

# LLM integration
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""
    answer: str
    source_chunks: List[Dict]
    query: str
    confidence: float
    processing_time: float
    model_used: str


class RAGPipeline:
    """
    Complete RAG pipeline that combines retrieval and generation.
    """

    def __init__(self,
                 vector_db_path: str = "vectorstore",
                 vector_db_name: str = "document_vectors",
                 embedding_model_type: str = "sentence-bert",
                 llm_model: str = "gpt-3.5-turbo",
                 temperature: float = 0.1,
                 max_context_chunks: int = 5):
        """
        Initialize the RAG pipeline.

        Args:
            vector_db_path: Path to the vector database
            vector_db_name: Name of the vector database files
            embedding_model_type: Type of embedding model to use
            llm_model: LLM model name for generation
            temperature: Temperature for LLM generation (0.0 = deterministic)
            max_context_chunks: Maximum number of chunks to use as context
        """
        self.vector_db_path = vector_db_path
        self.vector_db_name = vector_db_name
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_context_chunks = max_context_chunks

        # Initialize components
        self.embedder = None
        self.vector_db = None
        self.openai_client = None

        self._initialize_components(embedding_model_type)

    def _initialize_components(self, embedding_model_type: str):
        """Initialize all pipeline components."""
        print("🚀 Initializing RAG pipeline...")

        # Initialize embedding generator
        print("   Loading embedding model...")
        self.embedder = EmbeddingGenerator(model_type=embedding_model_type)
        print("   ✓ Embedding model loaded")

        # Load vector database
        print("   Loading vector database...")
        try:
            self.vector_db = VectorDatabase.load(self.vector_db_path, self.vector_db_name)
            print("   ✓ Vector database loaded")
        except FileNotFoundError:
            print("   ❌ Vector database not found!")
            print("   💡 Run the embedding and vector database setup first")
            raise

        # Initialize OpenAI client
        print("   Setting up LLM connection...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("   ❌ OPENAI_API_KEY not found in environment variables")
            print("   💡 Add your OpenAI API key to the .env file")
            raise ValueError("OpenAI API key required for LLM generation")

        openai.api_key = api_key
        self.openai_client = openai
        print("   ✓ LLM connection ready")

        print("✅ RAG pipeline initialized successfully!")

    def _retrieve_relevant_chunks(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve relevant document chunks for the query."""
        top_k = top_k or self.max_context_chunks

        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(query)

        # Search vector database
        relevant_chunks = self.vector_db.search(query_embedding, top_k=top_k)

        return relevant_chunks

    def _build_context_prompt(self, query: str, chunks: List[Dict]) -> str:
        """Build the context prompt for the LLM."""
        if not chunks:
            return f"""
You are a helpful AI assistant. A user has asked a question, but no relevant context was found in the knowledge base.

Question: {query}

Please politely inform the user that you don't have specific information about their question in the available documents, but offer to help in other ways if possible.
"""

        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get('document_id', 'Unknown source')
            text = chunk.get('text', '')
            score = chunk.get('similarity_score', 0)

            context_parts.append(f"[Source {i}: {source} (relevance: {score:.2f})]")
            context_parts.append(text)
            context_parts.append("")  # Empty line between chunks

        context = "\n".join(context_parts)

        prompt = f"""You are a helpful AI assistant with access to a knowledge base. Use the provided context to answer the user's question accurately and comprehensively.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, say so clearly
- Be specific and cite which sources support your answer when possible
- If you're uncertain about something, express that uncertainty
- Keep your answer helpful and conversational
- If the context contains contradictory information, acknowledge it

ANSWER:"""

        return prompt

    def _generate_answer(self, prompt: str) -> str:
        """Generate an answer using the LLM."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant that answers questions based on provided context from documents."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=1000
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"

    def _calculate_confidence(self, chunks: List[Dict]) -> float:
        """Calculate confidence score based on retrieval results."""
        if not chunks:
            return 0.0

        # Simple confidence based on top similarity score and number of relevant chunks
        top_score = chunks[0].get('similarity_score', 0) if chunks else 0

        # Confidence factors:
        # - Higher top similarity score = higher confidence
        # - More chunks with decent scores = higher confidence
        relevant_chunks = len([c for c in chunks if c.get('similarity_score', 0) > 0.3])

        confidence = min(1.0, top_score + (relevant_chunks * 0.1))
        return confidence

    def query(self, question: str, top_k: Optional[int] = None, return_sources: bool = True) -> RAGResponse:
        """
        Main method to query the RAG system.

        Args:
            question: User's question
            top_k: Number of chunks to retrieve (uses default if None)
            return_sources: Whether to include source chunks in response

        Returns:
            RAGResponse with answer and metadata
        """
        start_time = time.time()

        print(f"🔍 Processing query: '{question}'")

        # Step 1: Retrieve relevant chunks
        print("   📚 Retrieving relevant documents...")
        relevant_chunks = self._retrieve_relevant_chunks(question, top_k)
        print(f"   ✓ Found {len(relevant_chunks)} relevant chunks")

        # Step 2: Build context prompt
        print("   📝 Building context prompt...")
        prompt = self._build_context_prompt(question, relevant_chunks)

        # Step 3: Generate answer
        print("   🤖 Generating answer...")
        answer = self._generate_answer(prompt)
        print("   ✓ Answer generated")

        # Step 4: Calculate confidence and prepare response
        confidence = self._calculate_confidence(relevant_chunks)
        processing_time = time.time() - start_time

        response = RAGResponse(
            answer=answer,
            source_chunks=relevant_chunks if return_sources else [],
            query=question,
            confidence=confidence,
            processing_time=processing_time,
            model_used=self.llm_model
        )

        print(f"   ⚡ Query processed in {processing_time:.2f}s")
        print(f"   📊 Confidence: {confidence:.2f}")

        return response

    def chat(self):
        """Interactive chat interface."""
        print("\n🤖 RAG Chatbot Ready!")
        print("Ask me anything about your documents. Type 'quit' to exit.")
        print("=" * 50)

        while True:
            try:
                # Get user input
                question = input("\n🧑 You: ").strip()

                if question.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break

                if not question:
                    continue

                # Process the query
                response = self.query(question)

                # Display the answer
                print(f"\n🤖 Assistant: {response.answer}")

                # Show sources if confidence is reasonable
                if response.confidence > 0.2 and response.source_chunks:
                    print(f"\n📚 Sources (confidence: {response.confidence:.2f}):")
                    for i, chunk in enumerate(response.source_chunks[:3], 1):
                        source = chunk.get('document_id', 'Unknown')
                        score = chunk.get('similarity_score', 0)
                        print(f"   {i}. {source} (relevance: {score:.2f})")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        db_stats = self.vector_db.get_stats() if self.vector_db else {}

        return {
            'vector_database': db_stats,
            'embedding_model': f"{self.embedder.model_type}:{self.embedder.model_name}",
            'llm_model': self.llm_model,
            'max_context_chunks': self.max_context_chunks,
            'temperature': self.temperature
        }


# Utility functions for easy setup
def create_rag_pipeline_from_documents(doc_directory: str,
                                       output_dir: str = "vectorstore",
                                       embedding_model: str = "sentence-bert") -> RAGPipeline:
    """
    Convenience function to create a complete RAG pipeline from a directory of documents.

    Args:
        doc_directory: Directory containing documents to process
        output_dir: Directory to save the vector database
        embedding_model: Embedding model type to use

    Returns:
        Initialized RAG pipeline
    """
    from ingestion.document_processor import DocumentProcessor
    from vectorstore.vector_db import build_vector_database_from_embeddings

    print("🏗️ Building RAG pipeline from documents...")

    # Process documents
    processor = DocumentProcessor()
    processed_docs = processor.process_directory(doc_directory)

    # Flatten chunks
    all_chunks = []
    for doc in processed_docs:
        all_chunks.extend(doc['chunks'])

    # Generate embeddings
    embedder = EmbeddingGenerator(model_type=embedding_model)
    embedded_chunks = embedder.embed_document_chunks(all_chunks)

    # Build vector database
    vector_db = build_vector_database_from_embeddings(
        embedded_chunks,
        embedder.embedding_dim,
        "flat"
    )

    # Save vector database
    vector_db.save(output_dir, "document_vectors")

    # Create and return pipeline
    return RAGPipeline(vector_db_path=output_dir)


# Example usage and testing
if __name__ == "__main__":
    try:
        # Initialize RAG pipeline
        rag = RAGPipeline()

        # Test queries
        test_queries = [
            "What is RAG and how does it work?",
            "What file formats are supported?",
            "How do I add new documents?",
            "What are the benefits of using this system?"
        ]

        print("\n🧪 Testing RAG Pipeline...")

        for query in test_queries:
            print(f"\n" + "=" * 60)
            response = rag.query(query)

            print(f"❓ Question: {response.query}")
            print(f"🤖 Answer: {response.answer}")
            print(f"📊 Confidence: {response.confidence:.2f}")
            print(f"⚡ Time: {response.processing_time:.2f}s")

        # Show pipeline stats
        print(f"\n📊 Pipeline Statistics:")
        stats = rag.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")

    except Exception as e:
        print(f"❌ Pipeline test failed: {str(e)}")
        print("\n💡 Make sure you have:")
        print("   1. OPENAI_API_KEY in your .env file")
        print("   2. Vector database created (run test_vector_upgrade.py)")
        print("   3. All dependencies installed")
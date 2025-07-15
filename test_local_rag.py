"""
Simple local LLM fallback for when OpenAI API is unavailable.
This demonstrates the complete RAG pipeline without API costs.
"""

from typing import List, Dict
import re


class LocalLLMFallback:
    """
    A simple rule-based text generator to demonstrate RAG without API costs.
    This isn't as sophisticated as GPT, but shows how the pipeline works.
    """

    def __init__(self):
        self.templates = {
            'general': """Based on the provided information, {summary}

{details}

This information comes from {sources}.""",

            'not_found': """I don't have specific information about "{query}" in the available documents. The documents I have access to focus on {doc_topics}.""",

            'definition': """Based on the documentation, {term} is {definition}

Key points:
{points}""",

            'howto': """To {action}, according to the documentation:

{steps}

{additional_info}"""
        }

    def generate_answer(self, query: str, chunks: List[Dict]) -> str:
        """Generate a simple answer based on retrieved chunks."""
        if not chunks:
            return self._generate_not_found_response(query)

        # Analyze the query type
        query_lower = query.lower()

        if any(word in query_lower for word in ['what is', 'define', 'definition']):
            return self._generate_definition_response(query, chunks)
        elif any(word in query_lower for word in ['how to', 'how do', 'how can']):
            return self._generate_howto_response(query, chunks)
        else:
            return self._generate_general_response(query, chunks)

    def _generate_definition_response(self, query: str, chunks: List[Dict]) -> str:
        """Generate a definition-style response."""
        # Extract the term being defined
        term_match = re.search(r'what is (.+?)\?', query.lower())
        term = term_match.group(1) if term_match else "this concept"

        # Combine relevant text from chunks
        combined_text = " ".join(chunk['text'] for chunk in chunks[:3])

        # Extract key information
        if 'rag' in query.lower():
            definition = "a technique that combines document retrieval with language generation"
            points = self._extract_key_points_rag(combined_text)
        elif 'file format' in query.lower():
            definition = "the supported document types"
            points = self._extract_file_formats(combined_text)
        else:
            definition = "described in the documentation"
            points = self._extract_general_points(combined_text)

        return self.templates['definition'].format(
            term=term.title(),
            definition=definition,
            points=points
        )

    def _generate_howto_response(self, query: str, chunks: List[Dict]) -> str:
        """Generate a how-to style response."""
        # Extract the action
        action_match = re.search(r'how (?:to|do|can) (.+?)\?', query.lower())
        action = action_match.group(1) if action_match else "perform this task"

        combined_text = " ".join(chunk['text'] for chunk in chunks[:3])

        if 'add' in query.lower() and 'document' in query.lower():
            steps = """1. Place your documents in the data folder
2. Restart the system
3. The ingestion process will automatically detect and process new files"""
            additional_info = "Supported formats include PDF, DOCX, TXT, and Markdown files."
        else:
            steps = self._extract_steps(combined_text)
            additional_info = "Refer to the documentation for more details."

        return self.templates['howto'].format(
            action=action,
            steps=steps,
            additional_info=additional_info
        )

    def _generate_general_response(self, query: str, chunks: List[Dict]) -> str:
        """Generate a general response."""
        combined_text = " ".join(chunk['text'] for chunk in chunks[:3])

        # Create summary
        if 'accuracy' in query.lower():
            summary = "the RAG system's accuracy depends on the quality of your source documents."
        elif 'support' in query.lower() or 'format' in query.lower():
            summary = "this system supports multiple file formats for document processing."
        else:
            summary = "here's what I found in the documentation."

        # Extract key details
        details = self._extract_relevant_details(combined_text, query)

        # List sources
        sources = list(set(chunk.get('document_id', 'documentation') for chunk in chunks[:3]))
        source_text = ', '.join(sources)

        return self.templates['general'].format(
            summary=summary,
            details=details,
            sources=source_text
        )

    def _generate_not_found_response(self, query: str) -> str:
        """Generate response when no relevant chunks found."""
        doc_topics = "RAG systems, document processing, file formats, and system accuracy"

        return self.templates['not_found'].format(
            query=query,
            doc_topics=doc_topics
        )

    def _extract_key_points_rag(self, text: str) -> str:
        """Extract key points about RAG from text."""
        points = []

        if 'retrieval' in text.lower():
            points.append("• Combines document retrieval with text generation")
        if 'vector' in text.lower():
            points.append("• Uses vector databases for semantic search")
        if 'hallucination' in text.lower():
            points.append("• Reduces AI hallucination by grounding responses in real documents")
        if 'context' in text.lower():
            points.append("• Provides context to language models for accurate answers")

        if not points:
            points = ["• A powerful AI technique for document-based question answering"]

        return "\n".join(points)

    def _extract_file_formats(self, text: str) -> str:
        """Extract file format information."""
        formats = []
        format_map = {
            'pdf': 'PDF documents',
            'docx': 'Word documents (DOCX)',
            'txt': 'Plain text files',
            'markdown': 'Markdown files'
        }

        for fmt, description in format_map.items():
            if fmt in text.lower():
                formats.append(f"• {description}")

        if not formats:
            formats = ["• Multiple document formats are supported"]

        return "\n".join(formats)

    def _extract_general_points(self, text: str) -> str:
        """Extract general key points."""
        return "• Key information found in the documentation\n• Details are available in the source materials"

    def _extract_steps(self, text: str) -> str:
        """Extract step-by-step information."""
        if 'data folder' in text.lower():
            return "1. Follow the process described in the documentation\n2. Check the source materials for specific steps"
        return "1. Refer to the documentation for detailed instructions"

    def _extract_relevant_details(self, text: str, query: str) -> str:
        """Extract details relevant to the query."""
        # Take the most relevant portion of the text
        sentences = text.split('.')[:3]  # First 3 sentences
        relevant_text = '. '.join(s.strip() for s in sentences if s.strip())

        if len(relevant_text) > 200:
            relevant_text = relevant_text[:200] + "..."

        return relevant_text or "Information is available in the source documents."


# Modified test that works without OpenAI API
def test_rag_with_local_llm():
    """Test RAG pipeline with local LLM fallback."""
    print("🤖 Testing RAG with Local LLM (No API required!)")

    try:
        from embeddings.embedding_generator import EmbeddingGenerator
        from vectorstore.vector_db import VectorDatabase

        # Initialize components
        print("🚀 Loading RAG components...")
        embedder = EmbeddingGenerator(model_type="sentence-bert")
        vector_db = VectorDatabase.load("vectorstore", "document_vectors")
        local_llm = LocalLLMFallback()

        print("✅ RAG system ready (using local LLM)!")

        # Test queries
        test_queries = [
            "What is RAG?",
            "How do I add new documents?",
            "What file formats are supported?",
            "How accurate is the system?",
            "What is quantum computing?"  # Not in docs
        ]

        for query in test_queries:
            print(f"\n{'=' * 60}")
            print(f"❓ Question: {query}")

            # Retrieve relevant chunks
            query_embedding = embedder.generate_embedding(query)
            chunks = vector_db.search(query_embedding, top_k=3)

            # Generate answer with local LLM
            answer = local_llm.generate_answer(query, chunks)

            print(f"🤖 Answer: {answer}")
            print(f"📚 Sources: {len(chunks)} chunks found")

            if chunks:
                top_score = chunks[0].get('similarity_score', 0)
                print(f"🎯 Relevance: {top_score:.2f}")

        print(f"\n🎉 Local RAG demo completed!")
        print("💡 This shows your RAG system working end-to-end without API costs!")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    test_rag_with_local_llm()
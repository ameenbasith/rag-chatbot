"""
Streamlit web interface for the RAG chatbot.
A beautiful, user-friendly interface for interacting with your RAG system.
"""

import streamlit as st
import os
import sys
import time
from pathlib import Path
from typing import List, Dict

# Add the parent directory to Python path so we can import our modules
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Import our RAG components
try:
    from embeddings.embedding_generator import EmbeddingGenerator
    from vectorstore.vector_db import VectorDatabase
    from test_local_rag import LocalLLMFallback
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.info("Make sure you're running from the project root directory!")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🤖 RAG Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

class StreamlitRAGSystem:
    """RAG system optimized for Streamlit interface."""

    def __init__(self):
        self.embedder = None
        self.vector_db = None
        self.local_llm = None
        self.is_initialized = False

    def initialize(self):
        """Initialize the RAG system with caching."""
        try:
            with st.spinner("🚀 Loading RAG system..."):
                # Load embedding model
                self.embedder = EmbeddingGenerator(model_type="sentence-bert")

                # Load vector database
                self.vector_db = VectorDatabase.load("vectorstore", "document_vectors")

                # Initialize local LLM
                self.local_llm = LocalLLMFallback()

                self.is_initialized = True
                st.success("✅ RAG system loaded successfully!")
                return True

        except Exception as e:
            st.error(f"❌ Failed to load RAG system: {str(e)}")
            st.info("💡 Make sure you've run the setup scripts first!")
            st.info("Run: python test_vector_upgrade.py")
            return False

    def query(self, question: str, top_k: int = 3) -> Dict:
        """Query the RAG system."""
        if not self.is_initialized:
            return {"error": "System not initialized"}

        try:
            start_time = time.time()

            # Generate query embedding
            query_embedding = self.embedder.generate_embedding(question)

            # Search vector database
            chunks = self.vector_db.search(query_embedding, top_k=top_k)

            # Generate answer
            answer = self.local_llm.generate_answer(question, chunks)

            processing_time = time.time() - start_time

            # Calculate confidence
            confidence = chunks[0].get('similarity_score', 0) if chunks else 0

            return {
                "answer": answer,
                "chunks": chunks,
                "confidence": confidence,
                "processing_time": processing_time,
                "success": True
            }

        except Exception as e:
            return {"error": str(e), "success": False}

def display_chat_interface():
    """Display the main chat interface."""
    st.title("🤖 RAG Document Assistant")
    st.markdown("Ask me anything about your documents!")

    # Initialize RAG system
    if st.session_state.rag_system is None:
        st.session_state.rag_system = StreamlitRAGSystem()

    if not st.session_state.rag_system.is_initialized:
        if not st.session_state.rag_system.initialize():
            st.stop()

    # Sample questions as buttons
    st.markdown("### 💡 Try these sample questions:")
    col1, col2, col3 = st.columns(3)

    sample_questions = [
        "What is RAG?",
        "How do I add new documents?",
        "What file formats are supported?",
        "How accurate is the system?",
        "What are the benefits?",
        "What is quantum computing?"
    ]

    for i, question in enumerate(sample_questions):
        col = [col1, col2, col3][i % 3]
        with col:
            if st.button(question, key=f"sample_{i}"):
                # Add the question to chat
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()

    st.markdown("---")

    # Chat interface
    chat_container = st.container()

    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message["role"] == "assistant" and "metadata" in message:
                    # Show metadata in an expander
                    with st.expander("📊 Response Details", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Confidence", f"{message['metadata']['confidence']:.1%}")
                        with col2:
                            st.metric("Sources", len(message['metadata']['chunks']))
                        with col3:
                            st.metric("Time", f"{message['metadata']['processing_time']:.2f}s")

                        if message['metadata']['chunks']:
                            st.subheader("📚 Sources Used:")
                            for i, chunk in enumerate(message['metadata']['chunks'][:3], 1):
                                with st.container():
                                    st.write(f"**{i}. {chunk.get('document_id', 'Unknown')}** (Relevance: {chunk.get('similarity_score', 0):.1%})")
                                    with st.expander(f"View text from {chunk.get('document_id', 'source')}"):
                                        st.write(chunk.get('text', ''))

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.rag_system.query(prompt)

            if response.get("success", False):
                st.markdown(response["answer"])

                # Add assistant message with metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "metadata": {
                        "confidence": response["confidence"],
                        "chunks": response["chunks"],
                        "processing_time": response["processing_time"]
                    }
                })
            else:
                error_msg = f"❌ Error: {response.get('error', 'Unknown error')}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

def display_sidebar():
    """Display the sidebar with system info and controls."""
    st.sidebar.title("🛠️ System Dashboard")

    # System status
    st.sidebar.subheader("📊 System Status")

    if st.session_state.rag_system and st.session_state.rag_system.is_initialized:
        st.sidebar.success("✅ RAG System Online")

        # Get system stats
        try:
            stats = st.session_state.rag_system.vector_db.get_stats()
            st.sidebar.metric("Documents Indexed", stats.get('total_chunks', 0))
            st.sidebar.metric("Vector Dimension", stats.get('embedding_dimension', 0))
            st.sidebar.metric("Index Type", stats.get('index_type', 'Unknown'))
        except:
            st.sidebar.info("Stats unavailable")
    else:
        st.sidebar.warning("⚠️ System Loading...")

    # Clear chat button
    st.sidebar.subheader("🗑️ Chat Controls")
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # System information
    st.sidebar.subheader("ℹ️ About This RAG System")
    st.sidebar.info("""
    🔍 **Semantic Search**: Uses Sentence-BERT embeddings to understand meaning
    
    🗄️ **Vector Database**: FAISS for lightning-fast similarity search
    
    🤖 **Local Generation**: Rule-based answer generation (no API costs!)
    
    📚 **Source Attribution**: Shows which documents were used for each answer
    
    **Built with**: Python, Streamlit, Sentence-Transformers, FAISS
    """)

    # Performance info
    st.sidebar.subheader("⚡ Performance")
    if st.session_state.rag_system and st.session_state.rag_system.is_initialized:
        st.sidebar.success("🚀 System ready for queries")
        st.sidebar.info("Typical response time: 0.1-2 seconds")

    # Project info
    st.sidebar.subheader("🎯 What This Demonstrates")
    st.sidebar.write("""
    This is a complete RAG (Retrieval-Augmented Generation) system that showcases:
    
    • Document ingestion and chunking
    • Semantic embeddings
    • Vector similarity search  
    • Context-aware answer generation
    • Source attribution
    • Web interface
    
    Perfect for internal knowledge bases, customer support, or document Q&A!
    """)

def main():
    """Main app function."""
    # Display sidebar
    display_sidebar()

    # Main content
    display_chat_interface()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
        "🤖 RAG Document Assistant | Built with Streamlit, Sentence-BERT & FAISS | "
        "This demonstrates a complete production-ready RAG system"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
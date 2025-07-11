"""
Embedding generation module for the RAG chatbot.
Converts text chunks into vector representations for semantic search.
"""

import os
import numpy as np
from typing import List, Dict, Optional
import pickle
from pathlib import Path

# Import embedding models
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class EmbeddingGenerator:
    """
    A class to generate embeddings from text using various models.
    Supports both local models (Sentence-BERT) and API-based models (OpenAI).
    """

    def __init__(self, model_type: str = "sentence-bert", model_name: Optional[str] = None):
        """
        Initialize the embedding generator.

        Args:
            model_type: Type of model to use ("sentence-bert" or "openai")
            model_name: Specific model name (optional)
        """
        self.model_type = model_type
        self.model = None
        self.embedding_dim = None

        if model_type == "sentence-bert":
            # Use a lighter model for development - faster and less memory
            self.model_name = model_name or "all-MiniLM-L6-v2"
            self._load_sentence_bert_model()
        elif model_type == "openai":
            self.model_name = model_name or "text-embedding-3-small"
            self._setup_openai()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _load_sentence_bert_model(self):
        """Load the Sentence-BERT model."""
        try:
            print(f"Loading Sentence-BERT model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"✓ Model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            raise Exception(f"Failed to load Sentence-BERT model: {str(e)}")

    def _setup_openai(self):
        """Setup OpenAI API client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        openai.api_key = api_key
        # OpenAI's text-embedding-3-small has 1536 dimensions
        self.embedding_dim = 1536 if "3-small" in self.model_name else 1536
        print(f"✓ OpenAI API configured. Model: {self.model_name}")

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Numpy array containing the embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Empty text provided for embedding")

        try:
            if self.model_type == "sentence-bert":
                embedding = self.model.encode(text, convert_to_numpy=True)
            elif self.model_type == "openai":
                response = openai.embeddings.create(
                    model=self.model_name,
                    input=text
                )
                embedding = np.array(response.data[0].embedding)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            return embedding

        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embeddings = []

        try:
            if self.model_type == "sentence-bert":
                # Sentence-BERT can handle batches efficiently
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    batch_embeddings = self.model.encode(
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=True
                    )
                    embeddings.extend(batch_embeddings)

            elif self.model_type == "openai":
                # OpenAI API processes batches differently
                for i, text in enumerate(texts):
                    if i % 10 == 0:  # Progress indicator
                        print(f"Processing embedding {i + 1}/{len(texts)}")

                    embedding = self.generate_embedding(text)
                    embeddings.append(embedding)

            return embeddings

        except Exception as e:
            raise Exception(f"Failed to generate batch embeddings: {str(e)}")

    def embed_document_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for document chunks and add them to the chunk data.

        Args:
            chunks: List of chunk dictionaries from document processor

        Returns:
            List of chunks with embeddings added
        """
        if not chunks:
            return []

        print(f"Generating embeddings for {len(chunks)} chunks...")

        # Extract texts from chunks
        texts = [chunk['text'] for chunk in chunks]

        # Generate embeddings
        embeddings = self.generate_embeddings_batch(texts)

        # Add embeddings to chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding['embedding'] = embedding
            chunk_with_embedding['embedding_model'] = f"{self.model_type}:{self.model_name}"
            embedded_chunks.append(chunk_with_embedding)

        print(f"✓ Generated {len(embeddings)} embeddings")
        return embedded_chunks

    def save_embeddings(self, embedded_chunks: List[Dict], filepath: str):
        """
        Save embedded chunks to disk.

        Args:
            embedded_chunks: Chunks with embeddings
            filepath: Path to save the embeddings
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'chunks': embedded_chunks,
                    'model_info': {
                        'type': self.model_type,
                        'name': self.model_name,
                        'dimension': self.embedding_dim
                    }
                }, f)

            print(f"✓ Embeddings saved to {filepath}")

        except Exception as e:
            raise Exception(f"Failed to save embeddings: {str(e)}")

    def load_embeddings(self, filepath: str) -> List[Dict]:
        """
        Load embedded chunks from disk.

        Args:
            filepath: Path to the saved embeddings

        Returns:
            List of chunks with embeddings
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Embeddings file not found: {filepath}")

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            print(f"✓ Loaded {len(data['chunks'])} embedded chunks")
            print(f"  Model: {data['model_info']['type']}:{data['model_info']['name']}")
            print(f"  Dimension: {data['model_info']['dimension']}")

            return data['chunks']

        except Exception as e:
            raise Exception(f"Failed to load embeddings: {str(e)}")

    def similarity_search(self, query: str, embedded_chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Find the most similar chunks to a query.

        Args:
            query: Search query
            embedded_chunks: Chunks with embeddings
            top_k: Number of top results to return

        Returns:
            List of most similar chunks with similarity scores
        """
        if not embedded_chunks:
            return []

        # Generate embedding for the query
        query_embedding = self.generate_embedding(query)

        # Calculate similarities
        similarities = []
        for chunk in embedded_chunks:
            # Cosine similarity
            similarity = np.dot(query_embedding, chunk['embedding']) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk['embedding'])
            )
            similarities.append((similarity, chunk))

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for similarity, chunk in similarities[:top_k]:
            result = chunk.copy()
            result['similarity_score'] = float(similarity)
            results.append(result)

        return results


# Example usage and testing
if __name__ == "__main__":
    # Test the embedding generator
    try:
        print("🧪 Testing Embedding Generator...")

        # Initialize with the lighter model for testing
        embedder = EmbeddingGenerator(model_type="sentence-bert")

        # Test single embedding
        test_text = "This is a test document about artificial intelligence and machine learning."
        embedding = embedder.generate_embedding(test_text)
        print(f"✓ Generated embedding with shape: {embedding.shape}")

        # Test batch embeddings
        test_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing helps computers understand human language.",
            "Vector databases store high-dimensional embeddings efficiently."
        ]

        embeddings = embedder.generate_embeddings_batch(test_texts)
        print(f"✓ Generated {len(embeddings)} batch embeddings")

        # Test similarity search
        query = "What is AI?"

        # Create mock chunks for testing
        mock_chunks = []
        for i, (text, embedding) in enumerate(zip(test_texts, embeddings)):
            mock_chunks.append({
                'id': f'test_chunk_{i}',
                'text': text,
                'embedding': embedding
            })

        results = embedder.similarity_search(query, mock_chunks, top_k=2)
        print(f"✓ Found {len(results)} similar chunks for query: '{query}'")

        for i, result in enumerate(results):
            print(f"  {i + 1}. Score: {result['similarity_score']:.3f}")
            print(f"     Text: {result['text'][:60]}...")

        print("\n✅ Embedding generator test completed!")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
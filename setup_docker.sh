#!/bin/bash

# Docker Setup Script for RAG Chatbot
echo "🐳 Setting up Docker for RAG Chatbot..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first:"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose:"
    echo "   https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker is installed"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data vectorstore embeddings

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t rag-chatbot .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"

    echo ""
    echo "🚀 To run your RAG chatbot:"
    echo "   docker run -p 8501:8501 rag-chatbot"
    echo ""
    echo "🐳 Or use Docker Compose:"
    echo "   docker-compose up"
    echo ""
    echo "🌐 Then open: http://localhost:8501"
    echo ""
    echo "💡 To run in background:"
    echo "   docker-compose up -d"
    echo ""
    echo "📊 To view logs:"
    echo "   docker-compose logs -f"
    echo ""
    echo "🛑 To stop:"
    echo "   docker-compose down"

else
    echo "❌ Docker build failed. Check the output above for errors."
    exit 1
fi
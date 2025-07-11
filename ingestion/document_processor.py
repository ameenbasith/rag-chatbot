"""
Document processing module for the RAG chatbot.
Handles loading, parsing, and chunking of various document formats.
"""

import os
import re
from typing import List, Dict
from pathlib import Path

# Document processing libraries
import PyPDF2
from docx import Document
import markdown


class DocumentProcessor:
    """
    A class to handle document loading, parsing, and chunking.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the document processor.

        Args:
            chunk_size: Maximum size of each text chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_formats = {'.pdf', '.txt', '.md', '.docx'}

    def load_document(self, file_path: str) -> Dict[str, str]:
        """
        Load a document and extract its text content.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary containing document metadata and content
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Extract text based on file type
        if file_path.suffix.lower() == '.pdf':
            text = self._extract_pdf_text(file_path)
        elif file_path.suffix.lower() == '.docx':
            text = self._extract_docx_text(file_path)
        elif file_path.suffix.lower() == '.md':
            text = self._extract_markdown_text(file_path)
        elif file_path.suffix.lower() == '.txt':
            text = self._extract_txt_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        return {
            'filename': file_path.name,
            'filepath': str(file_path),
            'content': text,
            'size': len(text),
            'format': file_path.suffix.lower()
        }

    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
        return text

    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")

    def _extract_markdown_text(self, file_path: Path) -> str:
        """Extract text from Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
                # Convert markdown to plain text (removing markdown syntax)
                html = markdown.markdown(md_content)
                # Simple HTML tag removal
                text = re.sub(r'<[^>]+>', '', html)
                return text
        except Exception as e:
            raise Exception(f"Error reading Markdown: {str(e)}")

    def _extract_txt_text(self, file_path: Path) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Error reading TXT: {str(e)}")

    def chunk_text(self, text: str, document_id: str = None) -> List[Dict[str, str]]:
        """
        Split text into overlapping chunks.

        Args:
            text: The text to chunk
            document_id: Optional identifier for the source document

        Returns:
            List of text chunks with metadata
        """
        if not text.strip():
            return []

        # Clean up the text
        text = self._clean_text(text)

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size

            # If we're not at the end of the text, try to break at a sentence
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(end - 100, start)
                sentence_endings = ['.', '!', '?', '\n\n']

                best_break = -1
                for ending in sentence_endings:
                    pos = text.rfind(ending, search_start, end)
                    if pos > best_break:
                        best_break = pos

                if best_break > start:
                    end = best_break + 1

            # Extract the chunk
            chunk_text = text[start:end].strip()

            if chunk_text:  # Only add non-empty chunks
                chunks.append({
                    'id': f"{document_id}_{chunk_id}" if document_id else f"chunk_{chunk_id}",
                    'text': chunk_text,
                    'start_pos': start,
                    'end_pos': end,
                    'document_id': document_id,
                    'chunk_index': chunk_id
                })
                chunk_id += 1

            # Move to next chunk with overlap
            start = end - self.chunk_overlap

            # Safety check to avoid infinite loop
            if start <= 0:
                break

        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\']+', '', text)
        return text.strip()

    def process_directory(self, directory_path: str) -> List[Dict]:
        """
        Process all supported documents in a directory.

        Args:
            directory_path: Path to directory containing documents

        Returns:
            List of processed documents with chunks
        """
        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        processed_docs = []

        for file_path in directory_path.iterdir():
            if file_path.suffix.lower() in self.supported_formats:
                try:
                    print(f"Processing: {file_path.name}")

                    # Load document
                    doc_data = self.load_document(file_path)

                    # Create chunks
                    chunks = self.chunk_text(
                        doc_data['content'],
                        document_id=doc_data['filename']
                    )

                    processed_docs.append({
                        'document': doc_data,
                        'chunks': chunks
                    })

                    print(f"✓ Processed {file_path.name}: {len(chunks)} chunks")

                except Exception as e:
                    print(f"✗ Error processing {file_path.name}: {str(e)}")
                    continue

        return processed_docs


# Example usage and testing
if __name__ == "__main__":
    # Create a test processor
    processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)

    # Test with a sample text
    sample_text = """
    This is a sample document for testing the RAG chatbot system.
    The document processor should be able to handle various types of content.

    It should split long documents into manageable chunks while preserving context.
    Each chunk should overlap slightly with the next to maintain continuity.

    This approach helps ensure that the retrieval system can find relevant information
    even when the answer spans multiple sections of a document.
    """

    chunks = processor.chunk_text(sample_text, "sample_doc")

    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"Text: {chunk['text'][:100]}...")
        print(f"Length: {len(chunk['text'])} characters")
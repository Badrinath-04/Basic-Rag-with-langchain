"""
Document loaders for various file formats in the RAG application.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path

# LangChain document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredWordDocumentLoader
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter


class MultiFormatDocumentLoader:
    """
    A unified document loader that handles multiple file formats.
    """
    
    def __init__(self):
        self.supported_formats = {
            '.pdf': self._load_pdf,
            '.docx': self._load_docx,
            '.txt': self._load_txt,
            '.csv': self._load_csv,
            '.json': self._load_json
        }
        
        # Text splitters for different document types
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.csv_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            separator="\n"
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document based on its file extension.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        return self.supported_formats[file_extension](str(file_path))
    
    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory containing documents
            
        Returns:
            List of Document objects from all supported files
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        all_documents = []
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    documents = self.load_document(str(file_path))
                    all_documents.extend(documents)
                    print(f"Loaded {len(documents)} chunks from {file_path.name}")
                except Exception as e:
                    print(f"Error loading {file_path.name}: {str(e)}")
        
        return all_documents
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF documents."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def _load_docx(self, file_path: str) -> List[Document]:
        """Load DOCX documents."""
        loader = UnstructuredWordDocumentLoader(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def _load_txt(self, file_path: str) -> List[Document]:
        """Load TXT documents."""
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def _load_csv(self, file_path: str) -> List[Document]:
        """Load CSV documents."""
        loader = CSVLoader(file_path, encoding='utf-8')
        documents = loader.load()
        return self.csv_splitter.split_documents(documents)
    
    def _load_json(self, file_path: str) -> List[Document]:
        """Load JSON documents."""
        loader = JSONLoader(
            file_path=file_path,
            jq_schema='.[]',
            text_content=False
        )
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def get_document_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata about a document without loading its content.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing document metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        metadata = {
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'file_extension': file_path.suffix.lower(),
            'file_path': str(file_path),
            'is_supported': file_path.suffix.lower() in self.supported_formats
        }
        
        return metadata


def create_sample_documents():
    """
    Create sample documents for testing the RAG application.
    """
    # Create sample directory
    sample_dir = Path("sample_documents")
    sample_dir.mkdir(exist_ok=True)
    
    # Sample TXT document
    with open(sample_dir / "sample.txt", "w", encoding="utf-8") as f:
        f.write("""
        Artificial Intelligence and Machine Learning
        
        Artificial Intelligence (AI) is a broad field of computer science focused on creating 
        intelligent machines that can perform tasks that typically require human intelligence. 
        Machine Learning (ML) is a subset of AI that enables computers to learn and improve 
        from experience without being explicitly programmed.
        
        Key concepts in machine learning include:
        - Supervised learning: Learning with labeled training data
        - Unsupervised learning: Finding patterns in data without labels
        - Reinforcement learning: Learning through interaction with an environment
        
        Deep learning, a subset of machine learning, uses neural networks with multiple layers 
        to model and understand complex patterns in data.
        """)
    
    # Sample JSON document
    with open(sample_dir / "sample.json", "w", encoding="utf-8") as f:
        json.dump([
            {
                "title": "Python Programming",
                "content": "Python is a high-level programming language known for its simplicity and readability.",
                "topics": ["programming", "python", "software development"]
            },
            {
                "title": "Data Science",
                "content": "Data science combines statistics, programming, and domain expertise to extract insights from data.",
                "topics": ["data science", "statistics", "analytics"]
            },
            {
                "title": "Web Development",
                "content": "Web development involves creating websites and web applications using various technologies.",
                "topics": ["web development", "html", "css", "javascript"]
            }
        ], f, indent=2)
    
    # Sample CSV document
    with open(sample_dir / "sample.csv", "w", encoding="utf-8") as f:
        f.write("""Name,Age,Occupation,Skills
John Doe,30,Software Engineer,"Python, JavaScript, React"
Jane Smith,28,Data Scientist,"Python, R, Machine Learning"
Bob Johnson,35,DevOps Engineer,"Docker, Kubernetes, AWS"
Alice Brown,32,Product Manager,"Agile, Scrum, User Research"
""")
    
    print(f"Sample documents created in {sample_dir}")
    return str(sample_dir)


if __name__ == "__main__":
    # Test the document loader
    loader = MultiFormatDocumentLoader()
    
    # Create sample documents
    sample_dir = create_sample_documents()
    
    # Load all documents
    documents = loader.load_documents_from_directory(sample_dir)
    
    print(f"\nLoaded {len(documents)} document chunks:")
    for i, doc in enumerate(documents[:3]):  # Show first 3 chunks
        print(f"\nChunk {i+1}:")
        print(f"Content: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")
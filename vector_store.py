"""
Vector store implementation using ChromaDB for the RAG application.
"""

import os
import chromadb
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid

from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings


class RAGVectorStore:
    """
    A vector store implementation using ChromaDB for document storage and retrieval.
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: str = "openai",
        collection_name: str = "rag_documents"
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the ChromaDB database
            embedding_model: Type of embedding model to use ('openai' or 'huggingface')
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = self._initialize_embeddings()
        
        # Initialize ChromaDB
        self.vector_store = self._initialize_vector_store()
    
    def _initialize_embeddings(self) -> Embeddings:
        """Initialize the embedding model."""
        if self.embedding_model == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI embeddings")
            return OpenAIEmbeddings()
        
        elif self.embedding_model == "huggingface":
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
        
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")
    
    def _initialize_vector_store(self) -> Chroma:
        """Initialize the ChromaDB vector store."""
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs that were added
        """
        if not documents:
            return []
        
        # Add documents to the vector store
        document_ids = self.vector_store.add_documents(documents)
        
        # Persist the changes
        self.vector_store.persist()
        
        print(f"Added {len(documents)} documents to vector store")
        return document_ids
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """
        Add text strings to the vector store.
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of document IDs that were added
        """
        if not texts:
            return []
        
        # Generate document IDs if not provided
        if metadatas is None:
            metadatas = [{"source": f"text_{i}"} for i in range(len(texts))]
        
        # Add texts to the vector store
        document_ids = self.vector_store.add_texts(texts, metadatas)
        
        # Persist the changes
        self.vector_store.persist()
        
        print(f"Added {len(texts)} text chunks to vector store")
        return document_ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Query string
            k: Number of similar documents to return
            filter: Optional metadata filter
            
        Returns:
            List of similar Document objects
        """
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[tuple]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Query string
            k: Number of similar documents to return
            filter: Optional metadata filter
            
        Returns:
            List of tuples containing (Document, score)
        """
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
    
    def get_retriever(self, search_type: str = "similarity", **kwargs):
        """
        Get a retriever for the vector store.
        
        Args:
            search_type: Type of search ('similarity', 'mmr', etc.)
            **kwargs: Additional arguments for the retriever
            
        Returns:
            VectorStoreRetriever object
        """
        return self.vector_store.as_retriever(
            search_type=search_type,
            **kwargs
        )
    
    def delete_collection(self):
        """Delete the entire collection."""
        # Get the ChromaDB client
        client = chromadb.PersistentClient(path=self.persist_directory)
        
        try:
            client.delete_collection(self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {str(e)}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dictionary containing collection information
        """
        try:
            # Get the ChromaDB client
            client = chromadb.PersistentClient(path=self.persist_directory)
            collection = client.get_collection(self.collection_name)
            
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_model
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_model,
                "error": str(e)
            }
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        self.delete_collection()
        self.vector_store = self._initialize_vector_store()
        print("Collection cleared")


class VectorStoreManager:
    """
    Manager class for handling multiple vector stores and collections.
    """
    
    def __init__(self, base_persist_directory: str = "./chroma_db"):
        self.base_persist_directory = base_persist_directory
        self.vector_stores: Dict[str, RAGVectorStore] = {}
    
    def create_vector_store(
        self,
        name: str,
        embedding_model: str = "openai",
        collection_name: Optional[str] = None
    ) -> RAGVectorStore:
        """
        Create a new vector store.
        
        Args:
            name: Name for the vector store
            embedding_model: Type of embedding model to use
            collection_name: Optional custom collection name
            
        Returns:
            RAGVectorStore instance
        """
        if collection_name is None:
            collection_name = f"{name}_documents"
        
        persist_directory = os.path.join(self.base_persist_directory, name)
        
        vector_store = RAGVectorStore(
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            collection_name=collection_name
        )
        
        self.vector_stores[name] = vector_store
        return vector_store
    
    def get_vector_store(self, name: str) -> Optional[RAGVectorStore]:
        """Get an existing vector store by name."""
        return self.vector_stores.get(name)
    
    def list_vector_stores(self) -> List[str]:
        """List all available vector stores."""
        return list(self.vector_stores.keys())
    
    def delete_vector_store(self, name: str):
        """Delete a vector store."""
        if name in self.vector_stores:
            self.vector_stores[name].delete_collection()
            del self.vector_stores[name]
            print(f"Deleted vector store: {name}")


if __name__ == "__main__":
    # Test the vector store
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Create a test vector store
    vector_store = RAGVectorStore(
        persist_directory="./test_chroma_db",
        embedding_model="huggingface"  # Use HuggingFace for testing
    )
    
    # Test documents
    test_documents = [
        Document(
            page_content="This is a test document about artificial intelligence.",
            metadata={"source": "test1.txt", "topic": "AI"}
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "test2.txt", "topic": "ML"}
        ),
        Document(
            page_content="Deep learning uses neural networks for complex pattern recognition.",
            metadata={"source": "test3.txt", "topic": "Deep Learning"}
        )
    ]
    
    # Add documents
    vector_store.add_documents(test_documents)
    
    # Test similarity search
    results = vector_store.similarity_search("What is machine learning?", k=2)
    
    print("Similarity search results:")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
    
    # Get collection info
    info = vector_store.get_collection_info()
    print(f"\nCollection info: {info}")
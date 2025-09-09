"""
RAG (Retrieval-Augmented Generation) pipeline implementation using LangChain.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever

from document_loaders import MultiFormatDocumentLoader
from vector_store import RAGVectorStore, VectorStoreManager


class RAGPipeline:
    """
    A complete RAG pipeline that handles document ingestion, vector storage, and question answering.
    """
    
    def __init__(
        self,
        vector_store: RAGVectorStore,
        llm_model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: Vector store instance for document storage and retrieval
            llm_model: OpenAI model to use for generation
            temperature: Temperature for LLM generation
            max_tokens: Maximum tokens for LLM response
        """
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize document loader
        self.document_loader = MultiFormatDocumentLoader()
        
        # Initialize retrieval QA chain
        self.qa_chain = self._initialize_qa_chain()
        
        # Initialize conversational chain
        self.conversational_chain = self._initialize_conversational_chain()
    
    def _initialize_llm(self):
        """Initialize the language model."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        return ChatOpenAI(
            model_name=self.llm_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def _initialize_qa_chain(self):
        """Initialize the question-answering chain."""
        retriever = self.vector_store.get_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {question}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def _initialize_conversational_chain(self):
        """Initialize the conversational retrieval chain."""
        retriever = self.vector_store.get_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True
        )
    
    def ingest_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Ingest documents into the vector store.
        
        Args:
            file_paths: List of file paths to ingest
            
        Returns:
            Dictionary containing ingestion results
        """
        results = {
            "total_documents": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "errors": [],
            "document_ids": []
        }
        
        all_documents = []
        
        for file_path in file_paths:
            try:
                documents = self.document_loader.load_document(file_path)
                all_documents.extend(documents)
                results["successful_documents"] += 1
                print(f"Successfully loaded {len(documents)} chunks from {file_path}")
            except Exception as e:
                error_msg = f"Error loading {file_path}: {str(e)}"
                results["errors"].append(error_msg)
                results["failed_documents"] += 1
                print(error_msg)
        
        results["total_documents"] = len(all_documents)
        
        if all_documents:
            # Add documents to vector store
            document_ids = self.vector_store.add_documents(all_documents)
            results["document_ids"] = document_ids
        
        return results
    
    def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Ingest all supported documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            Dictionary containing ingestion results
        """
        try:
            documents = self.document_loader.load_documents_from_directory(directory_path)
            
            if documents:
                document_ids = self.vector_store.add_documents(documents)
                return {
                    "total_documents": len(documents),
                    "successful_documents": len(documents),
                    "failed_documents": 0,
                    "errors": [],
                    "document_ids": document_ids
                }
            else:
                return {
                    "total_documents": 0,
                    "successful_documents": 0,
                    "failed_documents": 0,
                    "errors": ["No supported documents found in directory"],
                    "document_ids": []
                }
        except Exception as e:
            return {
                "total_documents": 0,
                "successful_documents": 0,
                "failed_documents": 0,
                "errors": [f"Error processing directory: {str(e)}"],
                "document_ids": []
            }
    
    def ask_question(self, question: str, use_conversation: bool = False) -> Dict[str, Any]:
        """
        Ask a question and get an answer from the RAG system.
        
        Args:
            question: The question to ask
            use_conversation: Whether to use conversational memory
            
        Returns:
            Dictionary containing the answer and metadata
        """
        try:
            if use_conversation:
                result = self.conversational_chain({"question": question})
                return {
                    "question": question,
                    "answer": result["answer"],
                    "source_documents": result.get("source_documents", []),
                    "chat_history": result.get("chat_history", []),
                    "success": True
                }
            else:
                result = self.qa_chain({"query": question})
                return {
                    "question": question,
                    "answer": result["result"],
                    "source_documents": result.get("source_documents", []),
                    "success": True
                }
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "source_documents": [],
                "success": False,
                "error": str(e)
            }
    
    def search_documents(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for relevant documents without generating an answer.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of relevant documents
        """
        return self.vector_store.similarity_search(query, k=k)
    
    def search_documents_with_scores(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        Search for relevant documents with relevance scores.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of tuples containing (Document, score)
        """
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the current pipeline state.
        
        Returns:
            Dictionary containing pipeline information
        """
        vector_store_info = self.vector_store.get_collection_info()
        
        return {
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "vector_store": vector_store_info,
            "supported_formats": self.document_loader.supported_formats.keys()
        }
    
    def clear_memory(self):
        """Clear the conversational memory."""
        if hasattr(self.conversational_chain, 'memory'):
            self.conversational_chain.memory.clear()
        print("Conversational memory cleared")


class RAGApplication:
    """
    Main application class that manages multiple RAG pipelines.
    """
    
    def __init__(self, base_persist_directory: str = "./chroma_db"):
        self.base_persist_directory = base_persist_directory
        self.vector_store_manager = VectorStoreManager(base_persist_directory)
        self.pipelines: Dict[str, RAGPipeline] = {}
    
    def create_pipeline(
        self,
        name: str,
        embedding_model: str = "openai",
        llm_model: str = "gpt-3.5-turbo",
        **kwargs
    ) -> RAGPipeline:
        """
        Create a new RAG pipeline.
        
        Args:
            name: Name for the pipeline
            embedding_model: Embedding model to use
            llm_model: LLM model to use
            **kwargs: Additional arguments for RAGPipeline
            
        Returns:
            RAGPipeline instance
        """
        vector_store = self.vector_store_manager.create_vector_store(
            name=name,
            embedding_model=embedding_model
        )
        
        pipeline = RAGPipeline(
            vector_store=vector_store,
            llm_model=llm_model,
            **kwargs
        )
        
        self.pipelines[name] = pipeline
        return pipeline
    
    def get_pipeline(self, name: str) -> Optional[RAGPipeline]:
        """Get an existing pipeline by name."""
        return self.pipelines.get(name)
    
    def list_pipelines(self) -> List[str]:
        """List all available pipelines."""
        return list(self.pipelines.keys())
    
    def delete_pipeline(self, name: str):
        """Delete a pipeline."""
        if name in self.pipelines:
            self.vector_store_manager.delete_vector_store(name)
            del self.pipelines[name]
            print(f"Deleted pipeline: {name}")


if __name__ == "__main__":
    # Test the RAG pipeline
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Create RAG application
    app = RAGApplication()
    
    # Create a pipeline
    pipeline = app.create_pipeline(
        name="test_pipeline",
        embedding_model="huggingface",  # Use HuggingFace for testing
        llm_model="gpt-3.5-turbo"
    )
    
    # Create sample documents
    from document_loaders import create_sample_documents
    sample_dir = create_sample_documents()
    
    # Ingest documents
    print("Ingesting documents...")
    results = pipeline.ingest_directory(sample_dir)
    print(f"Ingestion results: {results}")
    
    # Ask questions
    questions = [
        "What is artificial intelligence?",
        "What are the key concepts in machine learning?",
        "What programming languages are mentioned?",
        "Tell me about data science"
    ]
    
    print("\nAsking questions...")
    for question in questions:
        answer = pipeline.ask_question(question)
        print(f"\nQ: {question}")
        print(f"A: {answer['answer']}")
        if answer.get('source_documents'):
            print(f"Sources: {len(answer['source_documents'])} documents")
    
    # Get pipeline info
    info = pipeline.get_pipeline_info()
    print(f"\nPipeline info: {info}")
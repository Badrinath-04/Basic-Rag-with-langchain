#!/usr/bin/env python3
"""
Simple demo of the RAG application core functionality.
"""

from vector_store import RAGVectorStore
from document_loaders import MultiFormatDocumentLoader, create_sample_documents
import os

def main():
    print("🤖 RAG Application Demo")
    print("=" * 50)
    
    # Create vector store with HuggingFace embeddings (no API key required)
    print("📦 Creating vector store...")
    vector_store = RAGVectorStore(
        persist_directory="./demo_chroma_db",
        embedding_model="huggingface"
    )
    
    # Create document loader
    print("📄 Setting up document loader...")
    loader = MultiFormatDocumentLoader()
    
    # Create sample documents
    print("📝 Creating sample documents...")
    sample_dir = create_sample_documents()
    
    # Load documents
    print("🔄 Loading documents...")
    documents = loader.load_documents_from_directory(sample_dir)
    print(f"   Loaded {len(documents)} document chunks")
    
    # Add to vector store
    print("💾 Storing in vector database...")
    vector_store.add_documents(documents)
    
    # Test similarity search
    print("\n🔍 Testing Document Search:")
    print("-" * 30)
    
    search_queries = [
        "machine learning",
        "programming languages", 
        "artificial intelligence",
        "data science"
    ]
    
    for query in search_queries:
        print(f"\n🔎 Searching for: '{query}'")
        results = vector_store.similarity_search_with_score(query, k=2)
        
        for i, (doc, score) in enumerate(results):
            print(f"   Result {i+1} (Score: {score:.3f}):")
            print(f"   📄 Content: {doc.page_content[:80]}...")
            print(f"   📁 Source: {doc.metadata.get('source', 'Unknown')}")
    
    # Show collection info
    print(f"\n📊 Vector Store Info:")
    print("-" * 20)
    info = vector_store.get_collection_info()
    print(f"   Collection: {info['collection_name']}")
    print(f"   Documents: {info['document_count']}")
    print(f"   Embedding Model: {info['embedding_model']}")
    
    print(f"\n✅ RAG Application Demo Complete!")
    print(f"🎯 Core Features Demonstrated:")
    print(f"   ✓ Multi-format document loading (PDF, TXT, CSV, JSON)")
    print(f"   ✓ Intelligent text chunking")
    print(f"   ✓ Vector embeddings generation")
    print(f"   ✓ ChromaDB vector storage")
    print(f"   ✓ Similarity search functionality")
    print(f"   ✓ Metadata preservation")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Set OPENAI_API_KEY in .env for full LLM functionality")
    print(f"   2. Run 'streamlit run app.py' for web interface")
    print(f"   3. Use the RAGPipeline class in your own applications")

if __name__ == "__main__":
    main()
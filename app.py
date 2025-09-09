"""
Streamlit web interface for the RAG application.
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import json

from dotenv import load_dotenv
from rag_pipeline import RAGApplication, RAGPipeline
from document_loaders import MultiFormatDocumentLoader

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Application",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .info-box {
        background-color: #e7f3ff;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_app' not in st.session_state:
    st.session_state.rag_app = RAGApplication()
if 'current_pipeline' not in st.session_state:
    st.session_state.current_pipeline = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_pipeline(pipeline_name: str, embedding_model: str, llm_model: str) -> RAGPipeline:
    """Initialize a new RAG pipeline."""
    try:
        pipeline = st.session_state.rag_app.create_pipeline(
            name=pipeline_name,
            embedding_model=embedding_model,
            llm_model=llm_model
        )
        return pipeline
    except Exception as e:
        st.error(f"Error initializing pipeline: {str(e)}")
        return None

def display_chat_message(role: str, content: str, sources: List[Dict] = None):
    """Display a chat message with proper formatting."""
    if role == "user":
        st.markdown(f"**You:** {content}")
    else:
        st.markdown(f"**Assistant:** {content}")
        
        if sources and len(sources) > 0:
            with st.expander("📚 Sources"):
                for i, source in enumerate(sources):
                    st.markdown(f"**Source {i+1}:**")
                    st.markdown(f"Content: {source.page_content[:200]}...")
                    if source.metadata:
                        st.markdown(f"Metadata: {source.metadata}")

def main():
    # Main header
    st.markdown('<h1 class="main-header">🤖 RAG Application with LangChain</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Pipeline selection
        st.markdown("### Pipeline Management")
        pipeline_name = st.text_input("Pipeline Name", value="default_pipeline")
        
        # Model selection
        embedding_model = st.selectbox(
            "Embedding Model",
            ["openai", "huggingface"],
            help="OpenAI requires API key, HuggingFace works offline"
        )
        
        llm_model = st.selectbox(
            "LLM Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            help="OpenAI model for text generation"
        )
        
        # Initialize pipeline button
        if st.button("🚀 Initialize Pipeline", type="primary"):
            if not os.getenv("OPENAI_API_KEY") and embedding_model == "openai":
                st.error("Please set OPENAI_API_KEY environment variable for OpenAI embeddings")
            else:
                pipeline = initialize_pipeline(pipeline_name, embedding_model, llm_model)
                if pipeline:
                    st.session_state.current_pipeline = pipeline
                    st.success(f"Pipeline '{pipeline_name}' initialized successfully!")
        
        # Current pipeline info
        if st.session_state.current_pipeline:
            st.markdown("### Current Pipeline")
            info = st.session_state.current_pipeline.get_pipeline_info()
            st.json(info)
            
            # Clear pipeline button
            if st.button("🗑️ Clear Pipeline"):
                st.session_state.current_pipeline = None
                st.session_state.chat_history = []
                st.success("Pipeline cleared!")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📄 Document Upload", "💬 Chat Interface", "🔍 Document Search"])
    
    with tab1:
        st.markdown('<h2 class="section-header">Document Upload & Processing</h2>', unsafe_allow_html=True)
        
        if not st.session_state.current_pipeline:
            st.warning("Please initialize a pipeline first using the sidebar.")
        else:
            # File upload
            uploaded_files = st.file_uploader(
                "Upload Documents",
                type=['pdf', 'docx', 'txt', 'csv', 'json'],
                accept_multiple_files=True,
                help="Supported formats: PDF, DOCX, TXT, CSV, JSON"
            )
            
            if uploaded_files:
                st.session_state.uploaded_files = uploaded_files
                
                # Display uploaded files
                st.markdown("### Uploaded Files")
                for file in uploaded_files:
                    st.write(f"📄 {file.name} ({file.size} bytes)")
                
                # Process files button
                if st.button("🔄 Process Documents", type="primary"):
                    with st.spinner("Processing documents..."):
                        # Save uploaded files temporarily
                        temp_dir = tempfile.mkdtemp()
                        file_paths = []
                        
                        for file in uploaded_files:
                            temp_path = os.path.join(temp_dir, file.name)
                            with open(temp_path, "wb") as f:
                                f.write(file.getbuffer())
                            file_paths.append(temp_path)
                        
                        # Ingest documents
                        results = st.session_state.current_pipeline.ingest_documents(file_paths)
                        
                        # Display results
                        if results["successful_documents"] > 0:
                            st.success(f"✅ Successfully processed {results['successful_documents']} documents!")
                            st.markdown(f"**Total chunks:** {results['total_documents']}")
                        else:
                            st.error("❌ No documents were processed successfully.")
                        
                        if results["errors"]:
                            st.error("Errors encountered:")
                            for error in results["errors"]:
                                st.write(f"- {error}")
                        
                        # Clean up temp files
                        import shutil
                        shutil.rmtree(temp_dir)
            
            # Directory upload option
            st.markdown("### Or Upload from Directory")
            directory_path = st.text_input("Directory Path", placeholder="/path/to/documents")
            
            if st.button("📁 Process Directory") and directory_path:
                if os.path.exists(directory_path):
                    with st.spinner("Processing directory..."):
                        results = st.session_state.current_pipeline.ingest_directory(directory_path)
                        
                        if results["successful_documents"] > 0:
                            st.success(f"✅ Successfully processed {results['successful_documents']} documents from directory!")
                        else:
                            st.error("❌ No documents were processed from directory.")
                else:
                    st.error("Directory path does not exist.")
    
    with tab2:
        st.markdown('<h2 class="section-header">Chat Interface</h2>', unsafe_allow_html=True)
        
        if not st.session_state.current_pipeline:
            st.warning("Please initialize a pipeline and upload documents first.")
        else:
            # Chat input
            user_question = st.text_input(
                "Ask a question about your documents:",
                placeholder="What is the main topic of the documents?",
                key="chat_input"
            )
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                use_conversation = st.checkbox("Use Conversation Memory", value=True)
            
            with col2:
                if st.button("💬 Ask Question", type="primary"):
                    if user_question:
                        with st.spinner("Thinking..."):
                            response = st.session_state.current_pipeline.ask_question(
                                user_question, 
                                use_conversation=use_conversation
                            )
                            
                            if response["success"]:
                                # Add to chat history
                                st.session_state.chat_history.append({
                                    "role": "user",
                                    "content": user_question
                                })
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": response["answer"],
                                    "sources": response.get("source_documents", [])
                                })
                            else:
                                st.error(f"Error: {response.get('error', 'Unknown error')}")
            
            with col3:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.chat_history = []
                    if hasattr(st.session_state.current_pipeline, 'clear_memory'):
                        st.session_state.current_pipeline.clear_memory()
                    st.success("Chat history cleared!")
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("### Chat History")
                for message in st.session_state.chat_history:
                    display_chat_message(
                        message["role"],
                        message["content"],
                        message.get("sources")
                    )
                    st.markdown("---")
    
    with tab3:
        st.markdown('<h2 class="section-header">Document Search</h2>', unsafe_allow_html=True)
        
        if not st.session_state.current_pipeline:
            st.warning("Please initialize a pipeline and upload documents first.")
        else:
            # Search interface
            search_query = st.text_input(
                "Search for relevant documents:",
                placeholder="Enter search terms...",
                key="search_input"
            )
            
            k_results = st.slider("Number of results", min_value=1, max_value=10, value=4)
            
            if st.button("🔍 Search Documents", type="primary") and search_query:
                with st.spinner("Searching..."):
                    # Search with scores
                    results = st.session_state.current_pipeline.search_documents_with_scores(
                        search_query, k=k_results
                    )
                    
                    if results:
                        st.success(f"Found {len(results)} relevant documents:")
                        
                        for i, (doc, score) in enumerate(results):
                            with st.expander(f"Document {i+1} (Score: {score:.3f})"):
                                st.markdown(f"**Content:** {doc.page_content}")
                                if doc.metadata:
                                    st.markdown(f"**Metadata:** {doc.metadata}")
                    else:
                        st.info("No relevant documents found.")
            
            # Show all documents option
            if st.button("📋 Show All Documents"):
                with st.spinner("Loading documents..."):
                    # Get collection info
                    info = st.session_state.current_pipeline.get_pipeline_info()
                    st.info(f"Total documents in collection: {info['vector_store']['document_count']}")
                    
                    # Search for all documents (using a broad query)
                    all_docs = st.session_state.current_pipeline.search_documents("", k=50)
                    
                    if all_docs:
                        st.markdown("### All Documents")
                        for i, doc in enumerate(all_docs):
                            with st.expander(f"Document {i+1}"):
                                st.markdown(f"**Content:** {doc.page_content[:500]}...")
                                if doc.metadata:
                                    st.markdown(f"**Metadata:** {doc.metadata}")

if __name__ == "__main__":
    main()
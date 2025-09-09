# RAG Application with LangChain

A comprehensive Retrieval-Augmented Generation (RAG) application that supports multiple file formats including PDF, DOCX, CSV, TXT, and JSON.

## Features

- **Multi-format Document Support**: PDF, DOCX, CSV, TXT, JSON
- **Intelligent Text Chunking**: Optimized chunking strategies for different document types
- **Vector Storage**: ChromaDB for efficient similarity search
- **Multiple Embedding Models**: OpenAI and Hugging Face support
- **Web Interface**: Streamlit-based UI for easy interaction
- **LangChain Integration**: Full pipeline orchestration

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Upload documents through the web interface
2. Process documents to create embeddings
3. Ask questions about your documents
4. Get contextual answers based on retrieved information

## Architecture

- **Document Loaders**: Handle various file formats
- **Text Splitters**: Chunk documents for optimal retrieval
- **Embeddings**: Convert text to vector representations
- **Vector Store**: ChromaDB for similarity search
- **LLM**: OpenAI GPT for answer generation
- **LangChain**: Orchestrates the entire pipeline
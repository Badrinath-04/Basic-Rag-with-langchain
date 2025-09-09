"""
Configuration settings for the RAG application.
"""

import os
from typing import Dict, Any, List
from pathlib import Path

# Base configuration
BASE_DIR = Path(__file__).parent
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
UPLOAD_DIR = BASE_DIR / "uploads"
SAMPLE_DIR = BASE_DIR / "sample_documents"

# Supported file formats
SUPPORTED_FORMATS = {
    '.pdf': 'PDF documents',
    '.docx': 'Microsoft Word documents',
    '.txt': 'Plain text files',
    '.csv': 'Comma-separated values',
    '.json': 'JSON data files'
}

# Text splitting configuration
TEXT_SPLITTER_CONFIG = {
    'chunk_size': 1000,
    'chunk_overlap': 200,
    'length_function': len,
    'separators': ["\n\n", "\n", " ", ""]
}

# CSV-specific splitting
CSV_SPLITTER_CONFIG = {
    'chunk_size': 1000,
    'chunk_overlap': 0,
    'separator': "\n"
}

# Embedding models configuration
EMBEDDING_MODELS = {
    'openai': {
        'class': 'OpenAIEmbeddings',
        'requires_api_key': True,
        'description': 'OpenAI text-embedding-ada-002 model'
    },
    'huggingface': {
        'class': 'HuggingFaceEmbeddings',
        'requires_api_key': False,
        'description': 'HuggingFace sentence-transformers model',
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2'
    }
}

# LLM models configuration
LLM_MODELS = {
    'gpt-3.5-turbo': {
        'max_tokens': 4000,
        'temperature': 0.7,
        'description': 'Fast and cost-effective model'
    },
    'gpt-4': {
        'max_tokens': 8000,
        'temperature': 0.7,
        'description': 'Most capable model with better reasoning'
    },
    'gpt-4-turbo-preview': {
        'max_tokens': 128000,
        'temperature': 0.7,
        'description': 'Latest GPT-4 model with larger context'
    }
}

# Vector store configuration
VECTOR_STORE_CONFIG = {
    'default_collection_name': 'rag_documents',
    'similarity_search_k': 4,
    'mmr_diversity_bias': 0.1
}

# Streamlit UI configuration
UI_CONFIG = {
    'page_title': 'RAG Application',
    'page_icon': '🤖',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Default prompts
DEFAULT_PROMPTS = {
    'qa_template': """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:""",
    
    'conversational_template': """You are a helpful AI assistant that answers questions based on the provided context. 
Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know.

Context:
{context}

Question: {question}

Answer:"""
}

# Error messages
ERROR_MESSAGES = {
    'missing_api_key': 'API key is required for this operation',
    'file_not_found': 'File not found',
    'unsupported_format': 'Unsupported file format',
    'empty_documents': 'No documents found to process',
    'vector_store_error': 'Error with vector store operation',
    'llm_error': 'Error with language model',
    'embedding_error': 'Error generating embeddings'
}

# Success messages
SUCCESS_MESSAGES = {
    'documents_processed': 'Documents processed successfully',
    'pipeline_initialized': 'Pipeline initialized successfully',
    'documents_uploaded': 'Documents uploaded successfully',
    'memory_cleared': 'Memory cleared successfully'
}

def get_config() -> Dict[str, Any]:
    """Get the complete configuration dictionary."""
    return {
        'base_dir': str(BASE_DIR),
        'chroma_db_dir': str(CHROMA_DB_DIR),
        'upload_dir': str(UPLOAD_DIR),
        'sample_dir': str(SAMPLE_DIR),
        'supported_formats': SUPPORTED_FORMATS,
        'text_splitter': TEXT_SPLITTER_CONFIG,
        'csv_splitter': CSV_SPLITTER_CONFIG,
        'embedding_models': EMBEDDING_MODELS,
        'llm_models': LLM_MODELS,
        'vector_store': VECTOR_STORE_CONFIG,
        'ui': UI_CONFIG,
        'prompts': DEFAULT_PROMPTS,
        'errors': ERROR_MESSAGES,
        'success': SUCCESS_MESSAGES
    }

def validate_config() -> List[str]:
    """Validate the configuration and return any issues."""
    issues = []
    
    # Check if required directories exist
    for directory in [CHROMA_DB_DIR, UPLOAD_DIR, SAMPLE_DIR]:
        if not directory.exists():
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create directory {directory}: {e}")
    
    # Check if .env file exists
    env_file = BASE_DIR / ".env"
    if not env_file.exists():
        issues.append(".env file not found. Please copy .env.example to .env")
    
    return issues

def get_embedding_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific embedding model."""
    if model_name not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown embedding model: {model_name}")
    
    return EMBEDDING_MODELS[model_name]

def get_llm_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific LLM model."""
    if model_name not in LLM_MODELS:
        raise ValueError(f"Unknown LLM model: {model_name}")
    
    return LLM_MODELS[model_name]

def is_api_key_required(embedding_model: str) -> bool:
    """Check if an API key is required for the embedding model."""
    config = get_embedding_model_config(embedding_model)
    return config.get('requires_api_key', False)

if __name__ == "__main__":
    # Print configuration
    config = get_config()
    print("RAG Application Configuration")
    print("=" * 40)
    
    for key, value in config.items():
        print(f"{key}: {value}")
    
    # Validate configuration
    print("\nValidating configuration...")
    issues = validate_config()
    
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("✅ Configuration is valid!")
# 🚀 Quick Start Guide - RAG Application

## Overview
A complete RAG application built with LangChain that supports multiple file formats (PDF, DOCX, CSV, TXT, JSON) and provides both programmatic and web interfaces.

## 🏃‍♂️ Quick Start

### 1. Setup (One-time)
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your OpenAI API key

# Run setup script
python setup.py
```

### 2. Run the Application

#### Option A: Web Interface (Recommended)
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

#### Option B: Command Line Examples
```bash
python run_examples.py
```

#### Option C: Programmatic Usage
```python
from rag_pipeline import RAGApplication

# Create application
app = RAGApplication()

# Create pipeline
pipeline = app.create_pipeline(
    name="my_pipeline",
    embedding_model="huggingface",  # or "openai"
    llm_model="gpt-3.5-turbo"
)

# Upload documents
results = pipeline.ingest_directory("./my_documents")

# Ask questions
answer = pipeline.ask_question("What is the main topic?")
print(answer['answer'])
```

## 📁 Project Structure

```
rag-application/
├── app.py                 # Streamlit web interface
├── rag_pipeline.py        # Main RAG pipeline
├── document_loaders.py    # Multi-format document loaders
├── vector_store.py        # ChromaDB vector store
├── config.py             # Configuration settings
├── example_usage.py      # Usage examples
├── test_rag.py          # Test suite
├── run_examples.py      # Example runner
├── setup.py             # Setup script
├── requirements.txt     # Dependencies
├── .env.example        # Environment template
└── README.md           # Documentation
```

## 🔧 Features

### Document Support
- **PDF**: PyPDF2 for text extraction
- **DOCX**: python-docx for Word documents
- **TXT**: Plain text files
- **CSV**: Pandas for structured data
- **JSON**: Structured JSON data

### Vector Storage
- **ChromaDB**: Persistent vector database
- **Embeddings**: OpenAI or HuggingFace models
- **Similarity Search**: Cosine similarity with configurable k

### LLM Integration
- **OpenAI GPT**: 3.5-turbo, GPT-4, GPT-4-turbo
- **Conversational Memory**: Context-aware conversations
- **Custom Prompts**: Configurable prompt templates

### Web Interface
- **Streamlit UI**: Modern, responsive interface
- **File Upload**: Drag-and-drop document upload
- **Real-time Chat**: Interactive Q&A
- **Document Search**: Browse and search documents

## 🎯 Use Cases

1. **Document Q&A**: Ask questions about uploaded documents
2. **Knowledge Base**: Build searchable knowledge repositories
3. **Research Assistant**: Analyze research papers and documents
4. **Customer Support**: Create AI-powered support systems
5. **Content Analysis**: Extract insights from large document collections

## 🔑 API Keys Required

- **OpenAI API Key**: For LLM generation (required for full functionality)
- **HuggingFace API Key**: Optional, for alternative embeddings

## 🧪 Testing

```bash
# Run all tests
python test_rag.py

# Run specific test categories
python -m unittest test_rag.TestDocumentLoaders
python -m unittest test_rag.TestVectorStore
```

## 📊 Performance Tips

1. **Chunk Size**: Adjust `chunk_size` in config for optimal performance
2. **Embedding Model**: Use HuggingFace for offline testing, OpenAI for production
3. **Vector Store**: ChromaDB persists data, so documents are remembered between sessions
4. **Memory Management**: Clear conversation memory for long-running sessions

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**: Make sure OPENAI_API_KEY is set in .env
2. **Import Errors**: Run `pip install -r requirements.txt`
3. **Memory Issues**: Clear vector store or restart application
4. **File Format Issues**: Check supported formats in config.py

### Getting Help

- Check the example_usage.py for detailed examples
- Run the test suite to verify installation
- Review the configuration in config.py
- Check the Streamlit logs for web interface issues

## 🚀 Next Steps

1. **Customize Prompts**: Edit prompts in config.py
2. **Add File Formats**: Extend document_loaders.py
3. **Deploy**: Use Streamlit Cloud or similar for deployment
4. **Scale**: Consider using more powerful embedding models
5. **Integrate**: Use the RAGPipeline class in your own applications

---

**Happy RAG-ing! 🤖📚**
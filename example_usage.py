"""
Example usage of the RAG application with different scenarios.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

from rag_pipeline import RAGApplication
from document_loaders import MultiFormatDocumentLoader, create_sample_documents

# Load environment variables
load_dotenv()


def example_basic_usage():
    """Basic usage example of the RAG application."""
    print("=== Basic RAG Usage Example ===\n")
    
    # Create RAG application
    app = RAGApplication()
    
    # Create a pipeline
    pipeline = app.create_pipeline(
        name="basic_example",
        embedding_model="huggingface",  # Use HuggingFace for testing
        llm_model="gpt-3.5-turbo"
    )
    
    # Create sample documents
    print("Creating sample documents...")
    sample_dir = create_sample_documents()
    
    # Ingest documents
    print("Ingesting documents...")
    results = pipeline.ingest_directory(sample_dir)
    print(f"Ingestion results: {results}\n")
    
    # Ask questions
    questions = [
        "What is artificial intelligence?",
        "What are the key concepts in machine learning?",
        "What programming languages are mentioned?",
        "Tell me about data science"
    ]
    
    print("Asking questions...")
    for question in questions:
        print(f"\nQ: {question}")
        answer = pipeline.ask_question(question)
        print(f"A: {answer['answer']}")
        if answer.get('source_documents'):
            print(f"Sources: {len(answer['source_documents'])} documents")
    
    return pipeline


def example_conversational_rag():
    """Example of conversational RAG with memory."""
    print("\n=== Conversational RAG Example ===\n")
    
    # Create RAG application
    app = RAGApplication()
    
    # Create a pipeline
    pipeline = app.create_pipeline(
        name="conversational_example",
        embedding_model="huggingface",
        llm_model="gpt-3.5-turbo"
    )
    
    # Create and ingest sample documents
    sample_dir = create_sample_documents()
    pipeline.ingest_directory(sample_dir)
    
    # Conversational questions
    conversation = [
        "What is machine learning?",
        "How does it relate to artificial intelligence?",
        "What about deep learning?",
        "Can you give me examples of each?"
    ]
    
    print("Conversational Q&A:")
    for question in conversation:
        print(f"\nQ: {question}")
        answer = pipeline.ask_question(question, use_conversation=True)
        print(f"A: {answer['answer']}")
    
    # Clear memory
    pipeline.clear_memory()
    print("\nMemory cleared!")


def example_document_search():
    """Example of document search functionality."""
    print("\n=== Document Search Example ===\n")
    
    # Create RAG application
    pipeline = example_basic_usage()
    
    # Search queries
    search_queries = [
        "programming languages",
        "neural networks",
        "statistics",
        "web development"
    ]
    
    print("Document search results:")
    for query in search_queries:
        print(f"\nSearching for: '{query}'")
        results = pipeline.search_documents_with_scores(query, k=2)
        
        for i, (doc, score) in enumerate(results):
            print(f"  Result {i+1} (Score: {score:.3f}):")
            print(f"    Content: {doc.page_content[:100]}...")
            print(f"    Metadata: {doc.metadata}")


def example_multiple_pipelines():
    """Example of managing multiple RAG pipelines."""
    print("\n=== Multiple Pipelines Example ===\n")
    
    # Create RAG application
    app = RAGApplication()
    
    # Create different pipelines for different topics
    tech_pipeline = app.create_pipeline(
        name="technology",
        embedding_model="huggingface",
        llm_model="gpt-3.5-turbo"
    )
    
    business_pipeline = app.create_pipeline(
        name="business",
        embedding_model="huggingface",
        llm_model="gpt-3.5-turbo"
    )
    
    # Create sample documents for each pipeline
    tech_docs = create_sample_documents()
    tech_pipeline.ingest_directory(tech_docs)
    
    # Create business sample documents
    business_dir = Path("business_documents")
    business_dir.mkdir(exist_ok=True)
    
    with open(business_dir / "business.txt", "w") as f:
        f.write("""
        Business Strategy and Management
        
        Business strategy is the long-term plan of action designed to achieve specific goals.
        It involves analyzing the competitive environment, understanding customer needs,
        and allocating resources effectively.
        
        Key components of business strategy include:
        - Market analysis and competitive positioning
        - Customer segmentation and targeting
        - Value proposition development
        - Resource allocation and budgeting
        - Performance measurement and KPIs
        """)
    
    business_pipeline.ingest_directory(str(business_dir))
    
    # Ask questions to different pipelines
    tech_questions = [
        "What is machine learning?",
        "Tell me about programming languages"
    ]
    
    business_questions = [
        "What is business strategy?",
        "What are the key components of business strategy?"
    ]
    
    print("Technology Pipeline Answers:")
    for question in tech_questions:
        answer = tech_pipeline.ask_question(question)
        print(f"Q: {question}")
        print(f"A: {answer['answer']}\n")
    
    print("Business Pipeline Answers:")
    for question in business_questions:
        answer = business_pipeline.ask_question(question)
        print(f"Q: {question}")
        print(f"A: {answer['answer']}\n")
    
    # List all pipelines
    print(f"Available pipelines: {app.list_pipelines()}")
    
    # Clean up
    import shutil
    shutil.rmtree(business_dir)


def example_custom_documents():
    """Example with custom document types."""
    print("\n=== Custom Documents Example ===\n")
    
    # Create RAG application
    app = RAGApplication()
    pipeline = app.create_pipeline(
        name="custom_docs",
        embedding_model="huggingface",
        llm_model="gpt-3.5-turbo"
    )
    
    # Create custom documents
    custom_dir = Path("custom_documents")
    custom_dir.mkdir(exist_ok=True)
    
    # Create a detailed technical document
    with open(custom_dir / "technical_guide.txt", "w") as f:
        f.write("""
        Advanced Machine Learning Techniques
        
        This guide covers advanced machine learning techniques used in modern AI systems.
        
        Deep Learning Architectures:
        1. Convolutional Neural Networks (CNNs)
           - Used for image recognition and computer vision
           - Key layers: Convolutional, Pooling, Fully Connected
           - Popular architectures: ResNet, VGG, Inception
        
        2. Recurrent Neural Networks (RNNs)
           - Designed for sequential data processing
           - Variants: LSTM, GRU, Bidirectional RNNs
           - Applications: NLP, time series analysis
        
        3. Transformer Networks
           - Attention mechanism for parallel processing
           - Models: BERT, GPT, T5
           - Revolutionized natural language processing
        
        Training Techniques:
        - Transfer Learning: Using pre-trained models
        - Data Augmentation: Increasing dataset diversity
        - Regularization: Preventing overfitting
        - Hyperparameter Tuning: Optimizing model performance
        
        Evaluation Metrics:
        - Classification: Accuracy, Precision, Recall, F1-Score
        - Regression: MSE, MAE, R²
        - Ranking: NDCG, MAP, MRR
        """)
    
    # Create a JSON document with structured data
    with open(custom_dir / "ml_models.json", "w") as f:
        json.dump([
            {
                "name": "Random Forest",
                "type": "Ensemble",
                "use_cases": ["Classification", "Regression"],
                "pros": ["Handles missing values", "Feature importance", "Robust to outliers"],
                "cons": ["Can overfit", "Memory intensive", "Black box"]
            },
            {
                "name": "Support Vector Machine",
                "type": "Supervised",
                "use_cases": ["Classification", "Regression"],
                "pros": ["Works well with high dimensions", "Memory efficient", "Versatile"],
                "cons": ["Slow on large datasets", "Sensitive to feature scaling", "No probabilistic output"]
            },
            {
                "name": "K-Means",
                "type": "Unsupervised",
                "use_cases": ["Clustering", "Dimensionality reduction"],
                "pros": ["Simple and fast", "Scales well", "Works with any distance metric"],
                "cons": ["Requires predefined k", "Sensitive to initialization", "Assumes spherical clusters"]
            }
        ], f, indent=2)
    
    # Create a CSV with model performance data
    with open(custom_dir / "model_performance.csv", "w") as f:
        f.write("""Model,Accuracy,Precision,Recall,F1_Score,Training_Time
Random Forest,0.95,0.94,0.96,0.95,120
SVM,0.92,0.91,0.93,0.92,45
Logistic Regression,0.89,0.88,0.90,0.89,15
Neural Network,0.97,0.96,0.98,0.97,300
Gradient Boosting,0.96,0.95,0.97,0.96,180
""")
    
    # Ingest documents
    print("Ingesting custom documents...")
    results = pipeline.ingest_directory(str(custom_dir))
    print(f"Ingestion results: {results}\n")
    
    # Ask detailed questions
    detailed_questions = [
        "What are the different types of neural networks?",
        "What are the pros and cons of Random Forest?",
        "Which model has the highest accuracy?",
        "What is transfer learning?",
        "What evaluation metrics are used for classification?"
    ]
    
    print("Detailed Q&A:")
    for question in detailed_questions:
        print(f"\nQ: {question}")
        answer = pipeline.ask_question(question)
        print(f"A: {answer['answer']}")
        if answer.get('source_documents'):
            print(f"Sources: {len(answer['source_documents'])} documents")
    
    # Clean up
    import shutil
    shutil.rmtree(custom_dir)


def example_error_handling():
    """Example of error handling in the RAG application."""
    print("\n=== Error Handling Example ===\n")
    
    # Create RAG application
    app = RAGApplication()
    pipeline = app.create_pipeline(
        name="error_example",
        embedding_model="huggingface",
        llm_model="gpt-3.5-turbo"
    )
    
    # Test with non-existent file
    print("Testing with non-existent file...")
    try:
        results = pipeline.ingest_documents(["non_existent_file.txt"])
        print(f"Results: {results}")
    except Exception as e:
        print(f"Error caught: {str(e)}")
    
    # Test with unsupported file format
    print("\nTesting with unsupported file format...")
    try:
        # Create a file with unsupported extension
        with open("test_file.xyz", "w") as f:
            f.write("This is a test file with unsupported format.")
        
        results = pipeline.ingest_documents(["test_file.xyz"])
        print(f"Results: {results}")
        
        # Clean up
        os.remove("test_file.xyz")
    except Exception as e:
        print(f"Error caught: {str(e)}")
    
    # Test with empty directory
    print("\nTesting with empty directory...")
    empty_dir = Path("empty_directory")
    empty_dir.mkdir(exist_ok=True)
    
    results = pipeline.ingest_directory(str(empty_dir))
    print(f"Results: {results}")
    
    # Clean up
    empty_dir.rmdir()


def main():
    """Run all examples."""
    print("RAG Application Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_usage()
        example_conversational_rag()
        example_document_search()
        example_multiple_pipelines()
        example_custom_documents()
        example_error_handling()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("Make sure you have set up your environment variables correctly.")


if __name__ == "__main__":
    main()
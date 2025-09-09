"""
Test suite for the RAG application.
"""

import os
import tempfile
import shutil
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

# Add current directory to Python path
import sys
sys.path.append(str(Path(__file__).parent))

from document_loaders import MultiFormatDocumentLoader
from vector_store import RAGVectorStore
from rag_pipeline import RAGPipeline, RAGApplication
from config import get_config, validate_config


class TestDocumentLoaders(unittest.TestCase):
    """Test cases for document loaders."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = MultiFormatDocumentLoader()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_supported_formats(self):
        """Test that supported formats are correctly identified."""
        expected_formats = {'.pdf', '.docx', '.txt', '.csv', '.json'}
        actual_formats = set(self.loader.supported_formats.keys())
        self.assertEqual(actual_formats, expected_formats)
    
    def test_txt_loading(self):
        """Test loading of text files."""
        # Create a test text file
        test_file = Path(self.temp_dir) / "test.txt"
        test_content = "This is a test document.\n\nIt has multiple paragraphs."
        
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)
        
        # Load the document
        documents = self.loader.load_document(str(test_file))
        
        # Verify results
        self.assertIsInstance(documents, list)
        self.assertGreater(len(documents), 0)
        
        for doc in documents:
            self.assertIsInstance(doc.page_content, str)
            self.assertIsInstance(doc.metadata, dict)
    
    def test_json_loading(self):
        """Test loading of JSON files."""
        # Create a test JSON file
        test_file = Path(self.temp_dir) / "test.json"
        test_data = [
            {"title": "Test 1", "content": "This is test content 1"},
            {"title": "Test 2", "content": "This is test content 2"}
        ]
        
        import json
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
        
        # Load the document
        documents = self.loader.load_document(str(test_file))
        
        # Verify results
        self.assertIsInstance(documents, list)
        self.assertGreater(len(documents), 0)
    
    def test_csv_loading(self):
        """Test loading of CSV files."""
        # Create a test CSV file
        test_file = Path(self.temp_dir) / "test.csv"
        test_content = "Name,Age,City\nJohn,30,New York\nJane,25,London"
        
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)
        
        # Load the document
        documents = self.loader.load_document(str(test_file))
        
        # Verify results
        self.assertIsInstance(documents, list)
        self.assertGreater(len(documents), 0)
    
    def test_unsupported_format(self):
        """Test handling of unsupported file formats."""
        # Create a file with unsupported extension
        test_file = Path(self.temp_dir) / "test.xyz"
        with open(test_file, "w") as f:
            f.write("Test content")
        
        # Attempt to load the document
        with self.assertRaises(ValueError):
            self.loader.load_document(str(test_file))
    
    def test_nonexistent_file(self):
        """Test handling of non-existent files."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.txt"
        
        with self.assertRaises(FileNotFoundError):
            self.loader.load_document(str(nonexistent_file))
    
    def test_directory_loading(self):
        """Test loading documents from a directory."""
        # Create multiple test files
        files = {
            "test1.txt": "Content of test file 1",
            "test2.txt": "Content of test file 2",
            "test.json": '[{"content": "JSON content"}]'
        }
        
        for filename, content in files.items():
            file_path = Path(self.temp_dir) / filename
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        
        # Load documents from directory
        documents = self.loader.load_documents_from_directory(self.temp_dir)
        
        # Verify results
        self.assertIsInstance(documents, list)
        self.assertGreater(len(documents), 0)


class TestVectorStore(unittest.TestCase):
    """Test cases for vector store."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = RAGVectorStore(
            persist_directory=self.temp_dir,
            embedding_model="huggingface"  # Use HuggingFace for testing
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_vector_store_initialization(self):
        """Test vector store initialization."""
        self.assertIsNotNone(self.vector_store.vector_store)
        self.assertIsNotNone(self.vector_store.embeddings)
    
    def test_add_texts(self):
        """Test adding texts to vector store."""
        texts = [
            "This is a test document about artificial intelligence.",
            "Machine learning is a subset of AI.",
            "Deep learning uses neural networks."
        ]
        
        metadatas = [
            {"source": "test1.txt"},
            {"source": "test2.txt"},
            {"source": "test3.txt"}
        ]
        
        # Add texts
        doc_ids = self.vector_store.add_texts(texts, metadatas)
        
        # Verify results
        self.assertIsInstance(doc_ids, list)
        self.assertEqual(len(doc_ids), len(texts))
    
    def test_similarity_search(self):
        """Test similarity search functionality."""
        # Add test documents
        texts = [
            "Artificial intelligence is the simulation of human intelligence.",
            "Machine learning algorithms learn from data.",
            "Deep learning uses neural networks with multiple layers."
        ]
        
        self.vector_store.add_texts(texts)
        
        # Perform similarity search
        results = self.vector_store.similarity_search("What is AI?", k=2)
        
        # Verify results
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)
    
    def test_similarity_search_with_scores(self):
        """Test similarity search with scores."""
        # Add test documents
        texts = ["Test document about machine learning."]
        self.vector_store.add_texts(texts)
        
        # Perform similarity search with scores
        results = self.vector_store.similarity_search_with_score("machine learning", k=1)
        
        # Verify results
        self.assertIsInstance(results, list)
        if results:
            doc, score = results[0]
            self.assertIsInstance(score, (int, float))
    
    def test_get_collection_info(self):
        """Test getting collection information."""
        info = self.vector_store.get_collection_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn("collection_name", info)
        self.assertIn("document_count", info)


class TestRAGPipeline(unittest.TestCase):
    """Test cases for RAG pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = RAGVectorStore(
            persist_directory=self.temp_dir,
            embedding_model="huggingface"
        )
        
        # Mock the LLM to avoid API calls
        with patch('rag_pipeline.ChatOpenAI') as mock_llm:
            mock_llm.return_value = MagicMock()
            self.pipeline = RAGPipeline(
                vector_store=self.vector_store,
                llm_model="gpt-3.5-turbo"
            )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline.vector_store)
        self.assertIsNotNone(self.pipeline.llm)
        self.assertIsNotNone(self.pipeline.qa_chain)
    
    def test_ingest_documents(self):
        """Test document ingestion."""
        # Create test files
        test_files = []
        for i in range(2):
            test_file = Path(self.temp_dir) / f"test{i}.txt"
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(f"Test document {i} content.")
            test_files.append(str(test_file))
        
        # Ingest documents
        results = self.pipeline.ingest_documents(test_files)
        
        # Verify results
        self.assertIsInstance(results, dict)
        self.assertIn("total_documents", results)
        self.assertIn("successful_documents", results)
        self.assertIn("failed_documents", results)
    
    def test_ask_question(self):
        """Test asking questions."""
        # Add test documents
        texts = ["Artificial intelligence is the simulation of human intelligence."]
        self.vector_store.add_texts(texts)
        
        # Mock the QA chain response
        with patch.object(self.pipeline.qa_chain, '__call__') as mock_qa:
            mock_qa.return_value = {
                "result": "AI is the simulation of human intelligence.",
                "source_documents": []
            }
            
            # Ask a question
            response = self.pipeline.ask_question("What is AI?")
            
            # Verify results
            self.assertIsInstance(response, dict)
            self.assertIn("question", response)
            self.assertIn("answer", response)
            self.assertIn("success", response)
    
    def test_search_documents(self):
        """Test document search."""
        # Add test documents
        texts = ["Machine learning is a subset of artificial intelligence."]
        self.vector_store.add_texts(texts)
        
        # Search documents
        results = self.pipeline.search_documents("machine learning", k=1)
        
        # Verify results
        self.assertIsInstance(results, list)


class TestRAGApplication(unittest.TestCase):
    """Test cases for RAG application."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.app = RAGApplication(base_persist_directory=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_application_initialization(self):
        """Test application initialization."""
        self.assertIsNotNone(self.app.vector_store_manager)
        self.assertIsInstance(self.app.pipelines, dict)
    
    def test_create_pipeline(self):
        """Test pipeline creation."""
        with patch('rag_pipeline.RAGPipeline') as mock_pipeline:
            mock_pipeline.return_value = MagicMock()
            
            pipeline = self.app.create_pipeline(
                name="test_pipeline",
                embedding_model="huggingface"
            )
            
            self.assertIsNotNone(pipeline)
            self.assertIn("test_pipeline", self.app.pipelines)
    
    def test_list_pipelines(self):
        """Test listing pipelines."""
        # Create a test pipeline
        with patch('rag_pipeline.RAGPipeline') as mock_pipeline:
            mock_pipeline.return_value = MagicMock()
            self.app.create_pipeline("test_pipeline", "huggingface")
            
            pipelines = self.app.list_pipelines()
            self.assertIn("test_pipeline", pipelines)


class TestConfiguration(unittest.TestCase):
    """Test cases for configuration."""
    
    def test_get_config(self):
        """Test getting configuration."""
        config = get_config()
        
        self.assertIsInstance(config, dict)
        self.assertIn("supported_formats", config)
        self.assertIn("embedding_models", config)
        self.assertIn("llm_models", config)
    
    def test_validate_config(self):
        """Test configuration validation."""
        issues = validate_config()
        self.assertIsInstance(issues, list)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDocumentLoaders,
        TestVectorStore,
        TestRAGPipeline,
        TestRAGApplication,
        TestConfiguration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running RAG Application Tests")
    print("=" * 40)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    sys.exit(0 if success else 1)
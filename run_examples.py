"""
Quick script to run the RAG application examples.
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

def check_environment():
    """Check if the environment is properly set up."""
    print("Checking environment setup...")
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("⚠️  .env file not found. Please copy .env.example to .env and set your API keys.")
        return False
    
    # Check if required packages are installed
    try:
        import streamlit
        import langchain
        import chromadb
        import openai
        print("✅ All required packages are installed.")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def main():
    """Main function to run examples."""
    print("RAG Application - Example Runner")
    print("=" * 40)
    
    if not check_environment():
        print("\nPlease fix the environment issues before running examples.")
        return
    
    print("\nChoose an option:")
    print("1. Run basic examples (no API key required)")
    print("2. Run web interface")
    print("3. Run all examples (requires OpenAI API key)")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\nRunning basic examples...")
        from example_usage import example_basic_usage, example_document_search
        example_basic_usage()
        example_document_search()
        
    elif choice == "2":
        print("\nStarting web interface...")
        print("The web interface will open in your browser.")
        print("Press Ctrl+C to stop the server.")
        os.system("streamlit run app.py")
        
    elif choice == "3":
        print("\nRunning all examples...")
        from example_usage import main as run_all_examples
        run_all_examples()
        
    elif choice == "4":
        print("Goodbye!")
        
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
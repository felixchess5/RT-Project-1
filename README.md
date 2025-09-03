# Research Assistant RAG System

A Retrieval-Augmented Generation (RAG) system that processes research publications and provides intelligent answers to research questions using ChromaDB vector storage and LLaMA language models.

## Features

- **Document Processing**: Automatically loads and processes research publications from text files
- **Vector Storage**: Uses ChromaDB for efficient similarity search with cosine distance
- **Smart Chunking**: Splits documents into manageable chunks with overlap for context preservation
- **Semantic Search**: Leverages HuggingFace embeddings for semantic document retrieval
- **AI-Powered Answers**: Uses Groq's LLaMA-3.1-8B model for generating research-based responses
- **Interactive CLI**: Simple command-line interface for asking research questions

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster embeddings)
- Groq API key
- HuggingFace API token (optional)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RT-Project-1
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp example.env_example .env
```

4. Edit `.env` file with your API keys and configuration:
```env
GROQ_API_KEY='your_groq_api_key'
HUGGINGFACEHUB_API_TOKEN='your_hf_token'
DOCUMENTS_PATH='./Documents'
DB='./DB'
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## Usage

1. **Add Research Documents**: Place your research papers (`.txt` format) in the `Documents` directory

2. **Run the System**:
```bash
python main.py
```

3. **Ask Questions**: The system will prompt you to ask research questions. Type your question and get AI-powered answers based on your document collection.

4. **Exit**: Type 'exit' or 'quit' to close the application

## Architecture

- **main.py**: Core application logic and CLI interface
- **chroma_loader.py**: Document processing, embeddings, and ChromaDB operations
- **Documents/**: Directory for research publication text files
- **DB/**: ChromaDB persistent storage location

## How It Works

1. **Document Ingestion**: Text files are loaded from the Documents directory
2. **Text Chunking**: Documents are split into 1000-character chunks with 200-character overlap
3. **Embedding Generation**: Each chunk is converted to embeddings using sentence-transformers
4. **Vector Storage**: Embeddings are stored in ChromaDB with metadata
5. **Query Processing**: User questions are embedded and matched against stored chunks
6. **Answer Generation**: Top relevant chunks are used as context for LLaMA model to generate answers

## Configuration

Adjust settings in your `.env` file:
- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `DOCUMENTS_PATH`: Path to research documents
- `DB`: ChromaDB storage location

## Ready Tensor Certification Project 1

This project demonstrates proficiency in:
- Vector databases and similarity search
- Language model integration
- Document processing and chunking
- RAG system architecture
- Python development best practices
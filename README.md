# Research Assistant RAG System

A production-ready Retrieval-Augmented Generation (RAG) system that processes research publications and provides intelligent answers to research questions using ChromaDB vector storage and Groq's LLaMA language models.

## 🚀 Features

- **📄 Document Processing**: Automatically loads and processes research publications from text files
- **🔍 Vector Storage**: Uses ChromaDB 1.0.20 for efficient similarity search with cosine distance
- **✂️ Smart Chunking**: Splits documents into manageable chunks with overlap for context preservation
- **🧠 Semantic Search**: Leverages HuggingFace sentence-transformers for semantic document retrieval
- **🤖 AI-Powered Answers**: Uses Groq's LLaMA-3.1-8B-Instant model for generating research-based responses
- **💬 Interactive CLI**: Robust command-line interface with error handling and graceful exits
- **⚡ Hardware Optimization**: Automatic device detection (CUDA/MPS/CPU) for optimal performance

## 📋 Prerequisites

- **Python 3.8+** (tested with Python 3.11)
- **Groq API Key** (required) - Get from [console.groq.com](https://console.groq.com/)
- **HuggingFace API Token** (optional) - For enhanced model access
- **CUDA-compatible GPU** (optional) - For faster embedding generation

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/felixchess5/RT-Project-1.git
cd RT-Project-1
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
# Copy the example environment file
cp example.env_example .env
```

Edit the `.env` file with your actual API keys:
```env
# Required: Get your Groq API key from https://console.groq.com/
GROQ_API_KEY='your_actual_groq_api_key_here'

# Optional: HuggingFace token for enhanced model access
HUGGINGFACEHUB_API_TOKEN='your_hf_token_here'

# Project configuration (default values work fine)
DOCUMENTS_PATH='./Documents'
DB='./DB'
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

> ⚠️ **Important**: Never commit your `.env` file with real API keys to version control!

## 🚀 Usage

### 1. Prepare Your Research Documents
Place your research papers in `.txt` format in the `Documents/` directory. The system comes pre-loaded with 10 research papers on:
- Large Language Models
- RAG systems
- Machine Learning techniques
- Neural architectures

### 2. Run the RAG System
```bash
python -m main
# or
python main.py
```

### 3. Interactive Q&A Session
The system will:
- Automatically populate ChromaDB on first run (one-time setup)
- Display: "Ask a research question (type 'exit' to quit):"
- Process your questions and provide AI-powered answers with source citations

### 4. Example Queries
```
What are the main challenges in RAG systems?
How do transformer architectures improve language models?
What techniques are used for fine-tuning LLMs?
Explain the concept of attention mechanisms in neural networks.
```

### 5. Exit the Application
Type `exit`, `quit`, or press `Ctrl+C` to close the application gracefully.

## 📁 Project Structure

```
RT-Project-1/
├── main.py                    # Core application logic and CLI interface
├── chroma_loader.py           # Document processing, embeddings, and ChromaDB operations
├── requirements.txt           # Python dependencies (organized by category)
├── .env                       # Environment variables (create from example)
├── example.env_example        # Environment template
├── .gitignore                # Git ignore rules (protects .env, DB/, etc.)
├── README.md                 # This file
├── Documents/                # Research papers (.txt format)
│   ├── Brain LLM.txt
│   ├── LLM NEO.txt
│   ├── Selectionless RAG.txt
│   └── ... (7 more papers)
├── DB/                       # ChromaDB persistent storage (auto-created)
└── Deprecated Code/          # Legacy implementations
```

## ⚙️ How It Works

### RAG Pipeline
1. **📖 Document Ingestion**: Text files are loaded from the `Documents/` directory
2. **✂️ Text Chunking**: Documents are split into 1000-character chunks with 200-character overlap
3. **🔢 Embedding Generation**: Each chunk is converted to embeddings using `sentence-transformers/all-MiniLM-L6-v2`
4. **💾 Vector Storage**: Embeddings are stored in ChromaDB with metadata for fast similarity search
5. **🔍 Query Processing**: User questions are embedded and matched against stored chunks using cosine similarity
6. **🤖 Answer Generation**: Top 3 relevant chunks are used as context for Groq's LLaMA-3.1-8B-Instant model

### Key Technologies
- **ChromaDB 1.0.20**: Vector database for semantic search
- **LangChain**: Framework for LLM applications and document processing
- **Groq API**: Fast LLM inference with LLaMA-3.1-8B-Instant
- **HuggingFace Transformers**: Sentence embeddings and model handling
- **PyTorch**: Machine learning framework with automatic device detection

## 🔧 Configuration

Adjust settings in your `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Your Groq API key (required) | `''` |
| `HUGGINGFACEHUB_API_TOKEN` | HuggingFace API token (optional) | `''` |
| `DOCUMENTS_PATH` | Path to research documents | `'./Documents'` |
| `DB` | ChromaDB storage location | `'./DB'` |
| `CHUNK_SIZE` | Size of text chunks | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |

## 🏆 Ready Tensor Certification Project 1

This project demonstrates proficiency in:

### Technical Skills
- ✅ **Vector Databases**: ChromaDB implementation with cosine similarity search
- ✅ **Language Model Integration**: Groq API usage with proper error handling
- ✅ **Document Processing**: Text chunking and embedding generation
- ✅ **RAG System Architecture**: End-to-end retrieval-augmented generation pipeline
- ✅ **Python Development**: Clean code, proper error handling, environment management

### Software Engineering
- ✅ **Environment Management**: Secure API key handling and configuration
- ✅ **Version Control**: Proper `.gitignore` excluding sensitive files
- ✅ **Documentation**: Comprehensive README with installation and usage instructions
- ✅ **Dependency Management**: Organized `requirements.txt` with version pinning
- ✅ **Error Handling**: Graceful error handling and user-friendly messages

## 🚨 Troubleshooting

### Common Issues

**"ModuleNotFoundError"**: Install dependencies with `pip install -r requirements.txt`

**"GROQ_API_KEY is empty"**: Add your actual API key to the `.env` file

**"No documents found"**: Ensure `.txt` files are in the `Documents/` directory

**"Unicode encode error"**: The system now handles Windows console encoding issues gracefully

**"EOFError"**: This is expected when running in non-interactive environments

## 📄 License

MIT License © 2025 Felix Elias

## 🤝 Contributing

Pull requests welcome! For major changes, please open an issue first to discuss what you'd like to change.
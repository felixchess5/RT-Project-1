# 🧠 ML Research Assistant

A modular AI-powered assistant for ingesting, chunking, embedding, and querying machine learning research papers using LangChain, ChromaDB, HuggingFace embeddings, and Groq LLM.

---

## 🚀 Features

- ✅ Chunk and embed `.txt` research papers
- ✅ Store semantic vectors in ChromaDB
- ✅ Query with natural language and retrieve relevant findings
- ✅ Generate answers using Groq's LLaMA-3.1-8B-Instant
- ✅ Modular architecture with clean separation of ingestion and inference

---

## 📦 Installation

```bash
git clone https://github.com/your-username/ml-research-assistant.git
cd ml-research-assistant
pip install -r requirements.txt

Note: rename example.env_example -> .env and add missing information

📁 Project Structure
ml-research-assistant/
├── chroma_loader.py       # Loads, chunks, embeds, and stores documents
├── main.py                # Interactive Q&A interface
├── documents/             # Folder containing .txt research papers
├── .env                   # Environment variables
├── requirements.txt       # Python dependencies
└── README.md              # You're reading it!

🧪 Usage
python -u main.py

Then ask questions like:
- What are effective techniques for handling class imbalance?
- Summarize recent advances in transformer architectures.
- How does dropout improve generalization?
- exit [to exit application]

📜 License
MIT License © 2025 Felix Elias

🤝 Contributing
Pull requests welcome! For major changes, please open an issue first to discuss what you’d like to change.


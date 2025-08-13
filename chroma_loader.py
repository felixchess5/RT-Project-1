import os
import torch
import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

# Initialize ChromaDB
client = chromadb.PersistentClient(path=os.getenv("DB"))
collection = client.get_or_create_collection(
    name="ml_publications",
    metadata={"hnsw:space": "cosine"}
)

def load_research_publications(path) -> list[Document]:
    documents = []
    for file in os.listdir(path):
        if file.endswith(".txt"):
            file_path = os.path.join(path, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                doc = Document(page_content=text, metadata={"source": file})
                documents.append(doc)
                print(f"Loaded: {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    return documents

def chunk_research_paper(paper_content, title):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(paper_content)
    return [
        {
            "content": chunk,
            "title": title,
            "chunk_id": f"{title}_{i}"
        }
        for i, chunk in enumerate(chunks)
    ]

def get_embedding_model():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

def store_chunks_in_chromadb(chunks, embedding_model):
    texts = [chunk["content"] for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]
    metadatas = [{"title": chunk["title"]} for chunk in chunks]
    embeddings = embedding_model.embed_documents(texts)
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

def populate_chroma():
    docs = load_research_publications(os.getenv("DOCUMENTS_PATH"))
    if not docs:
        print("No documents found.")
        return
    embedding_model = get_embedding_model()
    for doc in docs:
        title = doc.metadata.get("source", "untitled").replace(".txt", "")
        chunks = chunk_research_paper(doc.page_content, title)
        store_chunks_in_chromadb(chunks, embedding_model)
        print(f"Stored chunks for: {title}")

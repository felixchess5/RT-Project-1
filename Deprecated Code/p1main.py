import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Groq
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

def load_documents(path: str):
    """Recursively load .txt files from a directory."""
    docs = []
    for root, _, files in os.walk(path):
        for fn in files:
            full = os.path.join(root, fn)
            if fn.lower().endswith(".txt"):
                docs.extend(TextLoader(full).load())
    return docs

def build_groq_index(docs, embeddings, config):
    """Embed documents and upsert into Groq vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(config["CHUNK_SIZE"]),
        chunk_overlap=int(config["CHUNK_OVERLAP"]),
    )
    chunks = splitter.split_documents(docs)

    store = Groq.from_documents(
        documents=chunks,
        embedding=embeddings,
        api_key=config["GROQ_API_KEY"],
        project=config["GROQ_PROJECT"],
        url=config["GROQ_URL"],
    )
    return store

def run_qa(retriever, llm):
    """Simple REPL loop for question answering."""
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )
    print("🚀 RAG Assistant ready!  (type 'exit' to quit)")
    while True:
        q = input("\n> ")
        if q.strip().lower() in {"exit", "quit"}:
            break
        print("\n" + qa.run(q) + "\n")

def main():
    load_dotenv()

    # Load config
    cfg = {
        k: os.getenv(k)
        for k in [
            "GROQ_API_KEY", "GROQ_PROJECT", "GROQ_URL",
            "HUGGINGFACEHUB_API_TOKEN",
            "DOCS_PATH", "CHUNK_SIZE", "CHUNK_OVERLAP"
        ]
    }

    # 1. Load + split docs
    print("🔍 Loading documents...")
    documents = load_documents(cfg["DOCS_PATH"])

    # 2. Initialize embeddings
    print("🔗 Initializing HF embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=cfg["HUGGINGFACEHUB_API_TOKEN"],
    )

    # 3. Build or update Groq index
    print("📦 Building Groq vector store...")
    groq_store = build_groq_index(documents, embeddings, cfg)

    # 4. Create retriever + LLM
    retriever = groq_store.as_retriever(search_kwargs={"k": 4})
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-small",
        model_kwargs={"temperature": 0.1},
        huggingfacehub_api_token=cfg["HUGGINGFACEHUB_API_TOKEN"],
    )

    # 5. Start QA loop
    run_qa(retriever, llm)

if __name__ == "__main__":
    main()


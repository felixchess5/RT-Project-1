import os
import chromadb
import torch
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import yaml
from langchain.prompts import PromptTemplate

load_dotenv()

# Initialize LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # Fast and capable
    temperature=0.7,
    api_key=os.getenv('GROQ_API_KEY')
)
# Initialize ChromaDB
client = chromadb.PersistentClient(path=os.getenv("DB"))
collection = client.get_or_create_collection(
    name="ml_publications",
    metadata={"hnsw:space": "cosine"}
)


def chunk_research_paper(paper_content, title):
    """Break a research paper into searchable chunks"""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,          # ~200 words per chunk
        chunk_overlap=200,        # Overlap to preserve context
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_text(paper_content)

    # Add metadata to each chunk
    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            "content": chunk,
            "title": title,
            "chunk_id": f"{title}_{i}",
        })

    return chunk_data

# Store embeddings in chromadb
def store_chunks_in_chromadb(chunks, embeddings_model):
    texts = [chunk["content"] for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]
    metadatas = [{"title": chunk["title"]} for chunk in chunks]
    embeddings = embeddings_model.embed_documents(texts)

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )


# # Set up our embedding model
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

def load_research_publications(path) -> list:
    # Load research publications from .txt files and return as list of strings

    # List to store all documents
    documents = []

    # Load each .txt file in the documents folder
    for file in os.listdir(path):
        if file.endswith(".txt"):
            file_path = os.path.join(path, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                doc = Document(page_content=text, metadata={"source": file})
                documents.append(doc)
                print(f"Successfully loaded: {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")

    return documents


def embed_documents(documents: list[str]) -> list[list[float]]:
    """
    Embed documents using a model.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )

    embeddings = model.embed_documents(documents)
    return embeddings

def search_research_db(query, collection, embeddings, top_k=5):
    """Find the most relevant research chunks for a query"""

    # Convert question to vector
    query_vector = embeddings.embed_query(query)

    # Search for similar content
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Format results
    relevant_chunks = []
    for i, doc in enumerate(results["documents"][0]):
        relevant_chunks.append({
            "content": doc,
            "title": results["metadatas"][0][i]["title"],
            "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
        })

    return relevant_chunks

def answer_research_question(query, collection, embeddings, llm):
    """Generate an answer based on retrieved research"""

    # Get relevant research chunks
    relevant_chunks = search_research_db(query, collection, embeddings, top_k=3)

    # Build context from research
    context = "\n\n".join([
        f"From {chunk['title']}:\n{chunk['content']}"
        for chunk in relevant_chunks
    ])

    # Create research-focused prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Based on the following research findings, answer the researcher's question:

Research Context:
{context}

Researcher's Question: {question}

Answer: Provide a comprehensive answer based on the research findings above.
"""
    )

    # Generate answer
    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)

    return response.content, relevant_chunks

def runner():
    docs = load_research_publications(path=os.getenv("DOCUMENTS_PATH"))
    if not docs:
        print("No documents found. Exiting.")
        return
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )

    for doc in docs:
        paper_content = doc.page_content
        title = doc.metadata.get("source", "untitled").replace(".txt", "")
        chunks = chunk_research_paper(paper_content, title)
        store_chunks_in_chromadb(chunks, embedding_model)
        print(f"Stored chunks for: {title}")

    # Initialize LLM and get answer
    llm = ChatGroq(model="llama3-8b-8192")
    answer, sources = answer_research_question(
        "What are effective techniques for handling class imbalance?",
        collection,
        embedding_model,
        llm
    )
    print("AI Answer:", answer)
    print("\nBased on sources:")
    for source in sources:
        print(f"- {source['title']}")

if __name__=='__main__':
    runner()
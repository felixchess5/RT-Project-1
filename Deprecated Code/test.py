from dotenv import load_dotenv
import os, chromadb
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import torch

def load_research_publications(documents_path):
    """Load research publications from .txt files and return as list of strings"""

    # List to store all documents
    documents = []

    # Load each .txt file in the documents folder
    for file in os.listdir(documents_path):
        if file.endswith(".txt"):
            file_path = os.path.join(documents_path, file)
            try:
                loader = TextLoader(file_path)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                print(f"Successfully loaded: {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")

    print(f"\nTotal documents loaded: {len(documents)}")

    # Extract content as strings and return
    publications = []
    for doc in documents:
        publications.append(doc.page_content)
    return publications

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

def insert_publications(collection: chromadb.Collection, publications: list[str]):
    """
    Insert documents into a ChromaDB collection.

    Args:
        collection (chromadb.Collection): The collection to insert documents into
        publications (list[str]): The documents to insert

    Returns:
        None
    """
    next_id = collection.count()

    for publication in publications:
        chunked_publication = chunk_publication(publication, title="")
        embeddings = embed_documents(chunked_publication)
        ids = list(range(next_id, next_id + len(chunked_publication)))
        ids = [f"document_{id}" for id in ids]
        collection.add(
            embeddings=embeddings,
            ids=ids,
            documents=chunked_publication,
        )
        next_id += len(chunked_publication)
    return embeddings

def chunk_publication(paper_content, title):
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

def runner():
    load_dotenv()
    client = chromadb.PersistentClient(path=os.getenv("DB"))
    collection = client.get_or_create_collection("ml_pubs", metadata={"hnsw:space":"cosine"})

    # 1. Load raw .txt files
    publications = load_research_publications(os.getenv("DOCUMENTS_PATH"))

    # 2. Insert into Chroma
    embeddings = insert_publications(collection, publications)

    # 3. Confirm data landed
    assert collection.count() > 0, "No chunks found in collection!"

    # Initialize LLM and get answer
    load_dotenv()

    llm = ChatGroq(model="llama3-8b-8192")
    answer, sources = answer_research_question(
        "What are effective techniques for handling class imbalance?",
        collection,
        embeddings,
        llm
    )

    print("AI Answer:", answer)
    print("\nBased on sources:")
    for source in sources:
        print(f"- {source['title']}")
        # 4. Ask a question
        answer, sources = answer_research_question(
            "What are effective techniques for handling class imbalance?",
            collection,
            embeddings,
            llm
        )

    print("Answer:\n", answer)
    print("Sources:", [s["title"] for s in sources])
    
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

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

if __name__ == '__main__':
    runner()
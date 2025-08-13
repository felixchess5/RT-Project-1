import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from chroma_loader import (
    collection,
    get_embedding_model,
    populate_chroma
)
from langchain.prompts import PromptTemplate

load_dotenv()

def search_research_db(query, collection, embeddings, top_k=5):
    query_vector = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    if not results["documents"] or not results["documents"][0]:
        return []
    return [
        {
            "content": doc,
            "title": results["metadatas"][0][i]["title"],
            "similarity": 1 - results["distances"][0][i]
        }
        for i, doc in enumerate(results["documents"][0])
    ]

def answer_research_question(query, collection, embeddings, llm):
    relevant_chunks = search_research_db(query, collection, embeddings, top_k=3)
    if not relevant_chunks:
        return "No relevant research findings found.", []

    context = "\n\n".join([
        f"From {chunk['title']}:\n{chunk['content']}"
        for chunk in relevant_chunks
    ])

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

    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)
    return response.content, relevant_chunks

def runner():
    if collection.count() == 0:
        print("Populating ChromaDB...")
        populate_chroma()

    embedding_model = get_embedding_model()
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        api_key=os.getenv('GROQ_API_KEY')
    )

    print("🔍 Ask a research question (type 'exit' to quit):")
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        answer, sources = answer_research_question(query, collection, embedding_model, llm)
        print("\n🧠 AI Answer:\n", answer)
        print("\n📚 Based on sources:")
        unique_titles = sorted(set(chunk["title"] for chunk in sources))
        for title in unique_titles:
            print(f"- {title}")



if __name__ == "__main__":
    runner()
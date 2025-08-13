import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from chroma_loader import collection, get_embedding_model, populate_chroma

load_dotenv()

def search_research_db(query, collection, embeddings, top_k=5):
    query_vector = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
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
    context = "\n\n".join([
        f"From {chunk['title']}:\n{chunk['content']}"
        for chunk in relevant_chunks
    ])
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=
        """
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
    embedding_model = get_embedding_model()
    populate_chroma()  # ✅ Ensure DB is populated
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        api_key=os.getenv('GROQ_API_KEY')
    )
    # print("Collection count:", collection.count())

    query = "What are effective techniques for handling class imbalance?"
    answer, sources = answer_research_question(query, collection, embedding_model, llm)
    print("AI Answer:\n", answer)
    print("\nBased on sources:")
    for source in sources:
        print(f"- {source['title']}")

if __name__ == "__main__":
    runner()
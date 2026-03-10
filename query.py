import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import pipeline


# Paths
VECTOR_STORE_PATH = "vector_store"

# Settings
TOP_K = 3


print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


print("Loading generator model (FLAN-T5)...")
generator = pipeline(
    "text-generation",
    model="google/flan-t5-base"
)


# Load FAISS index and metadata
def load_vector_store():

    index = faiss.read_index(VECTOR_STORE_PATH + "/index.faiss")

    with open(VECTOR_STORE_PATH + "/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


# Semantic search
def search(query, index, metadata):

    query_embedding = embed_model.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, TOP_K)

    results = []

    for i in indices[0]:
        results.append(metadata[i])

    return results


# TRUE RAG: Retrieval + Generation
def generate_answer(query, results):

    context = ""
    citations = ""

    for r in results:

        text = r.get("text", "")
        file = r.get("file", "Unknown")
        page = r.get("page", "?")

        context += text + "\n\n"

        citations += f"{file} (Page {page})\n"


    prompt = f"""
Answer the question based on the context below.

Context:
{context}

Question:
{query}

Provide a clear and accurate answer.
"""


    output = generator(prompt)

    answer = output[0]['generated_text']

    return answer, citations


# Main loop
def main():

    index, metadata = load_vector_store()

    print("\nPersonal Research AI Ready (Offline RAG)")
    print("----------------------------------------")

    while True:

        query = input("\nEnter your question (or type exit): ")

        if query.lower() == "exit":
            break

        results = search(query, index, metadata)

        answer, citations = generate_answer(query, results)

        print("\nAnswer:\n")
        print(answer)

        print("\nCitations:")
        print(citations)


if __name__ == "__main__":
    main()

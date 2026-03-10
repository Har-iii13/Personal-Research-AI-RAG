import os
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import pickle

DATA_PATH = "data"
VECTOR_STORE_PATH = "vector_store"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text(pdf_path):

    reader = PdfReader(pdf_path)

    pages = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append((page_num+1, text))

    return pages


def chunk_text(pages, filename):

    chunks = []
    metadata = []

    for page_num, text in pages:

        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE

            chunk = text[start:end]

            chunks.append(chunk)

            metadata.append({
                "text": chunk,
                "file": filename,
                "page": page_num
            })

            start = end - CHUNK_OVERLAP

    return chunks, metadata


def main():

    all_chunks = []
    all_metadata = []

    for file in os.listdir(DATA_PATH):

        if file.endswith(".pdf"):

            print("Processing:", file)

            pages = extract_text(os.path.join(DATA_PATH, file))

            chunks, metadata = chunk_text(pages, file)

            all_chunks.extend(chunks)
            all_metadata.extend(metadata)


    embeddings = model.encode(all_chunks)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)


    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    faiss.write_index(index, VECTOR_STORE_PATH+"/index.faiss")

    with open(VECTOR_STORE_PATH+"/metadata.pkl", "wb") as f:
        pickle.dump(all_metadata, f)


    print("Vector store created successfully")


if __name__ == "__main__":
    main()

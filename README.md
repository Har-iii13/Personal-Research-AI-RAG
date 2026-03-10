# Personal Research AI using Retrieval-Augmented Generation (RAG)

## Problem Statement

Researchers and students often work with multiple academic papers and documents. Searching through these documents manually to find relevant information is time-consuming and inefficient. Traditional keyword search methods fail to understand semantic meaning and context.

This project aims to develop a **Knowledge-Aware AI System using Retrieval-Augmented Generation (RAG)** that can read multiple research PDFs, store them as embeddings in a vector database, and answer user queries by retrieving relevant information from these documents while providing proper citations.

---

## Domain

Artificial Intelligence  
Natural Language Processing (NLP)  
Information Retrieval Systems

---

## Model Used

This system implements **Retrieval-Augmented Generation (RAG)** using:

Embedding Model:
- **SentenceTransformer (all-MiniLM-L6-v2)**

Generator Model (LLM):
- **OpenAI GPT-4o-mini**

The embedding model converts text into vector representations while the LLM generates context-aware answers based on retrieved document chunks.

---

## System Architecture

Research PDFs
↓
Text Extraction (PyPDF)
↓
Document Chunking
↓
Embedding Generation (SentenceTransformer)
↓
Vector Storage (FAISS)
↓
User Query
↓
Query Embedding
↓
Similarity Search
↓
Relevant Document Retrieval
↓
LLM Generation (OpenAI GPT)
↓
Final Answer + Citations


---

## Tech Stack

Programming Language:
- Python

Libraries & Frameworks:
- SentenceTransformers
- FAISS (Vector Database)
- PyPDF
- NumPy
- OpenAI API

Concepts Implemented:
- Embeddings
- Vector Representations
- Document Chunking
- Semantic Search
- Similarity Search
- Retrieval-Augmented Generation (RAG)

---

## Features

- Upload multiple research PDFs
- Convert documents into embeddings
- Store embeddings in FAISS vector database
- Perform semantic search instead of keyword search
- Retrieve relevant document chunks
- Generate accurate answers using LLM
- Provide source citations

---

## Example Output

### Query Example


---

## Tech Stack

Programming Language:
- Python

Libraries & Frameworks:
- SentenceTransformers
- FAISS (Vector Database)
- PyPDF
- NumPy
- OpenAI API

Concepts Implemented:
- Embeddings
- Vector Representations
- Document Chunking
- Semantic Search
- Similarity Search
- Retrieval-Augmented Generation (RAG)

---

## Features

- Upload multiple research PDFs
- Convert documents into embeddings
- Store embeddings in FAISS vector database
- Perform semantic search instead of keyword search
- Retrieve relevant document chunks
- Generate accurate answers using LLM
- Provide source citations

---

## Example Output

### Query Example
User Question:
What is deep learning?

### AI Response
Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn complex patterns from data.

Source:
deep_learning_sample.pdf (Page 1)


---

## Output Screenshots

### Query Example

![Output 1](outputs/output1.png)

### AI Generated Response

![Output 2](outputs/output2.png)

---

## Installation

Clone the repository:

git clone https://github.com/Har-iii13/Personal-Research-AI-RAG.git
git clone https://github.com/yourusername/Personal-Research-AI-RAG.git
pip install -r requirements.txt


Run document ingestion:


python ingest.py


Start the research assistant:


python query.py


---


---

## Future Improvements

- Web interface using Streamlit
- Conversation memory layer
- Support for larger document collections
- Hybrid search (keyword + semantic)

---

## Conclusion

The project demonstrates how **Retrieval-Augmented Generation (RAG)** can be used to build a knowledge-aware AI system capable of answering research queries using multiple documents while providing reliable citations.




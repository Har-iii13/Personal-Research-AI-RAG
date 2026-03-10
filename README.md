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

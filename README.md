# RAG Application: Smart Assistant for Technical Documentation

## Vision
To simplify complex technical content for users by providing an intelligent, fast, and scalable solution for documentation search and query resolution.

## Overview
This application leverages a **Retrieval-Augmented Generation (RAG)** architecture to provide precise, context-aware answers to user queries. It is trained on the "Attention is All You Need" research paper, making it a powerful assistant for understanding technical content.

---

## How It Works
1. **Document Loading**: The application loads the "Attention is All You Need" paper using `PyPDFLoader`.
2. **Chunking**: The document is split into smaller chunks to enable granular search and response generation.
3. **Embedding**: Each chunk is embedded using `OllamaEmbeddings`, capturing semantic context.
4. **Storage**: The embeddings are stored in a **FAISS database**, enabling efficient retrieval.
5. **Query Resolution**:
   - User queries are passed through a **retrieval chain**.
   - Relevant chunks are fetched from the database.
   - **Ollama LLM** processes the retrieved chunks to generate an accurate, context-aware response.
6. **LangSmith**: For monitoring purposes
7. **Frontend**: The application is served through a **Streamlit app**, providing a seamless user experience.

---

## Why It’s Awesome
- **Fast**: Retrieves and processes responses in real-time.
- **Accurate**: Leverages state-of-the-art embeddings and LLMs to ensure precision.
- **User-Friendly**: Offers an intuitive interface for non-technical users.
- **Scalable**: Can be adapted to other documentation or datasets with ease.

---

## Example Use Case
### Scenario
A user queries:  
> What does self-attention mean in the Transformer?

### Application Response
Retrieves the relevant section from the "Attention is All You Need" paper and provides a concise, context-aware explanation.

---

## Getting Started
1. Clone the repository.
2. Install the required dependencies.
3. Place your PDF document in the specified directory.
4. Run the Streamlit app to start querying.

---

## Demo
Demo of the application available in Github repository

